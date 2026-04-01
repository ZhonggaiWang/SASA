from __future__ import annotations
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from click.core import batch


# ------------------------------
# Utils
# ------------------------------

def masked_bce_with_logits(pred: torch.Tensor, target: torch.Tensor, ignore_val: float = -1.0,
                           reduction: str = "mean") -> torch.Tensor:
    """BCEWithLogitsLoss with ignore label support.
    pred, target: [B, C]; any target == ignore_val is ignored.
    """
    assert pred.shape == target.shape
    mask = (target != ignore_val).float()
    safe_target = torch.where(mask > 0, target, torch.zeros_like(target))
    loss = F.binary_cross_entropy_with_logits(pred, safe_target, reduction='none')
    if reduction == 'none':
        return loss * mask
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def symmetric_zero_diag(m: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (m + m.transpose(-1, -2))
    m = m - torch.diag_embed(torch.diagonal(m, dim1=-2, dim2=-1))
    return m


def normalize_adjacency(w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-normalize adjacency for message passing."""
    d = w.sum(-1, keepdim=True).clamp_min(eps)
    return w / d


def pairwise_cosine(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x: [C, F] -> cosine similarity matrix [C, C]."""
    x = F.normalize(x, dim=-1, eps=eps)
    return x @ x.t()


@torch.no_grad()
def to_pmi(joint: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    joint: [C, C] 非负（co-occur 计数或权重）。
    正确的 PMI 计算：用行/列和作为边际，而不是对角线。
    """
    # 归一化到联合分布
    total = joint.sum().clamp_min(eps)
    pxy = joint / total                         # [C, C]
    # 边际分布（行/列和）
    px = pxy.sum(dim=1, keepdim=True)           # [C, 1] = ∑_j pxy[i,j]
    py = pxy.sum(dim=0, keepdim=True)           # [1, C] = ∑_i pxy[i,j]
    # 期望独立下的联合
    denom = (px @ py).clamp_min(eps)            # [C, C]
    # PMI
    pmi = torch.log((pxy + eps) / denom)
    # 规整：对角置 0、对称化
    pmi.fill_diagonal_(0.0)
    pmi = 0.5 * (pmi + pmi.t())
    return pmi


@torch.no_grad()
def to_ppmi(joint: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """正PMI：= max(PMI, 0)。"""
    return to_pmi(joint, eps=eps).clamp_min(0.0)


def pmi_to_target(pmi: torch.Tensor, temp: float) -> torch.Tensor:
    """PMI → (0,1) 软目标；负相关<0.5，正相关>0.5。"""
    return torch.sigmoid(pmi / max(float(temp), 1e-6))


def min_max_norm(x: torch.Tensor, dim: Optional[int] = None, eps: float = 1e-6) -> torch.Tensor:
    if dim is None:
        x_min = x.amin()
        x_max = x.amax()
    else:
        x_min = x.amin(dim=dim, keepdim=True)
        x_max = x.amax(dim=dim, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def F_adaptive_avg_pool2d(x: torch.Tensor, out_size: int) -> torch.Tensor:
    return F.adaptive_avg_pool2d(x, out_size)


# ------------------------------
# Co-occurrence GNN
# ------------------------------

class CoocGNN(nn.Module):
    def __init__(self,
                 num_classes: int,
                 feat_channels: int,
                 hidden_dim: int = 256,
                 k_steps: int = 1,
                 edge_hidden: int = 64,
                 beta_pos: float = 0.5,
                 gamma_neg: float = 0.25,
                 edge_target_blend: float = 0.5,   # α: prior/batch 融合
                 ):
        """
        Args:
            num_classes: number of foreground classes (exclude background).
            feat_channels: channels of backbone feature map used for CAM.
            hidden_dim: node embedding dimension.
            k_steps: GNN message passing steps.
            edge_hidden: hidden size of edge MLP.
            beta_pos: strength for positive neighbor aggregation in logit refinement.
            gamma_neg: strength for negative (mutual exclusion) suppression.
            edge_target_blend: α for target_pmi = prior + α*(batch - prior)
        """
        super().__init__()
        C, F = num_classes, feat_channels
        self.C, self.F = C, F
        self.k = k_steps
        self.beta_pos = beta_pos
        self.gamma_neg = gamma_neg
        self.edge_target_blend = edge_target_blend

        # Project class prototypes F -> D
        self.proto_proj = nn.Sequential(
            nn.Linear(F, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Message passing transform
        self.msg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Edge descriptor from node embeddings (avoid feeding PMI as input)
        self.edge_desc_dim = 64
        self.edge_desc = nn.Sequential(
            nn.Linear(hidden_dim, self.edge_desc_dim),
            nn.ReLU(inplace=True),
        )
        edge_in_dim = 4 * self.edge_desc_dim + 3  # [zi, zj, |zi-zj|, zi* zj] + [cos, pi, pj]
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        nn.init.constant_(self.edge_mlp[-1].bias, -2.0)  # sparse start

        # Decode node embeddings to per-class channel weights (alpha >= 0)
        self.alpha_head = nn.Linear(hidden_dim, F)
        # Per-class logit bias from node embedding
        self.bias_head = nn.Linear(hidden_dim, 1)

        # Temperature for squeezing PMI to [0,1] target
        self.register_buffer('pmi_temp', torch.tensor(2.5))

    def _edge_feats_from_feats(self, Hn: torch.Tensor, cos: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """
        Hn: [C, D]   projected prototypes
        cos: [C, C]
        freq: [C]
        return: [C, C, 4E+3]
        """
        C, D = Hn.shape
        Z = self.edge_desc(Hn)               # [C, E]
        E = Z.size(1)
        zi = Z[:, None, :].expand(C, C, E)
        zj = Z[None, :, :].expand(C, C, E)
        pair = torch.cat([zi, zj, (zi - zj).abs(), zi * zj], dim=-1)  # [C,C,4E]

        pi = freq.view(C, 1).expand(C, C)
        pj = freq.view(1, C).expand(C, C)
        scalars = torch.stack([cos, pi, pj], dim=-1)                   # [C,C,3]
        feats = torch.cat([pair, scalars], dim=-1)                     # [C,C,4E+3]
        return feats

    def forward(self,
                feats: torch.Tensor,         # [B, F, H, W]
                cls_logits: torch.Tensor,    # [B, C]
                img_labels: torch.Tensor,    # [B, C] in {0,1,-1}
                prior_pmi: torch.Tensor,     # [C, C] (dataset PMI)
                class_mask: Optional[torch.Tensor] = None,  # [C] 1=active, for incremental
                ) -> Dict[str, torch.Tensor]:

        B, C_feat, H, W = feats.shape
        C = self.C
        assert C == cls_logits.size(1)

        # 1) probs / prototypes (use probs for features; labels for batch PMI teacher)
        probs = torch.sigmoid(cls_logits)                # [B, C]
        pooled = F_adaptive_avg_pool2d(feats, 1).flatten(1)  # [B, F]

        with torch.no_grad():
            weight_sum = probs.sum(0).clamp_min(1e-6)
            proto = (probs.t() @ pooled) / weight_sum.unsqueeze(-1)    # [C, F]
        cos = pairwise_cosine(proto).clamp(-1, 1)                       # [C, C]
        with torch.no_grad():
            freq = probs.mean(0)                                        # [C]

        # 2) batch PMI from labels (more stable on small batch)
        y = img_labels.float().clamp_min(0)  # [-1,0,1] → [0,0,1]
        joint = y.t() @ y  # [C,C] 批内共现计数
        joint = joint + 0.01  # 轻拉普拉斯，建议 laplace=0.1~0.5
        batch_pmi = to_pmi(joint)

        # 3) target PMI = prior + α*(batch - prior)
        alpha_blend = self.edge_target_blend
        target_pmi = prior_pmi  # [C, C]
        t = pmi_to_target(target_pmi, temp=float(self.pmi_temp))

        # 4) Edge logits = (frozen baseline) + residual(features)
        Hn = self.proto_proj(proto)                                     # [C, D]
        edge_feats = self._edge_feats_from_feats(Hn, cos, freq)         # [C,C,4E+3]
        h0 = (target_pmi / self.pmi_temp).detach()                      # baseline logits (no grad)
        r  = self.edge_mlp(edge_feats).squeeze(-1)                      # learnable residual
        h  = h0 + r                                                     # final logits
        W_adj = torch.sigmoid(h)
        W_adj = symmetric_zero_diag(W_adj)

        # Masks for supervision/usage
        edge_mask = torch.ones_like(W_adj, dtype=torch.bool)
        edge_mask.fill_diagonal_(False)
        if class_mask is not None:
            cm = class_mask.bool()
            edge_mask &= (cm.view(-1, 1) & cm.view(1, -1))
        present = (y.max(0).values > 0)                                 # any positive label in batch
        edge_mask &= (present.view(-1, 1) & present.view(1, -1))

        # 5) Message passing over class graph
        A = normalize_adjacency(W_adj)
        Z = Hn
        for _ in range(self.k):
            msg = self.msg(A @ Z)
            Z = F.relu(Z + msg, inplace=True)

        # 6) Decode to CAM channel weights + per-class bias
        alpha_w = F.softplus(self.alpha_head(Z))                        # [C, F], >=0
        alpha_w = alpha_w / alpha_w.sum(-1, keepdim=True).clamp_min(1e-6)
        dlog = self.bias_head(Z).squeeze(-1)                            # [C]

        # 7) Logit refinement with per-image gating
        gate = y[:, :, None] * y[:, None, :]                            # [B,C,C], 0/1
        W_img = W_adj.unsqueeze(0) * gate
        pos_agg = torch.einsum('bc,bcd->bd', probs, W_img)              # [B,C]
        neg_agg = torch.einsum('bc,bcd->bd', probs, 1.0 - W_img)        # [B,C]
        refined_logits = cls_logits + self.beta_pos * pos_agg - self.gamma_neg * neg_agg + dlog

        # 8) CAM from alpha_w
        cam = torch.einsum('cf,bfhw->bchw', alpha_w, feats)
        cam = F.relu(cam)
        cam_vis = min_max_norm(cam.flatten(2), dim=-1).view(B, C, H, W)  # for visualization/labeling

        # 9) Losses
        # (a) classification (image-level; ignore=-1)
        cls_loss = masked_bce_with_logits(refined_logits, img_labels)

        # (b) edge supervision in logits space (+ residual regularization + light sparsity)
        # positive/negative reweight (pos edges are rarer)
        pos = (t > 0.5)
        neg = ~pos
        n_pos = pos[edge_mask].float().sum().clamp_min(1.0)
        n_neg = neg[edge_mask].float().sum().clamp_min(1.0)
        w_pos = (n_neg / n_pos).clamp(1.0, 10.0).detach()
        weight = torch.where(pos, w_pos, torch.ones_like(t))
        edge_loss = F.binary_cross_entropy_with_logits(h[edge_mask], t[edge_mask],
                                                       weight=weight[edge_mask])
        r_reg = 1e-3 * r[edge_mask].abs().mean()
        sparsity = W_adj.mean()

        total_loss = cls_loss + 0.1 * edge_loss + r_reg + 0.01 * sparsity

        return {
            'W': W_adj,
            'alpha': alpha_w,
            'dlog': dlog,
            'cam': cam_vis,
            'refined_logits': refined_logits,
            'loss': total_loss,
            'cls_loss': cls_loss.detach(),
            'edge_loss': edge_loss.detach(),
            'sparsity': sparsity.detach(),
        }
