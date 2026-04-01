import pdb
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone as encoder
from . import decoder
from .GNN import CoocGNN

"""
Borrow from https://github.com/facebookresearch/dino
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthConsistencyLoss(nn.Module):
    """
    基于深度信息的结构化正则化损失 L_depth
    鼓励CAM在深度平坦区域保持一致，在深度边界处产生清晰变化。
    """

    def __init__(self, sigma_s=5, sigma_d=0.1, kernel_size=3, padding=1):
        super(DepthConsistencyLoss, self).__init__()
        # 空间带宽 sigma_s：控制空间邻域的范围
        self.sigma_s = sigma_s
        # 深度带宽 sigma_d：控制深度相似度的敏感度
        self.sigma_d = sigma_d

        # 局部窗口设置 (例如 3x3 窗口)
        self.kernel_size = kernel_size
        self.padding = padding

        # 预计算空间距离权重 (W_spatial)
        self.W_spatial = self._precompute_spatial_weights()

    def _precompute_spatial_weights(self):
        """
        计算局部窗口内像素之间的空间距离权重 W_spatial。
        """
        # 生成 (K*K) 个点的坐标网格
        coords = torch.arange(self.kernel_size ** 2).float()

        # 将一维索引转换为 (x, y) 坐标，中心点为 (K//2, K//2)
        k_half = self.kernel_size // 2

        y_coords = coords // self.kernel_size - k_half
        x_coords = coords % self.kernel_size - k_half

        # 计算空间距离的平方
        dist_sq = x_coords ** 2 + y_coords ** 2

        # 应用高斯核
        W_spatial = torch.exp(-dist_sq / (2 * self.sigma_s ** 2))

        # 将其重塑为卷积核的形状 (K, K)
        # 注意：这里我们使用一个 (1, 1, K, K) 的张量，方便后续使用 unfold 或 conv
        W_spatial = W_spatial.view(1, 1, self.kernel_size, self.kernel_size)
        return W_spatial

    def forward(self, cam_map, depth_map):
        """
        Args:
            cam_map (Tensor): 网络的CAM输出，形状 (N, C, H, W)
            depth_map (Tensor): 深度图，形状 (N, 1, H, W)，需要归一化到 [0, 1]

        Returns:
            Tensor: 深度一致性损失 L_depth
        """
        N, C, H, W = cam_map.shape

        # 1. 使用 nn.Unfold/Fold (或卷积) 来提取局部邻域：
        #    我们将 CAM 和 Depth 展开，以便计算中心像素与其邻域像素的差异。

        # 展开 CAM: shape (N, C * K*K, L) L = H*W (如果 stride=1, padding=1)
        cam_unfold = F.unfold(cam_map, self.kernel_size, padding=self.padding)
        # 展开 Depth: shape (N, 1 * K*K, L)
        depth_unfold = F.unfold(depth_map, self.kernel_size, padding=self.padding)

        # 提取中心像素 (Center Pixel) 的 CAM 和 Depth 值 (形状 (N, C, L) 和 (N, 1, L))
        # 假设 K=3, K*K=9。中心像素在第 4 维 (0-indexed)。
        center_idx = self.kernel_size ** 2 // 2

        cam_center = cam_unfold[:, center_idx * C: (center_idx + 1) * C, :]
        depth_center = depth_unfold[:, center_idx, :].unsqueeze(1)  # (N, 1, L)

        # 2. 计算差异项和权重项

        # a. CAM 差异项: |M_CAM(i) - M_CAM(j)|^2
        # (N, C * K*K, L) - (N, C, L) -> 扩展到 (N, C * K*K, L)
        cam_center_expanded = cam_center.repeat(1, self.kernel_size ** 2, 1)
        diff_cam_sq = (cam_unfold - cam_center_expanded) ** 2

        # b. 深度权重项 W_depth: exp(-|D(i) - D(j)|^2 / 2*sigma_d^2)
        # (N, K*K, L) - (N, 1, L) -> 扩展到 (N, K*K, L)
        depth_center_expanded = depth_center.repeat(1, self.kernel_size ** 2, 1)
        diff_depth_sq = (depth_unfold - depth_center_expanded) ** 2
        W_depth = torch.exp(-diff_depth_sq / (2 * self.sigma_d ** 2))

        # 3. 结合权重 W = W_spatial * W_depth

        # W_spatial 权重 (K*K,)
        W_spatial_flat = self.W_spatial.squeeze().to(cam_map.device)
        # 将空间权重扩展到 (N, K*K, L)
        W_spatial_expanded = W_spatial_flat.view(1, -1, 1).repeat(N, 1, cam_unfold.size(2))

        # 总权重 W (N, K*K, L)
        W = W_spatial_expanded * W_depth

        # 扩展权重以匹配 CAM 差异项的通道数 (N, C * K*K, L)
        W_expanded = W.repeat(1, C, 1)

        # 4. 计算 L_depth
        # L_depth = SUM (W * diff_cam_sq)
        L_depth_per_pixel = W_expanded * diff_cam_sq

        # 对所有邻域、所有通道求和，并取平均
        L_depth = L_depth_per_pixel.sum(dim=1).mean()

        return L_depth

class IncrementalClassifier(nn.ModuleList):
    def forward(self, input):
        out = []
        for mod in self:
            out.append(mod(input))
        sem_logits = torch.cat(out, dim=1)
        return sem_logits

    def get_weight(self):
        return torch.cat([mod.weight for mod in self], dim=0)


import torch
import torch.nn as nn
import torch.nn.functional as F


# 假设 encoder 和 decoder 是你项目中已经存在的模块
# from models import encoder, decoder

class IncrementalClassifier(nn.ModuleList):
    def forward(self, input):
        out = []
        for mod in self:
            out.append(mod(input))
        sem_logits = torch.cat(out, dim=1)
        return sem_logits

    def get_weight(self):
        return torch.cat([mod.weight for mod in self], dim=0)


class IncrementalPrototype(nn.Module):
    """
    Incrementally adds prototype tensors for new classes.
    prototypes_per_task: a list like [C0, C1, C2, ...]
    """

    def __init__(self, prototypes_per_task, feat_dim, init='normal', num_future=3):
        super().__init__()
        self.init = init
        self.feat_dim = feat_dim
        self.num_future = num_future  # [新增] 未来类别数量

        self.prototype = nn.ParameterList()

        # [新增] 3个未来原型，用于捕捉背景中的潜在物体
        # Shape: [num_future, feat_dim]
        self.future_prototypes = nn.Parameter(self._init_prototype(self.num_future))

        for idx, c in enumerate(prototypes_per_task):
            # c classes for this task
            if idx == 0:
                proto = nn.Parameter(self._init_prototype(c))
            else:
                proto = nn.Parameter(self._init_prototype(c))
            self.prototype.append(proto)

    def _init_prototype(self, num_classes):
        """Return (num_classes, feat_dim) prototype matrix."""
        if self.init == 'normal':
            return torch.randn(num_classes, self.feat_dim) * 0.01

        elif self.init == 'xavier':
            w = torch.empty(num_classes, self.feat_dim)
            nn.init.xavier_normal_(w)
            return w

        elif self.init == 'zeros':
            return torch.zeros(num_classes, self.feat_dim)

        else:
            raise ValueError(f"Unknown prototype init: {self.init}")

    def forward(self, with_future=False):
        """
        Return a concatenated prototype matrix.
        Shape: (Total_classes [+ num_future], feat_dim)
        """
        current_protos = torch.cat(list(self.prototype), dim=0)

        if with_future:
            # [修改] 拼接未来原型
            return torch.cat([current_protos, self.future_prototypes], dim=0)
        else:
            return current_protos

    @torch.no_grad()
    def init_new_classes_with_future(self, previous_future_weights):
        """
        [新增] 在增量 Step 开始时调用。
        使用上一轮训练好的未来原型来初始化当前新加入的任务原型。
        """
        if len(self.prototype) <= 1:
            print("Warning: No new task to initialize (step 0).")
            return

        # 获取当前最新加入的任务的原型参数
        new_task_proto_param = self.prototype[-1]
        num_new_classes = new_task_proto_param.shape[0]
        num_future_avail = previous_future_weights.shape[0]

        print(f"Initializing {num_new_classes} new classes using {num_future_avail} future prototypes...")

        # 简单的分配策略：
        if num_new_classes <= num_future_avail:
            # 如果新类少于未来原型，直接取前 N 个
            new_task_proto_param.data.copy_(previous_future_weights[:num_new_classes])
        else:
            # 如果新类多于未来原型，循环填充
            repeat_times = (num_new_classes // num_future_avail) + 1
            extended_weights = previous_future_weights.repeat(repeat_times, 1)
            new_task_proto_param.data.copy_(extended_weights[:num_new_classes])

        print("Initialization complete.")


class ResidualPrototypeModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.res_adapter = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )
        nn.init.zeros_(self.res_adapter[-1].weight)
        nn.init.zeros_(self.res_adapter[-1].bias)
        self.dim = dim

    def forward(self, fmap, global_prototypes):
        B, D, H, W = fmap.shape
        C = global_prototypes.shape[0]

        img_context = F.adaptive_avg_pool2d(fmap, (1, 1)).flatten(1)
        img_expanded = img_context.unsqueeze(1).expand(-1, C, -1)
        proto_expanded = global_prototypes.unsqueeze(0).expand(B, -1, -1)
        combined_feat = torch.cat([img_expanded, proto_expanded], dim=-1)

        delta_p = self.res_adapter(combined_feat)
        p_final = proto_expanded + delta_p

        p_final_norm = F.normalize(p_final, dim=2)
        fmap_norm = F.normalize(fmap.flatten(2), dim=1)
        logits = torch.bmm(p_final_norm, fmap_norm).view(B, C, H, W)

        return logits, delta_p, p_final


class ResidualPrototypeModel_cross(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()

        # Cross-Attention Adapter
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.output_scale = nn.Parameter(torch.zeros(1))

    def forward(self, fmap, global_prototypes):
        """
        fmap: [B, D, H, W]
        global_prototypes: [C_total + Future, D]
        """
        B, D, H, W = fmap.shape
        # C 自动根据传入的原型数量变化 (包含 Future)
        C = global_prototypes.shape[0]

        img_feats = fmap.flatten(2).transpose(1, 2)
        proto_query = global_prototypes.unsqueeze(0).expand(B, -1, -1)

        # Cross-Attention: Image -> Prototype
        attn_out, _ = self.cross_attn(
            query=proto_query,
            key=img_feats,
            value=img_feats
        )

        delta = self.norm1(attn_out)
        delta = self.norm2(delta + self.ffn(delta))

        delta_p = delta * self.output_scale

        p_final = proto_query + delta_p

        p_final_norm = F.normalize(p_final, dim=2)
        fmap_norm = F.normalize(img_feats, dim=2).transpose(1, 2)
        logits = torch.bmm(p_final_norm, fmap_norm).view(B, C, H, W)

        return logits, delta_p, p_final


class network(nn.Module):
    def __init__(self, backbone, num_classes=None, classes_list=None, pretrained=None, init_momentum=None,
                 aux_layer=None, step=0):
        super().__init__()
        self.num_classes = num_classes
        self.classes_list = classes_list
        self.init_momentum = init_momentum

        # 假设 encoder 在外部定义
        self.encoder = getattr(encoder, backbone)(pretrained=pretrained, aux_layer=aux_layer, step=step)

        self.in_channels = [self.encoder.embed_dim] * 4 if hasattr(self.encoder, "embed_dim") else [
                                                                                                       self.encoder.embed_dims[
                                                                                                           -1]
                                                                                                   ] * 4

        self.pooling = F.adaptive_max_pool2d

        # 假设 decoder 在外部定义
        self.decoder = decoder.LargeFOV(in_planes=self.in_channels[-1], out_planes=self.num_classes, dilation=5,
                                        classes_list=classes_list)

        self.classifier = IncrementalClassifier(
            [nn.Conv2d(in_channels=self.in_channels[-1], out_channels=(c - 1) if idx == 0 else c, kernel_size=1,
                       bias=False)
             for idx, c in enumerate(self.classes_list)]
        )
        self.aux_classifier = IncrementalClassifier(
            [nn.Conv2d(in_channels=self.in_channels[-1], out_channels=(c - 1) if idx == 0 else c, kernel_size=1,
                       bias=False)
             for idx, c in enumerate(self.classes_list)]
        )

        # [修改] 传入 num_future=3
        self.prototype_module = IncrementalPrototype(
            prototypes_per_task=self.classes_list,
            feat_dim=768,
            init='xavier',
            num_future=3
        )

        self.res_model = ResidualPrototypeModel_cross(768)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_param_groups(self):
        param_groups = [[], [], [], []]

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for classifier in self.classifier:
            param_groups[2].append(classifier.weight)
        for classifier in self.aux_classifier:
            param_groups[2].append(classifier.weight)

        # prototype 除了列表里的，还要把 future_prototypes 加进去
        for prototype in self.prototype_module.prototype:
            param_groups[2].append(prototype)
        param_groups[2].append(self.prototype_module.future_prototypes)  # [新增] 加入优化器

        for param in list(self.res_model.parameters()):
            param_groups[2].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def to_2D(self, x, h, w):
        n, hw, c = x.shape
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, x, depth, cam_only=False, cam_grad=False):
        cls_token, _x, x_aux = self.encoder.forward_features(x, depth)

        h, w = x.shape[-2] // self.encoder.patch_size, x.shape[-1] // self.encoder.patch_size
        _x4 = self.to_2D(_x, h, w)  # [B, F, H, W]
        _x_aux = self.to_2D(x_aux, h, w)

        seg = self.decoder(_x4)

        if cam_only:
            cam = F.conv2d(_x4, self.classifier.get_weight()).detach()
            cam_aux = F.conv2d(_x_aux, self.aux_classifier.get_weight()).detach()
            return cam_aux, cam

        cls_aux = self.pooling(_x_aux, (1, 1))
        cls_aux = self.aux_classifier(cls_aux)

        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes - 1)
        cls_aux = cls_aux.view(-1, self.num_classes - 1)

        # --- prototype segmentation head ---

        # [修改] 开启 with_future=True
        # P 的形状现在是: [Total_Known_Classes + 3, F]
        P = self.prototype_module(with_future=True)

        # res_model 会自动处理多出来的 3 个 token
        # seg_proto 输出形状: [B, Total_Known + 3, H, W]
        # 倒数第 1,2,3 个通道就是那 3 个未来原型的预测热力图
        seg_proto, delta_p, p_final = self.res_model(_x4, P)

        if cam_grad:
            cam_grad_map = F.conv2d(_x4, self.classifier.get_weight())
            return cam_grad_map, cls_x4, seg, _x4, cls_aux, seg_proto, P, delta_p, p_final

        return cls_x4, seg, _x4, cls_aux, seg_proto, P, delta_p, p_final



