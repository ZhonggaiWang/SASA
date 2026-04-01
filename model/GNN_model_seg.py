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


class IncrementalClassifier(nn.ModuleList):
    def forward(self, input):
        out = []
        for mod in self:
            out.append(mod(input))
        sem_logits = torch.cat(out, dim=1)
        return sem_logits

    def get_weight(self):
        return torch.cat([mod.weight for mod in self], dim=0)

# ------------------------------
# Network with controlled residual fusion
# ------------------------------

class network(nn.Module):
    def __init__(self, backbone, num_classes=None, classes_list=None, pretrained=None, init_momentum=None,
                 aux_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.classes_list = classes_list
        self.init_momentum = init_momentum

        self.encoder = getattr(encoder, backbone)(pretrained=pretrained, aux_layer=aux_layer)

        self.in_channels = [self.encoder.embed_dim] * 4 if hasattr(self.encoder, "embed_dim") else [
            self.encoder.embed_dims[-1]
        ] * 4

        self.pooling = F.adaptive_max_pool2d

        self.decoder = decoder.LargeFOV(in_planes=self.in_channels[-1], out_planes=self.num_classes, dilation=5,
                                        classes_list=classes_list)

        # Canvas GNN (hidden_dim/k_steps names)
        self.cooc_gnn = CoocGNN(self.num_classes - 1, self.in_channels[-1], hidden_dim=256, k_steps=1)
        self.register_buffer('A_prior', torch.eye(self.num_classes - 1))  # EMA prior (not used by canvas GNN directly)

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

        self._gnn_last_out: Optional[Dict[str, torch.Tensor]] = None

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_param_groups(self):
        param_groups = [[], [], [], [], []]  # backbone; backbone_norm; cls_head; seg_head; others

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for classifier in self.classifier:
            param_groups[2].append(classifier.weight)
        for classifier in self.aux_classifier:
            param_groups[2].append(classifier.weight)

        if hasattr(self, "cooc_gnn"):
            for p in self.cooc_gnn.parameters():
                if p.requires_grad:
                    param_groups[4].append(p)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def to_2D(self, x, h, w):
        n, hw, c = x.shape
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, x, cam_only=False, cam_grad=False, old_model=False,
                use_gnn=False, return_gnn=False,
                img_labels: Optional[torch.Tensor] = None,
                prior_pmi: Optional[torch.Tensor] = None,
                class_mask: Optional[torch.Tensor] = None):

        cls_token, _x, x_aux = self.encoder.forward_features(x)
        h, w = x.shape[-2] // self.encoder.patch_size, x.shape[-1] // self.encoder.patch_size

        _x4 = self.to_2D(_x, h, w)  # [B, F, H, W]
        _x_aux = self.to_2D(x_aux, h, w)
        seg = self.decoder(_x4)

        # --- CAM-only branch ---
        if cam_only:
            cam = F.conv2d(_x4, self.classifier.get_weight()).detach()
            cam_aux = F.conv2d(_x_aux, self.aux_classifier.get_weight()).detach()
            return cam_aux, cam

        # --- image-level classifier logits ---
        cls_aux = self.pooling(_x_aux, (1, 1))
        cls_aux = self.aux_classifier(cls_aux)

        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes - 1)  # foreground C
        cls_aux = cls_aux.view(-1, self.num_classes - 1)

        if old_model:
            # keep your original signature order for compatibility
            return cls_x4, cls_aux, seg, _x4

        # --- CoocGNN (canvas) with controlled residual fusion ---
        if use_gnn:
            C = self.num_classes - 1
            if (img_labels is None) or (prior_pmi is None):
                raise ValueError("use_gnn=True requires img_labels and prior_ppmi")

            gnn_out = self.cooc_gnn(
                feats=_x4,
                cls_logits=cls_x4,
                img_labels=img_labels[:, :C].float(),
                prior_pmi=prior_pmi,
                class_mask=class_mask,
            )

            if return_gnn:
                self._gnn_last_out = gnn_out

        if cam_grad:
            cam_grad_map = F.conv2d(_x4, self.classifier.get_weight())
            return cam_grad_map, cls_x4, seg, _x4, cls_aux, self._gnn_last_out

        return cls_x4, seg, _x4, cls_aux


