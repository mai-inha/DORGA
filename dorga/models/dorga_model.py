import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dorga.models.patch_importance import ConvPatchImportance, Pooler_Box
from dorga.models.pattern_prior import PatternPredictor, DynamicPrior
from dorga.models.projectors import SharedProjector, ROISpecificProjector
from dorga.models.graph_attention import DecoupledOrdinalGraphAttentionNet
from dorga.models.classifier import AngularClassifier


class BrixiaViT512Dynamic(nn.Module):
    def __init__(
        self,
        vit,
        num_regions: int = 6,
        num_classes: int = 4,
        num_patterns: int = 6,
        pool_mode="PC",
        proj_dim: int = 768,
        pi_global: torch.Tensor = None,
        pi_patterns: torch.Tensor = None,
        gnn_num_heads: int = 4,
        gnn_dropout: float = 0.1,
    ):
        super().__init__()
        self.vit = vit
        self.R = num_regions
        self.C = num_classes
        self.K = num_patterns
        self.D_vit = vit.embed_dim
        self.pool_mode = pool_mode
        self.proj_dim = proj_dim

        self.register_buffer("pi_patterns", pi_patterns)
        self.register_buffer("pi_global", pi_global)

        img_size = vit.patch_embed.img_size[0] if isinstance(vit.patch_embed.img_size, (list, tuple)) else vit.patch_embed.img_size
        patch_size = vit.patch_embed.patch_size[0] if isinstance(vit.patch_embed.patch_size, (list, tuple)) else vit.patch_embed.patch_size

        self.patch_importance = ConvPatchImportance(grid=img_size // patch_size, embed_dim=self.D_vit)
        self.pooler = Pooler_Box(img_size=img_size, patch_size=patch_size, num_regions=num_regions)
        self.pattern_predictor = PatternPredictor(in_dim=self.D_vit, num_patterns=num_patterns, num_regions=num_regions, num_classes=num_classes)
        self.dynamic_prior = DynamicPrior(self.pi_global, self.pi_patterns, smoothing=0.05)

        self.shared_projector = SharedProjector(in_dim=self.D_vit, out_dim=proj_dim)
        self.roi_specific_projector = ROISpecificProjector(in_dim=self.D_vit, out_dim=proj_dim, num_regions=num_regions)

        init_anchors = self._init_anchors_geodesic(num_classes, proj_dim)
        self.class_anchors = nn.Parameter(init_anchors)
        self.cls_head = AngularClassifier()

        self.gnn = DecoupledOrdinalGraphAttentionNet(
            dim=proj_dim, num_classes=num_classes,
            num_regions=num_regions, cls_head=self.cls_head,
        )

    def _init_anchors_geodesic(self, num_classes, dim):
        angles = torch.linspace(0, math.pi, num_classes)
        anchors = torch.zeros(num_classes, dim)
        anchors[:, 0] = torch.cos(angles)
        anchors[:, 1] = torch.sin(angles)
        return anchors

    def _forward_tokens(self, x):
        vit = self.vit
        B = x.shape[0]
        x = vit.patch_embed(x)
        cls = vit.cls_token.expand(B, -1, -1)
        if getattr(vit, "pos_embed", None) is not None:
            x = x + vit.pos_embed[:, 1:1 + x.shape[1]]
            cls = cls + vit.pos_embed[:, :1]
        x = vit.pos_drop(torch.cat([cls, x], dim=1))
        for blk in vit.blocks:
            x = blk(x)
        return vit.norm(x)

    def forward(self, x, rel, masks=None, labels=None) -> Dict:
        seq = self._forward_tokens(x)
        cls_tok, patch_tok = seq[:, 0, :], seq[:, 1:, :]

        s_patch, entropy = self.patch_importance(patch_tok)
        z, w, w_base = self.pooler(rel=rel, patch_tokens=patch_tok, masks=masks, mode="PC", s_patch=s_patch)

        u = self.shared_projector(z)
        v = self.roi_specific_projector(z)
        logits_s2 = self.cls_head(u, self.class_anchors, stage="s2")
        rho = self.pattern_predictor(cls_tok.detach(), logits_s2.detach())
        pi_prior = self.dynamic_prior(rho)
        if isinstance(pi_prior, torch.Tensor):
            pi_prior = pi_prior.to(rho.device)

        gnn_out = self.gnn(u=u, v=v, rho=rho, pi_prior=pi_prior, class_anchors=self.class_anchors, labels=labels)
        h = gnn_out["h"]
        logits_s3 = self.cls_head(h, self.class_anchors, stage="s3")

        return {
            "logits_s2": logits_s2,
            "logits_s3": logits_s3,
            "rho": rho,
            "cls_tok": cls_tok,
            "z": z, "u": u, "v": v, "w": w, "h": h,
            "w_base": w_base,
            "entropy": entropy,
            "gnn_out": gnn_out,
            "pi_prior": pi_prior,
        }
