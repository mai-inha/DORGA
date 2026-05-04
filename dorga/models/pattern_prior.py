import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PatternPredictor(nn.Module):
    def __init__(self, in_dim: int, num_patterns: int, num_regions: int = 6, num_classes: int = 4):
        super().__init__()
        self.K = num_patterns
        feat_dim = in_dim + num_regions * num_classes

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, num_patterns),
        )

    def forward(self, cls_token: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        logits_flat = logits.detach().flatten(1)
        feat = torch.cat([cls_token, logits_flat], dim=-1)
        return F.softmax(self.mlp(feat), dim=-1)


def pattern_loss(rho: torch.Tensor, pattern_gt: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(rho.clamp_min(1e-12).log(), pattern_gt.long())


class DynamicPrior(nn.Module):

    def __init__(
        self,
        pi_global: torch.Tensor,
        pi_patterns: torch.Tensor,
        smoothing: float = 0.05,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("pi_global", pi_global)
        self.register_buffer("pi_patterns", pi_patterns)
        self.K = pi_patterns.shape[0]
        self.R = pi_patterns.shape[1]
        self.C = pi_patterns.shape[3]

        pi_patterns_smooth = (1 - smoothing) * pi_patterns + smoothing / self.C
        pi_global_smooth = (1 - smoothing) * pi_global + smoothing / self.C

        pi_patterns_smooth = pi_patterns_smooth / pi_patterns_smooth.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        pi_global_smooth = pi_global_smooth / pi_global_smooth.sum(dim=-2, keepdim=True).clamp(min=1e-8)

        self.register_buffer("pi_patterns", pi_patterns_smooth)
        self.register_buffer("pi_global", pi_global_smooth)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        pi_dyn = torch.einsum("bk,kijcd->bijcd", rho, self.pi_patterns)
        pi_dyn = pi_dyn / pi_dyn.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        return pi_dyn
