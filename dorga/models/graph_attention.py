import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledOrdinalGraphAttentionNet(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        num_regions: int = 6,
        cls_head: Optional[nn.Module] = None,
        num_heads: int = 4,
        step_size_init: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.D = dim
        self.C = num_classes
        self.R = num_regions
        self.cls_head = cls_head

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.att_scale = nn.Parameter(torch.tensor(10.0))


        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.att_dropout = nn.Dropout(dropout)

        self.step_size = nn.Parameter(torch.tensor(step_size_init))

    def _l2n(self, x, eps=1e-8):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    def _slerp(self, h, target, t, eps=1e-7):
        cos_theta = (h * target).sum(-1, keepdim=True).clamp(-1 + eps, 1 - eps)
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        is_close = (theta.abs() < eps) | (sin_theta.abs() < eps)
        coef_h = torch.sin((1 - t) * theta) / sin_theta.clamp_min(eps)
        coef_t = torch.sin(t * theta) / sin_theta.clamp_min(eps)
        out = coef_h * h + coef_t * target
        fallback = self._l2n((1 - t) * h + t * target)
        return self._l2n(torch.where(is_close, fallback, out))

    def forward(self, u, v, rho, pi_prior, class_anchors, labels=None, **kwargs):
        B, R, D = u.shape
        device = u.device
        C = self.C

        h = self._l2n(u)
        a_unit = self._l2n(class_anchors)

        with torch.no_grad():
            cos_sim = torch.einsum("brd,cd->brc", h, a_unit)
            p = F.one_hot(cos_sim.argmax(-1), C).float()


        pi_base = pi_prior.detach()
        pi_refined = pi_base
        eye_CC = torch.eye(C, device=device)
        diag_mask = torch.eye(R, device=device, dtype=torch.bool)
        pi_refined = pi_refined.clone()
        pi_refined[:, diag_mask] = eye_CC

        q_ij = torch.einsum("bijcd,bjd->bijc", pi_refined, p)

        q_attn = self.q_proj(v).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        k_attn = self.k_proj(v).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        attn_logits = (q_attn @ k_attn.transpose(-2, -1)) * self.att_scale

        alpha_heads = F.softmax(attn_logits, dim=-1)
        alpha_heads = self.att_dropout(alpha_heads)
        alpha = alpha_heads.mean(dim=1)

        q_bar = torch.einsum("bij,bijc->bic", alpha, q_ij)

        class_idx = torch.arange(C, device=device, dtype=q_bar.dtype)
        mu = (q_bar * class_idx).sum(-1)
        var = (q_bar * (class_idx - mu.unsqueeze(-1)).pow(2)).sum(-1)
        V_max = ((C - 1) ** 2) / 4.0
        xi = (1.0 - var / V_max).clamp(0, 1)

        A = torch.einsum("brc,cd->brd", q_bar, a_unit.detach())
        t_base = torch.sigmoid(self.step_size)
        t_eff = t_base * xi
        h_updated = self._slerp(h, self._l2n(A), t_eff)

        return {
            "h": h_updated,
            "rho": rho,
            "attached": {
                "alpha": [alpha],
                "q": [q_ij],
            },
            "raw": {
                "p": [p.detach()],
                "qbar": [q_bar.detach()],
                "pi_refined": [pi_refined.detach()],
                "pi_base": [pi_base.detach()],
                "alpha": [alpha.detach()],
                "alpha_per_head": [alpha_heads.detach()],
                "attn_logits": [attn_logits.detach()],
                "t_eff": [t_eff.detach()],
                "gate": [xi.detach()],
                "q_conf": [xi.detach()],
                "step": [t_base.detach()],
                "q": [q_ij.detach()],
            },
        }
