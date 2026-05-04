import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ConvPatchImportance(nn.Module):
    def __init__(self, grid: int, embed_dim: int, use_pos_bias: bool = True, temp_max: float = 5.0):
        super().__init__()
        self.grid = grid
        self.P = grid * grid
        self.norm = nn.LayerNorm(embed_dim)
        self.dwconv = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim)
        self.pwconv = nn.Conv2d(embed_dim, 1, 1)
        self.use_pos = use_pos_bias
        if use_pos_bias:
            self.pos_bias = nn.Parameter(torch.zeros(self.P))
        self.temperature = nn.Parameter(torch.tensor(2.0))
        self.temp_max = temp_max

    def forward(self, patch_tok: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, P, C = patch_tok.shape
        g = self.grid
        x = self.norm(patch_tok).view(B, g, g, C).permute(0, 3, 1, 2)
        x = F.gelu(self.dwconv(x))
        s = self.pwconv(x).flatten(2).squeeze(1)
        if self.use_pos:
            s = s + self.pos_bias.view(1, self.P).expand(B, -1)

        temp = self.temperature.clamp(min=0.1, max=self.temp_max)
        s_softmax = torch.softmax(s / temp, dim=1)
        entropy = -(s_softmax * (s_softmax + 1e-8).log()).sum(dim=1)
        s_scaled = s_softmax * float(self.P)

        return s_scaled, entropy

class Pooler_Box(nn.Module):
    def __init__(self, img_size: int = 512, patch_size: int = 16, num_regions: int = 6, eps: float = 1e-6):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_regions = num_regions
        self.eps = eps

        grid = img_size // patch_size
        self.grid = grid
        P = grid * grid

        patch_boxes = []
        for py in range(grid):
            for px in range(grid):
                y0, x0 = py * patch_size, px * patch_size
                patch_boxes.append((y0, x0, y0 + patch_size, x0 + patch_size))

        self.register_buffer("patch_boxes", torch.tensor(patch_boxes, dtype=torch.float32).view(P, 4), persistent=False)
        self.patch_area = patch_size * patch_size

    def _grid_gap_mask(self, B: int, device, dtype) -> torch.Tensor:
        g = self.grid
        P = g * g
        R = self.num_regions
        if R != 6:
            raise ValueError("GAP grid mode assumes num_regions=6 (2x3).")

        grid_h, grid_w = 3, 2

        def get_boundaries(total, divisions):
            base = total // divisions
            remainder = total % divisions
            boundaries = [0]
            for i in range(divisions):
                extra = 1 if i < remainder else 0
                boundaries.append(boundaries[-1] + base + extra)
            return boundaries

        y_bounds = get_boundaries(g, grid_h)
        x_bounds = get_boundaries(g, grid_w)

        mask = torch.zeros((R, P), device=device, dtype=dtype)

        for ry in range(grid_h):
            for rx in range(grid_w):
                r = rx * grid_h + ry 
                y0, y1 = y_bounds[ry], y_bounds[ry + 1]
                x0, x1 = x_bounds[rx], x_bounds[rx + 1]

                for y in range(y0, y1):
                    for x in range(x0, x1):
                        p = y * g + x
                        mask[r, p] = 1.0

        return mask.unsqueeze(0).expand(B, -1, -1)

    def _rel_to_pixel(self, rel: torch.Tensor) -> torch.Tensor:
        pix = rel * float(self.img_size)
        pix[..., 0::2] = pix[..., 0::2].clamp(0, self.img_size)
        pix[..., 1::2] = pix[..., 1::2].clamp(0, self.img_size)
        return pix

    def _weights_area(self, rel: torch.Tensor) -> torch.Tensor:
        rois_pix = self._rel_to_pixel(rel)
        patches = self.patch_boxes.view(1, 1, -1, 4)
        rois = rois_pix.unsqueeze(2)

        y0 = torch.maximum(rois[..., 0], patches[..., 0])
        x0 = torch.maximum(rois[..., 1], patches[..., 1])
        y1 = torch.minimum(rois[..., 2], patches[..., 2])
        x1 = torch.minimum(rois[..., 3], patches[..., 3])

        inter = (y1 - y0).clamp(min=0) * (x1 - x0).clamp(min=0)
        return inter / float(self.patch_area)
    
    def _weights_from_mask(self, masks: torch.Tensor) -> torch.Tensor:
        w_grid = F.adaptive_avg_pool2d(masks, output_size=(self.grid, self.grid))
        w_area = w_grid.flatten(2)
        return w_area

    def forward(self, rel, patch_tokens, *, masks=None, mode="PC", s_patch=None):
        B, P, C = patch_tokens.shape

        if mode == "GAP":
            w_base = self._grid_gap_mask(B, patch_tokens.device, patch_tokens.dtype)
        else:
            if masks is not None:
                w_area = self._weights_from_mask(masks)
            else:
                w_area = self._weights_area(rel)

            if mode == "Mask":
                base = torch.ones((B, P), device=patch_tokens.device, dtype=patch_tokens.dtype)
            elif mode == "PC":
                if s_patch is None:
                    raise ValueError("mode='PC' requires s_patch.")
                s = s_patch.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
                if torch.isnan(s).any() or torch.isinf(s).any():
                    s = torch.nan_to_num(s, nan=1.0, posinf=3.0, neginf=0.0)
                s_mean = s.mean(dim=1, keepdim=True).clamp_min(self.eps)
                base = (s / s_mean).clamp(min=0.0, max=3.0)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            w_base = w_area * base.unsqueeze(1)

        w_base = torch.nan_to_num(w_base, nan=0.0, posinf=1.0, neginf=0.0)
        w_sum = w_base.sum(dim=-1, keepdim=True)
        zero_mask = (w_sum < self.eps)
        w_sum = w_sum.clamp_min(self.eps)
        w = w_base / w_sum
        if zero_mask.any():
            uniform_w = torch.ones_like(w_base) / P
            w = torch.where(zero_mask, uniform_w, w)

        z = torch.einsum("brp,bpc->brc", w, patch_tokens)
        return z, w, w_base
