import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden_dim = in_dim 
        
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim) 
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        u = self.proj(z)
        u = F.normalize(u, p=2, dim=-1) 
        return u


class ROISpecificProjector(nn.Module):


    def __init__(self, in_dim, out_dim, num_regions=6):
        super().__init__()
        self.num_regions = num_regions

        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, out_dim)
            )
            for _ in range(num_regions)
        ])

        self._init_weights()

    def _init_weights(self):
        for net in self.nets:
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):

        vs = []
        for r in range(self.num_regions):
            vs.append(self.nets[r](x[:, r, :]))
        v = torch.stack(vs, dim=1)
        return F.normalize(v, p=2, dim=-1)
