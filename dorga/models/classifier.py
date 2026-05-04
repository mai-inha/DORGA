import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularClassifier(nn.Module):
    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin
        self.logit_scale_s2 = nn.Parameter(torch.tensor(5.0).log())
        self.logit_scale_s3 = nn.Parameter(torch.tensor(5.0).log())

    def forward(self, x, anchors, stage="s3"):
        x_n = F.normalize(x, dim=-1)
        a_n = F.normalize(anchors, dim=-1)
        cos_sim = torch.einsum("...d,cd->...c", x_n, a_n)
        theta = torch.acos(cos_sim.clamp(-1 + 1e-7, 1 - 1e-7))
        if stage == "s2":
            scale = self.logit_scale_s2.exp().clamp(1.0, 10.0)
        else:
            scale = self.logit_scale_s3.exp().clamp(1.0, 10.0)
        return scale * (-theta)
