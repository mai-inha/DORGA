
import torch
import torch.nn as nn
import torch.nn.functional as F

from dorga.models.stn import STN


class SpatialAligner(nn.Module):

    def __init__(self, weights_path: str, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.stn = STN(in_shape=(1, 512, 512))
        state = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.stn.load_state_dict(state)
        self.stn.to(self.device)
        self.stn.eval()

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        if masks.shape[-2:] != (512, 512):
            masks = F.interpolate(
                masks.float(), size=(512, 512), mode="nearest"
            )

        masks = masks.to(self.device)
        images = images.to(self.device)

        theta = self.stn(masks)

        aligned = STN.transform(images, theta)

        return aligned.cpu()
