
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segmentation import LungSegmenter
from .alignment import SpatialAligner


class PreprocessPipeline(nn.Module):

    def __init__(
        self,
        seg_weights: str,
        stn_weights: str,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.segmenter = LungSegmenter(seg_weights, device=device)
        self.aligner = SpatialAligner(stn_weights, device=device)

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images_1024 = F.interpolate(
            images.float(), size=(1024, 1024), mode="bilinear", align_corners=False,
        )
        images_512 = F.interpolate(
            images.float(), size=(512, 512), mode="bilinear", align_corners=False,
        )

        masks = self.segmenter(images_1024)

        aligned = self.aligner(images_512, masks)

        return aligned, masks
