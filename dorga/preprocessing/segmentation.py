
import cv2
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from GUNet.GUNet_model import GUNet
from GUNet.utils import scipy_to_torch_sparse, genMatrixesLungsHeart


class LungSegmenter(nn.Module):

    def __init__(self, weights_path: str, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.model = self._build_and_load(weights_path)


    def _build_and_load(self, weights_path: str) -> GUNet:
        A, AD, D, U = genMatrixesLungsHeart()
        N1, N2 = A.shape[0], AD.shape[0]

        A = sp.csc_matrix(A).tocoo()
        AD = sp.csc_matrix(AD).tocoo()
        D = sp.csc_matrix(D).tocoo()
        U = sp.csc_matrix(U).tocoo()

        A_ = [A.copy()] * 3 + [AD.copy()] * 3
        D_ = [D.copy()]
        U_ = [U.copy()]

        A_t, D_t, U_t = (
            [scipy_to_torch_sparse(x).to(self.device) for x in X]
            for X in (A_, D_, U_)
        )

        f = 32
        config = {
            "n_nodes": [N1, N1, N1, N2, N2, N2],
            "latents": 64,
            "inputsize": 1024,
            "filters": [2, f, f, f, f // 2, f // 2, f // 2],
            "skip_features": f,
        }

        model = GUNet(config, D_t, U_t, A_t).to(self.device)
        state = torch.load(weights_path, map_location=self.device, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        return model


    @staticmethod
    def _landmarks_to_mask(landmarks: torch.Tensor, size: int = 1024) -> torch.Tensor:
        B = landmarks.shape[0]
        masks = torch.zeros(B, size, size, dtype=torch.uint8, device=landmarks.device)

        for i in range(B):
            rl = landmarks[i, :44].reshape(-1, 1, 2).to(torch.int32).cpu().numpy()
            ll = landmarks[i, 44:94].reshape(-1, 1, 2).to(torch.int32).cpu().numpy()
            mask_np = masks[i].cpu().numpy()
            cv2.drawContours(mask_np, [rl], -1, 255, -1)
            cv2.drawContours(mask_np, [ll], -1, 255, -1)
            masks[i] = torch.from_numpy(mask_np)

        return (masks.float() / 255.0)


    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        landmarks = self.model(images)[0]
        landmarks = (landmarks * 1024).int()
        masks = self._landmarks_to_mask(landmarks)
        return masks.unsqueeze(1).cpu()
