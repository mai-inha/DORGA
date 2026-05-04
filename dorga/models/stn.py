
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class LungSegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pre_mask = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=pre_mask)
            pre_mask = augmented['image']

        pre_mask = pre_mask / 255.0
        mask = mask / 255.0

        return pre_mask, mask


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.contiguous()
        targets = targets.contiguous()
        intersection = (inputs * targets).sum(dim=2).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            inputs.sum(dim=2).sum(dim=1) + targets.sum(dim=2).sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()


IDENTITY_THETA = torch.tensor(
    [[1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]], dtype=torch.float32
)


class STN(nn.Module):

    def __init__(self, in_shape=(1, 512, 512), mask_resize: int = 512,
                 dense_neurons=50, freeze_align_model=False,
                 identity_gate_threshold=0.01):
        super(STN, self).__init__()

        assert not in_shape[1] % mask_resize, "The STN size must be a multiple of mask size"
        trainable = not freeze_align_model

        self.identity_gate_threshold = identity_gate_threshold

        self.pool1 = nn.MaxPool2d(
            kernel_size=(in_shape[1] // mask_resize, in_shape[2] // mask_resize)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=in_shape[0], out_channels=20, kernel_size=5, stride=1)
        if not trainable:
            self.conv1.requires_grad_(False)

        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, stride=1)
        if not trainable:
            self.conv2.requires_grad_(False)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=self.calculate_flatten_size(in_shape),
            out_features=dense_neurons,
        )
        if not trainable:
            self.fc1.requires_grad_(False)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(in_features=dense_neurons, out_features=6)
        if not trainable:
            self.fc2.requires_grad_(False)

        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def calculate_flatten_size(self, in_shape):
        dummy_input = torch.randn(1, in_shape[0], in_shape[1], in_shape[2])
        x = self.pool1(dummy_input)
        x = self.pool2(self.conv1(x))
        x = self.pool3(self.conv2(x))
        return x.numel()


    def predict_theta(self, x):
        xs = self.pool1(x)
        xs = self.pool2(self.conv1(xs))
        xs = self.pool3(self.conv2(xs))
        xs = self.flatten(xs)
        xs = self.relu(self.fc1(xs))
        theta = self.fc2(xs)
        theta = theta.view(-1, 2, 3)
        return theta

    @staticmethod
    def transform(x, theta):
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


    def forward(self, x):
        theta = self.predict_theta(x)

        if not self.training and self.identity_gate_threshold > 0:
            theta = self._apply_identity_gate(theta)

        return theta


    def forward_with_idempotency(self, x):
        theta1 = self.predict_theta(x)
        aligned = self.transform(x, theta1)
        theta2 = self.predict_theta(aligned.detach())
        return theta1, aligned, theta2


    def _apply_identity_gate(self, theta):
        identity = IDENTITY_THETA.to(theta.device).unsqueeze(0)
        diff = (theta - identity).abs().mean(dim=(1, 2))
        mask = (diff < self.identity_gate_threshold).float()
        mask = mask.view(-1, 1, 1)
        return mask * identity + (1 - mask) * theta


class IdempotencyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, theta2):
        identity = IDENTITY_THETA.to(theta2.device).unsqueeze(0).expand_as(theta2)
        return F.mse_loss(theta2, identity)
