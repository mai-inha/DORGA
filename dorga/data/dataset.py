from __future__ import annotations
import ast
import csv
import cv2
import datetime
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
from skimage.measure import label, regionprops
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from sklearn.cluster import KMeans


def split_lungs_to_six(mask: np.ndarray, min_area: int = 1000):
    bin_mask = (mask > 0).astype(np.uint8)
    comps = [p for p in regionprops(label(bin_mask)) if p.area >= min_area]
    if len(comps) < 2: raise ValueError("mask needs two lung blobs ≥ min_area")
    left, right = min(comps, key=lambda p: p.centroid[1]), max(comps, key=lambda p: p.centroid[1])
    out = []
    for reg in (left, right):
        y0, x0, y1, x1 = reg.bbox
        thirds = np.linspace(y0, y1, 4)
        for i in range(3):
            ys, ye = map(int, np.round([thirds[i], thirds[i + 1]]))
            xs, xe = x0, x1

            if ye - ys < 1: ye = ys + 1
            if xe - xs < 1: xe = xs + 1

            out.append((ys / 512, xs / 512, ye / 512, xe / 512))
    return out


class ApplyCLAHE:
    def __init__(self, p: float = 0.2, clip_limit: float = 2.0):
        self.p, self.clip_limit = p, clip_limit
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        try:
            arr = np.array(img)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(arr)
            return Image.fromarray(enhanced)
        except Exception as e:
            print(f"CLAHE error: {e}")
            return img

class AddGaussianNoise:
    def __init__(self, p: float = 0.2, sigma_min: float = 0.005, sigma_max: float = 0.01):
        self.p, self.smin, self.smax = p, sigma_min, sigma_max
    
    def __call__(self, t: torch.Tensor):
        if random.random() < self.p:
            std = random.uniform(self.smin, self.smax)
            noise = torch.randn_like(t) * std
            t = (t + noise).clamp(0, 1)
        return t

class RandomGamma:
    def __init__(self, gamma_range=(0.8, 1.2), p: float = 0.5, gain: float = 1.0):
        self.lo, self.hi = gamma_range
        self.p = p
        self.gain = gain

    def __call__(self, img):
        if random.random() < self.p:
            gamma = random.uniform(self.lo, self.hi)
            return TF.adjust_gamma(img, gamma=gamma, gain=self.gain)
        return img

class BRIXIA(Dataset):
    def __init__(self, fns, labels, img_d, mask_d, mode="train", patterns=None, rotation_range=10):
        self.fns = list(map(str, fns))
        self.labels = labels
        self.img_d, self.mask_d = Path(img_d), Path(mask_d)
        self.mode = mode
        self.patterns = patterns
        self.rotation_range = rotation_range

        if mode == "train":
            self.photometric_tf = transforms.Compose([
                RandomGamma(gamma_range=(0.8, 1.2), p=0.2),
                transforms.ToTensor(),
                AddGaussianNoise(p=0.2),
                transforms.Normalize([0.56], [0.17]),
            ])
        else:
            self.photometric_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.56], [0.17]),
            ])

    def __len__(self):
        return len(self.fns)

    def _generate_roi_masks(self, mask_np: np.ndarray) -> torch.Tensor:
        H, W = mask_np.shape
        roi_masks = np.zeros((6, H, W), dtype=np.float32)

        try:
            coords = split_lungs_to_six(mask_np)
        except ValueError:
            return torch.from_numpy(roi_masks)

        for i, (y0n, x0n, y1n, x1n) in enumerate(coords):
            y0, y1 = int(y0n * H), int(y1n * H)
            x0, x1 = int(x0n * W), int(x1n * W)
            roi_masks[i, y0:y1, x0:x1] = mask_np[y0:y1, x0:x1] > 0

        return torch.from_numpy(roi_masks)

    def _masks_to_boxes(self, roi_masks: torch.Tensor, H: int, W: int) -> torch.Tensor:
        boxes = []
        for i in range(6):
            ys, xs = torch.where(roi_masks[i] > 0.5)
            if len(ys) > 0:
                y0, y1 = ys.min().item(), ys.max().item()
                x0, x1 = xs.min().item(), xs.max().item()
                boxes.append([y0/H, x0/W, y1/H, x1/W])
            else:
                boxes.append([0.0, 0.0, 0.01, 0.01])
        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx):
        try:
            stem = Path(self.fns[idx]).stem
            
            img = Image.open(self.img_d / f"{stem}.png").convert("L")
            mask = Image.open(self.mask_d / f"{stem}.png").convert("L")
            
            if self.mode == "train" and self.rotation_range > 0:
                angle = random.uniform(-self.rotation_range, self.rotation_range)
                img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

            mask_np = np.array(mask)
            H, W = mask_np.shape
            
            coords = split_lungs_to_six(mask_np)

            rel = torch.tensor(coords, dtype=torch.float32)

            roi_masks = np.zeros((6, H, W), dtype=np.float32)
            for i, (y0n, x0n, y1n, x1n) in enumerate(coords):
                y0, y1 = int(y0n * H), int(y1n * H)
                x0, x1 = int(x0n * W), int(x1n * W)
                roi_masks[i, y0:y1, x0:x1] = mask_np[y0:y1, x0:x1] > 0
            roi_masks = torch.from_numpy(roi_masks)

            x = self.photometric_tf(img)
            if torch.isnan(x).any(): x = torch.zeros_like(x)
            
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            pat = torch.tensor(self.patterns[idx], dtype=torch.long) if self.patterns is not None else torch.tensor(0, dtype=torch.long)
            
            return x, y, rel, roi_masks, pat, stem

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            x = torch.zeros(1, 512, 512)
            y = torch.zeros(6, dtype=torch.long)
            rel = torch.zeros(6, 4, dtype=torch.float32)
            roi_masks = torch.zeros(6, 512, 512, dtype=torch.float32)
            return x, y, rel, roi_masks, torch.tensor(0, dtype=torch.long), f"dummy_{idx}"


def _make_loader(
    df: pd.DataFrame,
    img_dir: str,
    mask_dir: str,
    mode: str,
    bs: int,
    shuffle: bool,
    drop_last: bool,
    label_cols: list[str] | None = None,
    include_pattern: bool = True,
    num_workers: int = 5,
):
    if label_cols is None:
        label_cols = [f"brixia{i}" for i in range(1, 7)]

    pattern_col = [c for c in df.columns if c.startswith("pattern")]
    patterns = df[pattern_col[0]].tolist() if include_pattern and pattern_col else None

    return DataLoader(
        BRIXIA(
            fns=df["Filename"].tolist(),
            labels=df[label_cols].values.tolist(),
            img_d=img_dir,
            mask_d=mask_dir,
            mode=mode,
            patterns=patterns,
            rotation_range=10,
        ),
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def compute_cond_mat(meta: pd.DataFrame) -> torch.Tensor:
    cols = [f"brixia{i}" for i in range(1,7)]
    val = meta[cols]
    out = np.zeros((6,6,4,4), dtype=np.float32)
    for i in range(6):
        for j in range(6):
            ct = pd.crosstab(val[cols[i]], val[cols[j]], normalize='columns')
            ct = ct.reindex(index=range(4), columns=range(4)).fillna(0.0)
            out[i, j] = ct.values
    return torch.tensor(out, dtype=torch.float32)

def generate_severity_patterns(meta: pd.DataFrame, n_patterns: int = 7, random_state: int = 42) -> pd.DataFrame:
    print(f"[Pattern] Generating {n_patterns} patterns using KMeans...")
    
    label_cols = [f"brixia{i}" for i in range(1, 7)]
    
    mask_tv = meta["split"].isin(["train", "valid"])
    X_tv = meta.loc[mask_tv, label_cols].to_numpy(dtype=np.float32)

    kmeans = KMeans(n_clusters=n_patterns, random_state=random_state, n_init=20)
    cluster_tv = kmeans.fit_predict(X_tv)

    cluster_severity = []
    for k in range(n_patterns):
        mean_sev = X_tv[cluster_tv == k].sum(axis=1).mean()
        cluster_severity.append((k, float(mean_sev)))
    
    cluster_severity.sort(key=lambda x: x[1])
    
    cluster_to_pattern = {old_id: new_id for new_id, (old_id, _) in enumerate(cluster_severity)}

    X_all = meta[label_cols].to_numpy(dtype=np.float32)
    cluster_all = kmeans.predict(X_all)
    
    pattern_col = f"pattern{n_patterns}"
    meta[pattern_col] = [cluster_to_pattern[c] for c in cluster_all]
    
    print(f"[Pattern] Done. Column '{pattern_col}' added.")
    print(meta[pattern_col].value_counts().sort_index())
    
    return meta

def compute_priors_dynamic(meta: pd.DataFrame, n_patterns: int) -> tuple[torch.Tensor, torch.Tensor]:
    print("[Prior] Computing dynamic priors with Robust Smoothing...")
    
    label_cols = [f"brixia{i}" for i in range(1, 7)]
    
    for col in label_cols:
        meta.loc[:, col] = pd.to_numeric(meta[col], errors='coerce').fillna(0).astype(int)
    
    R = 6
    C = 4
    K = n_patterns
    EPS = 1e-7

    df_tv = meta[meta["split"].isin(["train", "valid"])].copy()
    labels = df_tv[label_cols].to_numpy(dtype=np.int32)
    
    pattern_col = f"pattern{n_patterns}"
    if pattern_col not in df_tv.columns:
        raise ValueError(f"Pattern column '{pattern_col}' not found. Run generate_severity_patterns first.")
    pattern_ids = df_tv[pattern_col].to_numpy(dtype=np.int32)

    pi_global = np.zeros((R, R, C, C), dtype=np.float32)
    
    for i in range(R):
        for j in range(R):
            for d in range(C):
                mask_d = (labels[:, j] == d)
                n_d = mask_d.sum()
                
                if n_d == 0:
                    counts = np.zeros(C)
                else:
                    counts = np.bincount(labels[mask_d, i], minlength=C)
                
                probs = (counts + EPS) / (n_d + EPS * C)
                pi_global[i, j, :, d] = probs

    pi_patterns = np.zeros((K, R, R, C, C), dtype=np.float32)
    
    for k in range(K):
        mask_k = (pattern_ids == k)
        labels_k = labels[mask_k]
        N_k = labels_k.shape[0]
        
        if N_k < 10: 
            print(f"  [Warning] Pattern {k} has only {N_k} samples -> Using Global Prior")
            pi_patterns[k] = pi_global
            continue
            
        for i in range(R):
            for j in range(R):
                for d in range(C):
                    mask_d = (labels_k[:, j] == d)
                    n_d = mask_d.sum()
                    
                    if n_d == 0:
                        pi_patterns[k, i, j, :, d] = pi_global[i, j, :, d]
                    else:
                        counts = np.bincount(labels_k[mask_d, i], minlength=C)
                        probs = (counts + EPS) / (n_d + EPS * C)
                        pi_patterns[k, i, j, :, d] = probs

    if np.isnan(pi_global).any():
        print("!! Global Prior contains NaN. Filling with Uniform.")
        pi_global = np.nan_to_num(pi_global, nan=1.0/C)
        
    if np.isnan(pi_patterns).any():
        print("!! Pattern Prior contains NaN. Filling with Uniform.")
        pi_patterns = np.nan_to_num(pi_patterns, nan=1.0/C)

    print(f"[Prior] Done. pi_global: {pi_global.shape}, pi_patterns: {pi_patterns.shape}")
    
    return torch.from_numpy(pi_global), torch.from_numpy(pi_patterns)


from sklearn.mixture import GaussianMixture

def generate_residual_patterns(meta: pd.DataFrame, n_patterns: int = 7, random_state: int = 42) -> pd.DataFrame:
    print(f"[Pattern] Generating {n_patterns} patterns using Residual GMM (Shape-based)...")
    
    label_cols = [f"brixia{i}" for i in range(1, 7)]
    
    mask_tv = meta["split"].isin(["train", "valid"])
    X_tv = meta.loc[mask_tv, label_cols].to_numpy(dtype=np.float32)

    means_tv = X_tv.mean(axis=1, keepdims=True)
    residuals_tv = X_tv - means_tv

    gmm = GaussianMixture(n_components=n_patterns, random_state=random_state, n_init=5)
    gmm.fit(residuals_tv)
    
    X_all = meta[label_cols].to_numpy(dtype=np.float32)
    means_all = X_all.mean(axis=1, keepdims=True)
    residuals_all = X_all - means_all
    
    cluster_all = gmm.predict(residuals_all)
    
    pattern_col = f"pattern{n_patterns}"
    meta[pattern_col] = cluster_all
    
    print(f"[Pattern] Done. Column '{pattern_col}' added (Shape-based).")
    print(meta[pattern_col].value_counts().sort_index())
    
    return meta

def compute_priors_residual_gmm(meta: pd.DataFrame, n_patterns: int) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"[Prior] Computing priors (K={n_patterns}) with Robust Back-off Strategy...")
    
    label_cols = [f"brixia{i}" for i in range(1, 7)]
    
    df = meta.copy()
    for col in label_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    df_tv = df[df["split"].isin(["train", "valid"])].copy()
    labels = df_tv[label_cols].to_numpy(dtype=np.float32)
    
    pattern_col = f"pattern{n_patterns}"
    if pattern_col not in df_tv.columns:
        raise ValueError(f"Pattern column '{pattern_col}' not found.")
    pattern_ids = df_tv[pattern_col].to_numpy(dtype=np.int32)

    R = 6
    C = 4
    K = n_patterns
    EPS_SIGMA = 0.5 
    
    def fit_discrete_gaussian(data_values, num_classes=4):
        if len(data_values) == 0:
            return np.ones(num_classes, dtype=np.float32) / num_classes
        
        mu = np.mean(data_values)
        sigma = np.std(data_values)
        sigma = max(sigma, EPS_SIGMA)
        
        x = np.arange(num_classes, dtype=np.float32)
        logits = -0.5 * ((x - mu) / sigma) ** 2
        probs = np.exp(logits)
        return probs / (probs.sum() + 1e-9)

    pi_global = np.zeros((R, R, C, C), dtype=np.float32)
    
    for i in range(R):
        for j in range(R):
            for d in range(C):
                mask_d = (labels[:, j] == d)
                target_vals = labels[mask_d, i]
                
                if len(target_vals) == 0:
                    pi_global[i, j, :, d] = np.ones(C) / C
                else:
                    pi_global[i, j, :, d] = fit_discrete_gaussian(target_vals, C)

    pi_patterns = np.zeros((K, R, R, C, C), dtype=np.float32)
    fallback_count = 0
    
    for k in range(K):
        mask_k = (pattern_ids == k)
        labels_k = labels[mask_k]
        
        for i in range(R):
            for j in range(R):
                for d in range(C):
                    mask_d = (labels_k[:, j] == d)
                    target_vals = labels_k[mask_d, i]
                    
                    if len(target_vals) > 0:
                        pi_patterns[k, i, j, :, d] = fit_discrete_gaussian(target_vals, C)
                    else:
                        pi_patterns[k, i, j, :, d] = pi_global[i, j, :, d]
                        fallback_count += 1

    total_transitions = K * R * R * C
    fallback_rate = (fallback_count / total_transitions) * 100
    print(f"[Prior] Done. Fallback Rate: {fallback_rate:.2f}% (replaced empty bins with global prior)")
    
    return torch.from_numpy(pi_global), torch.from_numpy(pi_patterns)
