
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from dorga.preprocessing.segmentation import LungSegmenter
from dorga.preprocessing.alignment import SpatialAligner


def load_grayscale(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in (".dcm", ".dicom"):
        import pydicom
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return (arr * 255).astype(np.uint8)
    else:
        return np.array(Image.open(path).convert("L"))


def main():
    parser = argparse.ArgumentParser(description="Batch preprocessing: seg + align")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    original_dir = Path(cfg["data"]["original_dir"])
    normalized_dir = Path(cfg["data"]["normalized_dir"])
    out_img_dir = normalized_dir / "images"
    out_mask_dir = normalized_dir / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    preproc_cfg = cfg.get("preprocessing", {})
    seg_weights = preproc_cfg.get("segmentation", {}).get("weights", "assets/weights/seg_weights.pt")
    stn_weights = preproc_cfg.get("alignment", {}).get("weights", "assets/weights/stn_weights.pth")

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Loading models...")
    segmenter = LungSegmenter(seg_weights, device=device)
    aligner = SpatialAligner(stn_weights, device=device)

    exts = (".png", ".jpg", ".jpeg", ".dcm", ".dicom")
    files = sorted([f for f in original_dir.iterdir() if f.suffix.lower() in exts])
    print(f"Found {len(files)} images in {original_dir}")

    existing = {f.stem for f in out_img_dir.glob("*.png")}
    todo = [f for f in files if f.stem not in existing]
    if len(todo) < len(files):
        print(f"Skipping {len(files) - len(todo)} already processed, {len(todo)} remaining")
    files = todo

    bs = args.batch_size
    for i in tqdm(range(0, len(files), bs), desc="Preprocessing"):
        batch_files = files[i:i + bs]
        stems = [f.stem for f in batch_files]

        imgs_1024 = []
        imgs_512 = []
        for fpath in batch_files:
            arr = load_grayscale(fpath).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            imgs_1024.append(F.interpolate(t, size=(1024, 1024), mode="bilinear", align_corners=False))
            imgs_512.append(F.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False))

        batch_1024 = torch.cat(imgs_1024, dim=0)
        batch_512 = torch.cat(imgs_512, dim=0)

        masks_1024 = segmenter(batch_1024)

        aligned_512 = aligner(batch_512, masks_1024)

        masks_512 = F.interpolate(masks_1024.float(), size=(512, 512), mode="nearest")

        for j, stem in enumerate(stems):
            img_np = (aligned_512[j, 0].numpy() * 255).clip(0, 255).astype(np.uint8)
            mask_np = (masks_512[j, 0].numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(out_img_dir / f"{stem}.png"), img_np)
            cv2.imwrite(str(out_mask_dir / f"{stem}.png"), mask_np)

    total = len(list(out_img_dir.glob("*.png")))
    print(f"Done. {total} images in {out_img_dir}")
    print(f"       {total} masks  in {out_mask_dir}")


if __name__ == "__main__":
    main()
