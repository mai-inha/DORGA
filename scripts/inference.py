
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image

from dorga.preprocessing.pipeline import PreprocessPipeline
from dorga.data.dataset import split_lungs_to_six
from dorga.models.backbone import load_mae_ckpt_to_512
from dorga.models.dorga_model import BrixiaViT512Dynamic


BRIXIA_ROI_NAMES = [
    "Left-Upper", "Left-Mid", "Left-Lower",
    "Right-Upper", "Right-Mid", "Right-Lower",
]

PRIVATEH_ROI_NAMES = ["RT", "RB", "LT", "LB"]
PRIVATEH_ROI_BOXES = [
    [0.0, 0.0, 0.5, 0.5],
    [0.5, 0.0, 1.0, 0.5],
    [0.0, 0.5, 0.5, 1.0],
    [0.5, 0.5, 1.0, 1.0],
]


def load_image(path: str) -> torch.Tensor:
    ext = Path(path).suffix.lower()

    if ext in (".dcm", ".dicom"):
        import pydicom
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    else:
        pil = Image.open(path).convert("L")
        arr = np.array(pil, dtype=np.float32) / 255.0

    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def build_model(cfg: dict, checkpoint: str, device: str) -> BrixiaViT512Dynamic:
    mcfg = cfg["model"]
    R, C, K = mcfg["num_regions"], mcfg["num_classes"], mcfg["num_patterns"]

    vit = load_mae_ckpt_to_512(
        path=mcfg["backbone"]["checkpoint"],
        num_classes=C,
        in_chans=mcfg["backbone"]["in_chans"],
        drop_path_rate=mcfg["backbone"]["drop_path_rate"],
    )

    model = BrixiaViT512Dynamic(
        vit=vit,
        num_regions=R,
        num_classes=C,
        num_patterns=K,
        proj_dim=mcfg["proj_dim"],
        pi_global=torch.zeros(R, R, C, C),
        pi_patterns=torch.zeros(K, R, R, C, C),
        gnn_num_heads=mcfg["gnn_num_heads"],
        gnn_dropout=mcfg["gnn_dropout"],
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def get_roi_coords(cfg: dict, masks: torch.Tensor) -> torch.Tensor:
    roi_mode = cfg.get("inference", {}).get("roi_mode", "mask")

    if roi_mode == "4roi":
        boxes = cfg["inference"]["roi_boxes"]
        return torch.tensor(boxes, dtype=torch.float32).unsqueeze(0)

    mask_512 = F.interpolate(masks.float(), size=(512, 512), mode="nearest")
    coords = split_lungs_to_six(mask_512[0, 0].numpy())
    return torch.tensor(coords, dtype=torch.float32).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="DORGA inference on a single CXR image")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path (default: from config)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    checkpoint = args.checkpoint or cfg.get("inference", {}).get("checkpoint")
    if not checkpoint:
        parser.error("--checkpoint is required (or set inference.checkpoint in config)")

    preproc_cfg = cfg.get("preprocessing", {})
    seg_weights = preproc_cfg.get("segmentation", {}).get("weights", "assets/weights/seg_weights.pt")
    stn_weights = preproc_cfg.get("alignment", {}).get("weights", "assets/weights/stn_weights.pth")

    R = cfg["model"]["num_regions"]
    C = cfg["model"]["num_classes"]
    roi_mode = cfg.get("inference", {}).get("roi_mode", "mask")
    dataset_name = "PrivateH" if roi_mode == "4roi" else "Brixia"
    roi_names = PRIVATEH_ROI_NAMES if roi_mode == "4roi" else BRIXIA_ROI_NAMES
    max_score = R * (C - 1)

    print(f"[1/4] Loading image: {args.image}  ({dataset_name}, R={R}, C={C})")
    raw = load_image(args.image)

    print(f"[2/4] Preprocessing (GUNet segmentation + STN alignment)")
    pipeline = PreprocessPipeline(seg_weights, stn_weights, device=device)
    aligned, masks = pipeline(raw)

    normalize = T.Normalize([0.56], [0.17])
    aligned_norm = normalize(aligned)

    rel = get_roi_coords(cfg, masks).to(device)

    print(f"[3/4] Running DORGA model")
    model = build_model(cfg, checkpoint, device)
    with torch.no_grad():
        out = model(aligned_norm.to(device), rel)
    preds = out["logits_s3"].argmax(-1)[0].cpu().numpy()

    print(f"[4/4] Results ({dataset_name})")
    print("=" * 45)
    print(f"  Image: {Path(args.image).name}")
    print(f"  Global Score: {preds.sum()}/{max_score}")
    print("-" * 45)
    for name, grade in zip(roi_names, preds):
        print(f"  {name:>14s}: Grade {grade}")
    print("=" * 45)

    return preds


if __name__ == "__main__":
    main()
