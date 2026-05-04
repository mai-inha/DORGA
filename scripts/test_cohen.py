from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from scipy import stats
from sklearn.metrics import cohen_kappa_score as sklearn_kappa

from typing import Optional, Union

from dorga.models.backbone import load_mae_ckpt_to_512
from dorga.models.dorga_model import BrixiaViT512Dynamic
from dorga.utils.monitor import bsn_metrics_from_preds


def quadratic_weighted_kappa(y1: np.ndarray, y2: np.ndarray, num_classes: int = 4) -> float:
    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a, b in zip(y1.flatten(), y2.flatten()):
        O[int(a), int(b)] += 1
    O = O / O.sum()

    hist1 = np.sum(O, axis=1)
    hist2 = np.sum(O, axis=0)
    E = np.outer(hist1, hist2)

    W = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    num = np.sum(W * O)
    den = np.sum(W * E)
    if den == 0:
        return 1.0
    return 1.0 - num / den


def linear_weighted_kappa(y1: np.ndarray, y2: np.ndarray, num_classes: int = 4) -> float:
    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a, b in zip(y1.flatten(), y2.flatten()):
        O[int(a), int(b)] += 1
    O = O / O.sum()

    hist1 = np.sum(O, axis=1)
    hist2 = np.sum(O, axis=0)
    E = np.outer(hist1, hist2)

    W = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = abs(i - j) / (num_classes - 1)

    num = np.sum(W * O)
    den = np.sum(W * E)
    if den == 0:
        return 1.0
    return 1.0 - num / den


def unweighted_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    return sklearn_kappa(y1.flatten(), y2.flatten(), weights=None)


def intraclass_correlation(y1: np.ndarray, y2: np.ndarray) -> float:
    y1_flat = y1.flatten().astype(np.float64)
    y2_flat = y2.flatten().astype(np.float64)

    n = len(y1_flat)
    mean_all = (y1_flat.mean() + y2_flat.mean()) / 2

    subject_means = (y1_flat + y2_flat) / 2
    MSB = 2 * np.sum((subject_means - mean_all) ** 2) / (n - 1)
    MSW = np.sum((y1_flat - subject_means) ** 2 + (y2_flat - subject_means) ** 2) / n

    if MSB + MSW == 0:
        return 1.0
    icc = (MSB - MSW) / (MSB + MSW)
    return icc


def spearmans_rho(y1: np.ndarray, y2: np.ndarray) -> float:
    rho, _ = stats.spearmanr(y1.flatten(), y2.flatten())
    return rho


def pearsons_r(y1: np.ndarray, y2: np.ndarray) -> float:
    r, _ = stats.pearsonr(y1.flatten(), y2.flatten())
    return r


class CohenDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        mask_dir: str,
        annotator: str = "S",
        mode: str = "eval",
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.mode = mode
        self.annotator = annotator

        self.filenames = self.df["filename"].tolist()

        label_cols = [f"{annotator}-A", f"{annotator}-B", f"{annotator}-C",
                      f"{annotator}-D", f"{annotator}-E", f"{annotator}-F"]

        if not all(col in self.df.columns for col in label_cols):
            raise ValueError(f"Missing columns for annotator '{annotator}'. "
                             f"Available: {self.df.columns.tolist()}")

        self.labels = self.df[label_cols].values.tolist()

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.56], [0.17]),
        ])

    def __len__(self):
        return len(self.filenames)

    def _split_lungs_to_six(self, mask: np.ndarray, min_area: int = 500):
        from skimage.measure import label, regionprops

        H, W = mask.shape
        bin_mask = (mask > 0).astype(np.uint8)
        comps = [p for p in regionprops(label(bin_mask)) if p.area >= min_area]

        if len(comps) < 2:
            coords = []
            for rx in range(2):
                for ry in range(3):
                    y0 = ry * H // 3
                    y1 = (ry + 1) * H // 3
                    x0 = rx * W // 2
                    x1 = (rx + 1) * W // 2
                    coords.append((y0 / H, x0 / W, y1 / H, x1 / W))
            return coords

        left = min(comps, key=lambda p: p.centroid[1])
        right = max(comps, key=lambda p: p.centroid[1])

        out = []
        for reg in (left, right):
            y0, x0, y1, x1 = reg.bbox
            thirds = np.linspace(y0, y1, 4)
            for i in range(3):
                ys, ye = map(int, np.round([thirds[i], thirds[i + 1]]))
                xs, xe = x0, x1
                if ye - ys < 1: ye = ys + 1
                if xe - xs < 1: xe = xs + 1
                out.append((ys / H, xs / W, ye / H, xe / W))
        return out

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        stem = Path(filename).stem

        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ""]:
            candidate = self.img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
            candidate = self.img_dir / f"{filename}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            raise FileNotFoundError(f"Image not found for: {filename}")

        img = Image.open(img_path).convert("L")

        mask_path = self.mask_dir / f"{stem}.png"
        if mask_path.exists():
            mask_np = np.array(Image.open(mask_path).convert("L"))
            coords = self._split_lungs_to_six(mask_np)
        else:
            coords = []
            for rx in range(2):
                for ry in range(3):
                    coords.append((ry / 3, rx / 2, (ry + 1) / 3, (rx + 1) / 2))

        rel = torch.tensor(coords, dtype=torch.float32)
        x = self.tf(img)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x, y, rel, stem


def _make_cohen_loader(
    df: pd.DataFrame,
    img_dir: str,
    mask_dir: str,
    annotator: str = "S",
    mode: str = "eval",
    bs: int = 32,
) -> DataLoader:
    return DataLoader(
        CohenDataset(df, img_dir, mask_dir, annotator=annotator, mode=mode),
        batch_size=bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )


def load_model_from_checkpoint(ckpt_path: str | Path, device: torch.device) -> BrixiaViT512Dynamic:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if "config" in ckpt:
        cfg = ckpt["config"]
    else:
        print("[warn] no 'config' in checkpoint, using defaults")
        cfg = {
            "num_regions": 6, "num_classes": 4, "num_patterns": 7,
            "proj_dim": 768, "gnn_num_heads": 4, "gnn_dropout": 0.1,
        }

    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    pi_global = state.get("pi_global", state.get("dynamic_prior.pi_global"))
    pi_patterns = state.get("pi_patterns", state.get("dynamic_prior.pi_patterns"))

    if pi_global is None or pi_patterns is None:
        pi_keys = [k for k in state.keys() if "pi" in k.lower()]
        raise RuntimeError(
            f"Cannot find pi_global/pi_patterns in checkpoint. Found pi-related keys: {pi_keys}"
        )

    print(f"[ckpt] R={cfg['num_regions']}, C={cfg['num_classes']}, "
          f"K={cfg['num_patterns']}, proj_dim={cfg['proj_dim']}")
    if "epoch" in ckpt:
        print(f"[ckpt] epoch={ckpt['epoch']}, best_mae={ckpt.get('best_mae', 'N/A'):.4f}")

    vit = load_mae_ckpt_to_512("MRM.pth", num_classes=cfg["num_classes"], in_chans=1, verbose=False)

    model = BrixiaViT512Dynamic(
        vit,
        num_regions=cfg["num_regions"],
        num_classes=cfg["num_classes"],
        num_patterns=cfg["num_patterns"],
        proj_dim=cfg["proj_dim"],
        pi_global=pi_global,
        pi_patterns=pi_patterns,
        gnn_num_heads=cfg.get("gnn_num_heads", 4),
        gnn_dropout=cfg.get("gnn_dropout", 0.1),
    )

    msg = model.load_state_dict(state, strict=False)
    if msg.missing_keys:
        print(f"[warn] missing keys: {msg.missing_keys}")
    if msg.unexpected_keys:
        print(f"[warn] unexpected keys: {msg.unexpected_keys}")

    model = model.to(device)
    model.eval()
    print("[ckpt] model loaded successfully")
    return model


def _update_cm(cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, K: int) -> None:
    idx = y_true * K + y_pred
    cm += np.bincount(idx, minlength=K * K).reshape(K, K)


def plot_confusion_grid_3x2(
    roi_cms: list[np.ndarray],
    roi_acc: list[float],
    roi_mae: list[float],
    overall_acc: float,
    overall_mae: float,
    save_path: Path,
    title_prefix: str = "",
    class_names: list[str] | None = None,
) -> None:
    if class_names is None:
        class_names = [str(i) for i in range(roi_cms[0].shape[0])]

    fig, axes = plt.subplots(3, 2, figsize=(8, 12), dpi=150)

    for r in range(6):
        col = r // 3
        row = r % 3
        ax = axes[row, col]
        cm = roi_cms[r]

        ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, origin="lower")
        ax.set_title(f"ROI{r+1} | ACC {roi_acc[r]*100:.1f}% | MAE {roi_mae[r]:.2f}", fontsize=10)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")

        thresh = cm.max() * 0.5 if cm.max() > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7,
                )

    fig.suptitle(
        f"{title_prefix} | Overall ACC {overall_acc*100:.2f}% | MAE {overall_mae:.4f}",
        y=0.995,
    )
    plt.subplots_adjust(wspace=0.30, hspace=0.35)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)


def plot_agreement_comparison(
    metrics_senior: dict,
    metrics_junior: dict,
    inter_rater: dict,
    save_path: Path,
) -> None:

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    ax1 = axes[0]
    metrics_names = ["UWK", "QWK", "LWK", "ICC", "Spearman"]
    x = np.arange(len(metrics_names))
    width = 0.25

    senior_vals = [metrics_senior["UWK"], metrics_senior["QWK"], metrics_senior["LWK"],
                   metrics_senior["ICC"], metrics_senior["Spearman"]]
    junior_vals = [metrics_junior["UWK"], metrics_junior["QWK"], metrics_junior["LWK"],
                   metrics_junior["ICC"], metrics_junior["Spearman"]]
    inter_vals = [inter_rater["UWK"], inter_rater["QWK"], inter_rater["LWK"],
                  inter_rater["ICC"], inter_rater["Spearman"]]

    bars1 = ax1.bar(x - width, senior_vals, width, label="Model vs Senior", color="steelblue")
    bars2 = ax1.bar(x, junior_vals, width, label="Model vs Junior", color="coral")
    bars3 = ax1.bar(x + width, inter_vals, width, label="Senior vs Junior", color="gray", alpha=0.7)

    ax1.set_ylabel("Agreement Score")
    ax1.set_title("Agreement Metrics Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=7)

    ax2 = axes[1]
    categories = ["Model vs\nSenior", "Model vs\nJunior", "Senior vs\nJunior"]
    mae_vals = [metrics_senior["MAE"], metrics_junior["MAE"], inter_rater["MAE"]]
    colors = ["steelblue", "coral", "gray"]

    bars = ax2.bar(categories, mae_vals, color=colors)
    ax2.set_ylabel("MAE")
    ax2.set_title("Mean Absolute Error Comparison")

    for bar, val in zip(bars, mae_vals):
        ax2.annotate(f'{val:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10)

    ax3 = axes[2]
    ax3.axis('off')

    senior_score = (metrics_senior["QWK"] + metrics_senior["ICC"]) / 2
    junior_score = (metrics_junior["QWK"] + metrics_junior["ICC"]) / 2

    if senior_score > junior_score:
        closer_to = "SENIOR"
        diff = senior_score - junior_score
    else:
        closer_to = "JUNIOR"
        diff = junior_score - senior_score

    summary_text = f"""
    ══════════════════════════════════════
           AGREEMENT ANALYSIS SUMMARY
    ══════════════════════════════════════

    Model is closer to: {closer_to}

    Average Agreement Score:
      • Model ↔ Senior: {senior_score:.4f}
      • Model ↔ Junior: {junior_score:.4f}
      • Difference: {diff:.4f}

    ──────────────────────────────────────
    Unweighted κ (comparable to BS-Net):
      • Model ↔ Senior: {metrics_senior["UWK"]:.4f}
      • Model ↔ Junior: {metrics_junior["UWK"]:.4f}
      • Senior ↔ Junior: {inter_rater["UWK"]:.4f}
      • BS-Net reported: ~0.40
    ──────────────────────────────────────
    κ Interpretation:
      • > 0.80: Almost perfect
      • 0.60-0.80: Substantial
      • 0.40-0.60: Moderate
      • < 0.40: Fair to poor
    ──────────────────────────────────────

    Inter-rater (Senior ↔ Junior):
      • QWK: {inter_rater["QWK"]:.4f}
      • UWK: {inter_rater["UWK"]:.4f}
      • MAE: {inter_rater["MAE"]:.4f}
    """

    ax3.text(0.1, 0.95, summary_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)


ArrayLike = Union[np.ndarray, torch.Tensor]


def print_bsn_metrics_table(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    roi_names: Optional[list[str]] = None,
    decimals: int = 3,
    title: str = "=== BS-Net Metrics ===",
) -> pd.DataFrame:
    if isinstance(y_true, np.ndarray):
        gt = torch.from_numpy(y_true)
    else:
        gt = y_true.detach().cpu()

    if isinstance(y_pred, np.ndarray):
        pred = torch.from_numpy(y_pred)
    else:
        pred = y_pred.detach().cpu()

    gt = gt.long()
    pred = pred.long()

    _, R = gt.shape
    if roi_names is None:
        roi_names = ["A", "B", "C", "D", "E", "F"][:R]

    m = bsn_metrics_from_preds(pred, gt)

    col_labels = [f"ROI{i+1}({roi_names[i]})" for i in range(R)] + ["Avg", "Global"]
    row_labels = ["ACC", "MEr", "MAE", "SD", "CC"]

    values = []
    for metric in row_labels:
        key_metric = "Acc" if metric == "ACC" else metric
        row = []
        for name in roi_names:
            row.append(float(m[name][key_metric].item()))
        row.append(float(m["avg"][key_metric].item()))
        row.append(float(m["global"][key_metric].item()))
        values.append(row)

    df = pd.DataFrame(values, index=row_labels, columns=col_labels).round(decimals)

    print(f"\n{title}")
    print(df.to_string())

    return df


def save_predictions_csv(
    out_path: Path,
    stems: list[str],
    y_true_senior: np.ndarray,
    y_true_junior: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (["stem"] +
              [f"senior_roi{i}" for i in range(1, 7)] +
              [f"junior_roi{i}" for i in range(1, 7)] +
              [f"pred_roi{i}" for i in range(1, 7)])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for stem, ys, yj, yp in zip(stems, y_true_senior, y_true_junior, y_pred):
            w.writerow([stem] + ys.tolist() + yj.tolist() + yp.tolist())
    print(f"Saved predictions to: {out_path}")


@torch.no_grad()
def _forward_logits(model, imgs: torch.Tensor, rels: torch.Tensor) -> torch.Tensor:
    out = model(imgs, rels)
    return out["logits_s3"]


@torch.no_grad()
def evaluate_cohen(
    model,
    loader: DataLoader,
    device: torch.device,
    K: int = 4,
    R: int = 6
) -> dict:
    model.eval()

    correct = np.zeros(R, dtype=np.int64)
    mae_sum = np.zeros(R, dtype=np.float64)
    roi_cm = [np.zeros((K, K), dtype=np.int64) for _ in range(R)]

    stems_all: list[str] = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    total = 0
    for batch in loader:
        imgs, labs, rels, stems = batch
        imgs, labs, rels = imgs.to(device), labs.to(device), rels.to(device)

        logits = _forward_logits(model, imgs, rels)
        pred = logits.argmax(dim=-1)

        yt = labs.detach().cpu().numpy().astype(np.int64)
        yp = pred.detach().cpu().numpy().astype(np.int64)

        bsz = yt.shape[0]
        total += bsz

        for r in range(R):
            correct[r] += (yp[:, r] == yt[:, r]).sum()
            mae_sum[r] += np.abs(yp[:, r] - yt[:, r]).sum()
            _update_cm(roi_cm[r], yt[:, r], yp[:, r], K)

        stems_all.extend(list(stems))
        y_true_all.append(yt)
        y_pred_all.append(yp)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    roi_acc = (correct / max(1, total)).tolist()
    roi_mae = (mae_sum / max(1, total)).tolist()
    overall_acc = float((y_true == y_pred).mean())
    overall_mae = float(np.abs(y_true - y_pred).mean())

    return {
        "N": total,
        "stems": stems_all,
        "y_true": y_true,
        "y_pred": y_pred,
        "roi_acc": roi_acc,
        "roi_mae": roi_mae,
        "overall_acc": overall_acc,
        "overall_mae": overall_mae,
        "roi_cm": roi_cm,
    }


def compute_agreement_metrics(y1: np.ndarray, y2: np.ndarray, K: int = 4) -> dict:
    return {
        "UWK": unweighted_kappa(y1, y2),
        "QWK": quadratic_weighted_kappa(y1, y2, K),
        "LWK": linear_weighted_kappa(y1, y2, K),
        "ICC": intraclass_correlation(y1, y2),
        "Spearman": spearmans_rho(y1, y2),
        "Pearson": pearsons_r(y1, y2),
        "MAE": float(np.abs(y1 - y2).mean()),
        "ACC": float((y1 == y2).mean()),
    }


def test_cohen_dual_annotator(
    model,
    cohen_df: pd.DataFrame,
    img_dir: str,
    mask_dir: str,
    run_dir: Path,
    bs: int = 32,
    save_preds: bool = True,
):

    run_dir = Path(run_dir)
    device = next(model.parameters()).device

    print("\n" + "=" * 70)
    print("TEST 1: Model vs SENIOR Annotator")
    print("=" * 70)

    loader_senior = _make_cohen_loader(
        cohen_df, img_dir, mask_dir, annotator="S", mode="eval", bs=bs
    )
    res_senior = evaluate_cohen(model, loader_senior, device=device, K=4, R=6)

    print_bsn_metrics_table(
        res_senior["y_true"], res_senior["y_pred"],
        title="=== BS-Net Metrics (Model vs Senior) ==="
    )

    print(f"\n[Senior] Samples: {res_senior['N']}")
    print(f"[Senior] Overall ACC: {res_senior['overall_acc'] * 100:.2f}%")
    print(f"[Senior] Overall MAE: {res_senior['overall_mae']:.4f}")

    plot_confusion_grid_3x2(
        roi_cms=res_senior["roi_cm"],
        roi_acc=res_senior["roi_acc"],
        roi_mae=res_senior["roi_mae"],
        overall_acc=res_senior["overall_acc"],
        overall_mae=res_senior["overall_mae"],
        save_path=run_dir / "cohen_senior_confusion.png",
        title_prefix="Cohen - Model vs Senior",
    )

    print("\n" + "=" * 70)
    print("TEST 2: Model vs JUNIOR Annotator")
    print("=" * 70)

    loader_junior = _make_cohen_loader(
        cohen_df, img_dir, mask_dir, annotator="J", mode="eval", bs=bs
    )
    res_junior = evaluate_cohen(model, loader_junior, device=device, K=4, R=6)

    print_bsn_metrics_table(
        res_junior["y_true"], res_junior["y_pred"],
        title="=== BS-Net Metrics (Model vs Junior) ==="
    )

    print(f"\n[Junior] Samples: {res_junior['N']}")
    print(f"[Junior] Overall ACC: {res_junior['overall_acc'] * 100:.2f}%")
    print(f"[Junior] Overall MAE: {res_junior['overall_mae']:.4f}")

    plot_confusion_grid_3x2(
        roi_cms=res_junior["roi_cm"],
        roi_acc=res_junior["roi_acc"],
        roi_mae=res_junior["roi_mae"],
        overall_acc=res_junior["overall_acc"],
        overall_mae=res_junior["overall_mae"],
        save_path=run_dir / "cohen_junior_confusion.png",
        title_prefix="Cohen - Model vs Junior",
    )

    print("\n" + "=" * 70)
    print("AGREEMENT ANALYSIS")
    print("=" * 70)

    y_pred = res_senior["y_pred"]
    y_senior = res_senior["y_true"]
    y_junior = res_junior["y_true"]

    metrics_vs_senior = compute_agreement_metrics(y_pred, y_senior, K=4)
    print("\n[Model vs Senior]")
    for k, v in metrics_vs_senior.items():
        print(f"  {k:12s}: {v:.4f}")

    metrics_vs_junior = compute_agreement_metrics(y_pred, y_junior, K=4)
    print("\n[Model vs Junior]")
    for k, v in metrics_vs_junior.items():
        print(f"  {k:12s}: {v:.4f}")

    metrics_inter_rater = compute_agreement_metrics(y_senior, y_junior, K=4)
    print("\n[Senior vs Junior (Inter-rater)]")
    for k, v in metrics_inter_rater.items():
        print(f"  {k:12s}: {v:.4f}")

    print("\n" + "=" * 70)
    print("PER-ROI KAPPA ANALYSIS")
    print("=" * 70)

    roi_names = ["A", "B", "C", "D", "E", "F"]

    print(f"\n{'ROI':<10} {'UWK vs Sr':<12} {'UWK vs Jr':<12} {'UWK Sr-Jr':<12} {'Closer to':<10}")
    print("-" * 56)
    for r, name in enumerate(roi_names):
        uwk_sr = unweighted_kappa(y_pred[:, r], y_senior[:, r])
        uwk_jr = unweighted_kappa(y_pred[:, r], y_junior[:, r])
        uwk_ir = unweighted_kappa(y_senior[:, r], y_junior[:, r])
        closer = "Senior" if uwk_sr > uwk_jr else "Junior"
        print(f"ROI{r+1}({name})  {uwk_sr:.4f}       {uwk_jr:.4f}       {uwk_ir:.4f}       {closer}")

    print(f"\n{'ROI':<10} {'QWK vs Sr':<12} {'QWK vs Jr':<12} {'QWK Sr-Jr':<12} {'Closer to':<10}")
    print("-" * 56)
    for r, name in enumerate(roi_names):
        qwk_sr = quadratic_weighted_kappa(y_pred[:, r], y_senior[:, r], 4)
        qwk_jr = quadratic_weighted_kappa(y_pred[:, r], y_junior[:, r], 4)
        qwk_ir = quadratic_weighted_kappa(y_senior[:, r], y_junior[:, r], 4)
        closer = "Senior" if qwk_sr > qwk_jr else "Junior"
        print(f"ROI{r+1}({name})  {qwk_sr:.4f}       {qwk_jr:.4f}       {qwk_ir:.4f}       {closer}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    senior_score = (
        0.4 * metrics_vs_senior["QWK"] +
        0.3 * metrics_vs_senior["ICC"] +
        0.2 * (1 - metrics_vs_senior["MAE"]) +
        0.1 * metrics_vs_senior["Spearman"]
    )
    junior_score = (
        0.4 * metrics_vs_junior["QWK"] +
        0.3 * metrics_vs_junior["ICC"] +
        0.2 * (1 - metrics_vs_junior["MAE"]) +
        0.1 * metrics_vs_junior["Spearman"]
    )

    print(f"\nWeighted Agreement Score:")
    print(f"  Model ↔ Senior: {senior_score:.4f}")
    print(f"  Model ↔ Junior: {junior_score:.4f}")

    if senior_score > junior_score:
        diff = senior_score - junior_score
        diff_pct = (diff / junior_score) * 100 if junior_score > 0 else 0
        print(f"\n★ Model is CLOSER to SENIOR annotator")
        print(f"  Difference: +{diff:.4f} ({diff_pct:.1f}% higher agreement)")
    else:
        diff = junior_score - senior_score
        diff_pct = (diff / senior_score) * 100 if senior_score > 0 else 0
        print(f"\n★ Model is CLOSER to JUNIOR annotator")
        print(f"  Difference: +{diff:.4f} ({diff_pct:.1f}% higher agreement)")

    print("\n" + "-" * 70)
    print("BS-Net COMPARISON (Unweighted κ)")
    print("-" * 70)
    print(f"  BS-Net inter-rater κ (reported):  ~0.40 (moderate)")
    print(f"  DORGA  Model ↔ Senior UWK:        {metrics_vs_senior['UWK']:.4f}")
    print(f"  DORGA  Model ↔ Junior UWK:        {metrics_vs_junior['UWK']:.4f}")
    print(f"  DORGA  Senior ↔ Junior UWK:       {metrics_inter_rater['UWK']:.4f}")

    plot_agreement_comparison(
        metrics_vs_senior,
        metrics_vs_junior,
        metrics_inter_rater,
        save_path=run_dir / "cohen_agreement_comparison.png",
    )
    print(f"\nSaved agreement comparison plot to: {run_dir / 'cohen_agreement_comparison.png'}")

    if save_preds:
        save_predictions_csv(
            run_dir / "cohen_predictions.csv",
            res_senior["stems"],
            y_senior,
            y_junior,
            y_pred,
        )

    summary_df = pd.DataFrame({
        "Metric": ["UWK", "QWK", "LWK", "ICC", "Spearman", "Pearson", "MAE", "ACC"],
        "Model_vs_Senior": [metrics_vs_senior[k] for k in ["UWK", "QWK", "LWK", "ICC", "Spearman", "Pearson", "MAE", "ACC"]],
        "Model_vs_Junior": [metrics_vs_junior[k] for k in ["UWK", "QWK", "LWK", "ICC", "Spearman", "Pearson", "MAE", "ACC"]],
        "Senior_vs_Junior": [metrics_inter_rater[k] for k in ["UWK", "QWK", "LWK", "ICC", "Spearman", "Pearson", "MAE", "ACC"]],
    })
    summary_df.to_csv(run_dir / "cohen_agreement_summary.csv", index=False)
    print(f"Saved agreement summary to: {run_dir / 'cohen_agreement_summary.csv'}")

    return {
        "senior": res_senior,
        "junior": res_junior,
        "metrics_vs_senior": metrics_vs_senior,
        "metrics_vs_junior": metrics_vs_junior,
        "metrics_inter_rater": metrics_inter_rater,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Cohen inter-rater agreement evaluation")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to DORGA checkpoint")
    ap.add_argument("--cohen_meta", type=str, default="cohen/cohen_meta.csv", help="Path to Cohen metadata CSV")
    ap.add_argument("--cohen_img_dir", type=str, default="cohen/Normalize/images", help="Cohen normalized images dir")
    ap.add_argument("--cohen_mask_dir", type=str, default="cohen/Normalize/masks", help="Cohen normalized masks dir")
    ap.add_argument("--batch_size", type=int, default=64)
    _args = ap.parse_args()

    COHEN_META = Path(_args.cohen_meta)
    COHEN_IMG_DIR = Path(_args.cohen_img_dir)
    COHEN_MASK_DIR = Path(_args.cohen_mask_dir)
    RUN_DIR = Path(_args.checkpoint).parent
    CKPT = Path(_args.checkpoint)
    BS = _args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading Cohen metadata from: {COHEN_META}")
    cohen_df = pd.read_csv(COHEN_META)
    print(f"Total samples: {len(cohen_df)}")
    print(f"Columns: {cohen_df.columns.tolist()}")

    model = load_model_from_checkpoint(CKPT, device)

    results = test_cohen_dual_annotator(
        model=model,
        cohen_df=cohen_df,
        img_dir=str(COHEN_IMG_DIR),
        mask_dir=str(COHEN_MASK_DIR),
        run_dir=RUN_DIR,
        bs=BS,
        save_preds=True,
    )
