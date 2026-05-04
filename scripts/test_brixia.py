from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from typing import Optional, Union

from dorga.models.backbone import load_mae_ckpt_to_512
from dorga.data.dataset import BRIXIA
from dorga.models.dorga_model import BrixiaViT512Dynamic
from dorga.utils.monitor import bsn_metrics_from_preds

def _make_loader(df: pd.DataFrame, img_dir: str, mask_dir: str,
                  label_cols: list, mode: str, bs: int = 32) -> DataLoader:
    return DataLoader(
        BRIXIA(
            df["Filename"].tolist(),
            df[label_cols].values.tolist(),
            img_dir,
            mask_dir,
            mode=mode,
        ),
        batch_size=bs,
        shuffle=(mode == "train"),
        num_workers=4,
        pin_memory=True,
        drop_last=(mode == "train"),
    )


def save_predictions_csv(out_path: Path, stems: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["stem"] + [f"roi{i}_true" for i in range(1, 7)] + [f"roi{i}_pred" for i in range(1, 7)]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for stem, yt, yp in zip(stems, y_true, y_pred):
            w.writerow([stem] + yt.tolist() + yp.tolist())


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
        f"Per-ROI Confusion | Overall ACC {overall_acc*100:.2f}% | MAE {overall_mae:.4f}",
        y=0.995,
    )
    plt.subplots_adjust(wspace=0.30, hspace=0.35)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)


ArrayLike = Union[np.ndarray, torch.Tensor]


def print_roi_class_recall_support_tables(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    K: int = 4,
    roi_names: Optional[list[str]] = None,
    decimals: int = 4,
    title_recall: str = "=== Class × ROI Recall (4×6) ===",
    title_support: str = "=== Class × ROI Support (4×6) ===",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(y_true, torch.Tensor):
        yt = y_true.detach().cpu().numpy().astype(np.int64)
    else:
        yt = np.asarray(y_true, dtype=np.int64)

    if isinstance(y_pred, torch.Tensor):
        yp = y_pred.detach().cpu().numpy().astype(np.int64)
    else:
        yp = np.asarray(y_pred, dtype=np.int64)

    if yt.ndim != 2 or yp.ndim != 2:
        raise ValueError(f"y_true/y_pred must be 2D [N,R]. got {yt.shape} / {yp.shape}")
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch. got {yt.shape} vs {yp.shape}")

    N, R = yt.shape
    if roi_names is None:
        roi_names = ["A", "B", "C", "D", "E", "F"][:R]

    roi_cm = [np.zeros((K, K), dtype=np.int64) for _ in range(R)]
    for r in range(R):
        _update_cm(roi_cm[r], yt[:, r], yp[:, r], K)

    support_rk = np.stack([cm.sum(axis=1) for cm in roi_cm], axis=0)
    tp_rk = np.stack([np.diag(cm) for cm in roi_cm], axis=0)
    recall_rk = np.divide(tp_rk, support_rk, where=support_rk > 0, out=np.zeros_like(tp_rk, dtype=np.float64))

    recall = recall_rk.T
    support = support_rk.T

    row_labels = [f"Class{c}" for c in range(K)]
    col_labels = [f"ROI{i+1}({roi_names[i]})" for i in range(R)]

    df_recall = pd.DataFrame(recall, index=row_labels, columns=col_labels).round(decimals)
    df_support = pd.DataFrame(support, index=row_labels, columns=col_labels)

    print("\n" + title_recall)
    print(df_recall.to_string())
    print("\n" + title_support)
    print(df_support.to_string())

    return df_recall, df_support


def print_roi_trueclass_mae_table(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    K: int = 4,
    roi_names: Optional[list[str]] = None,
    decimals: int = 4,
    title: str = "=== Class × ROI MAE (4×6) ===",
) -> pd.DataFrame:
    if isinstance(y_true, torch.Tensor):
        yt = y_true.detach().cpu().numpy().astype(np.int64)
    else:
        yt = np.asarray(y_true, dtype=np.int64)

    if isinstance(y_pred, torch.Tensor):
        yp = y_pred.detach().cpu().numpy().astype(np.int64)
    else:
        yp = np.asarray(y_pred, dtype=np.int64)

    N, R = yt.shape
    if roi_names is None:
        roi_names = ["A", "B", "C", "D", "E", "F"][:R]

    mae_rk = np.zeros((R, K), dtype=np.float64)
    for r in range(R):
        t = yt[:, r]
        p = yp[:, r]
        err = np.abs(p - t)
        for c in range(K):
            m = (t == c)
            mae_rk[r, c] = err[m].mean() if m.any() else np.nan

    mae = mae_rk.T
    row_labels = [f"Class{c}" for c in range(K)]
    col_labels = [f"ROI{i+1}({roi_names[i]})" for i in range(R)]
    df = pd.DataFrame(mae, index=row_labels, columns=col_labels).round(decimals)

    print("\n" + title)
    print(df.to_string())
    return df


def print_bsn_metrics_table(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    roi_names: Optional[list[str]] = None,
    decimals: int = 3,
    title: str = "=== BS-Net Metrics (Per-ROI / Avg / Global) ===",
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

    print("\n" + title)
    print(df.to_string())
    return df


@torch.no_grad()
def _forward_logits(model, imgs: torch.Tensor, rels: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    try:
        out = model(imgs, rels, masks=masks)
    except TypeError:
        out = model(imgs, rels)
    return out["logits_s2"]


@torch.no_grad()
def evaluate_roi(model, loader: DataLoader, device: torch.device, K: int = 4, R: int = 6):
    model.eval()

    correct = np.zeros(R, dtype=np.int64)
    mae_sum = np.zeros(R, dtype=np.float64)
    roi_cm = [np.zeros((K, K), dtype=np.int64) for _ in range(R)]

    stems_all: list[str] = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    total = 0
    for imgs, labs, rels, mask, _, stems in loader:
        imgs, labs, rels, mask = imgs.to(device), labs.to(device), rels.to(device), mask.to(device)

        logits = _forward_logits(model, imgs, rels, mask)
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


def test(model, loader_test: DataLoader, run_dir: str | Path):
    run_dir = Path(run_dir)
    device = next(model.parameters()).device
    res = evaluate_roi(model, loader_test, device=device, K=4, R=6)

    print_bsn_metrics_table(res["y_true"], res["y_pred"])
    print_roi_class_recall_support_tables(res["y_true"], res["y_pred"], K=4)
    print_roi_trueclass_mae_table(res["y_true"], res["y_pred"], K=4)

    print("\n=== TEST RESULTS ===")
    print(f"Samples     : {res['N']}")
    print(f"Overall ACC : {res['overall_acc']*100:.2f}%")
    print(f"Overall MAE : {res['overall_mae']:.4f}")
    for i in range(6):
        print(f"  ROI{i+1}: ACC={res['roi_acc'][i]*100:.2f}%  MAE={res['roi_mae'][i]:.4f}")

    plot_confusion_grid_3x2(
        roi_cms=res["roi_cm"],
        roi_acc=res["roi_acc"],
        roi_mae=res["roi_mae"],
        overall_acc=res["overall_acc"],
        overall_mae=res["overall_mae"],
        save_path=run_dir / "consensus_test_grid.png",
        class_names=[str(i) for i in range(4)],
    )
    print(f"\nSaved confusion matrices to: {run_dir}")
    return res


ROI_NAMES_SHORT = ["ROI1\n(UR)", "ROI2\n(MR)", "ROI3\n(LR)",
                   "ROI4\n(UL)", "ROI5\n(ML)", "ROI6\n(LL)"]


@torch.no_grad()
def extract_attention(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_alpha, all_y_true, all_y_pred = [], [], []

    for batch in loader:
        imgs, labs, rels, masks, _, stems = batch
        imgs = imgs.to(device)
        labs = labs.to(device)
        rels = rels.to(device)
        masks = masks.to(device)

        out = model(imgs, rels, masks=masks, labels=labs)

        alpha = out["gnn_out"]["raw"]["alpha"][0]
        pred = out["logits_s3"].argmax(dim=-1)

        all_alpha.append(alpha.cpu().numpy())
        all_y_true.append(labs.cpu().numpy().astype(int))
        all_y_pred.append(pred.cpu().numpy().astype(int))

    return {
        "alphas": np.concatenate(all_alpha, axis=0),
        "y_true": np.concatenate(all_y_true, axis=0),
        "y_pred": np.concatenate(all_y_pred, axis=0),
    }


def plot_attention_concentration(
    attn_res: dict,
    save_dir: Path,
    use_true_labels: bool = True,
    tag: str = "",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    alphas = attn_res["alphas"]
    labels = attn_res["y_true"] if use_true_labels else attn_res["y_pred"]
    N, R, _ = alphas.shape

    same_sum  = np.zeros((R, R))
    same_cnt  = np.zeros((R, R))
    cross_sum = np.zeros((R, R))
    cross_cnt = np.zeros((R, R))

    for i in range(R):
        for j in range(R):
            if i == j:
                continue
            same_mask  = (labels[:, i] == labels[:, j])
            cross_mask = ~same_mask

            same_sum[i, j]  = alphas[same_mask, i, j].sum()
            same_cnt[i, j]  = same_mask.sum()
            cross_sum[i, j] = alphas[cross_mask, i, j].sum()
            cross_cnt[i, j] = cross_mask.sum()

    same_avg  = np.divide(same_sum,  same_cnt,  where=same_cnt > 0,
                          out=np.full((R, R), np.nan))
    cross_avg = np.divide(cross_sum, cross_cnt, where=cross_cnt > 0,
                          out=np.full((R, R), np.nan))
    np.fill_diagonal(same_avg,  np.nan)
    np.fill_diagonal(cross_avg, np.nan)

    all_vals = np.concatenate([
        same_avg[~np.isnan(same_avg)],
        cross_avg[~np.isnan(cross_avg)]
    ])
    vmin, vmax = 0, max(all_vals.max(), 0.4)

    label_type = "Ground-Truth" if use_true_labels else "Predicted"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    for ax, data, cmap, title_part in [
        (axes[0], same_avg,  "Reds",  "Same-Grade"),
        (axes[1], cross_avg, "Blues", "Cross-Grade"),
    ]:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"{title_part} Attention\n({label_type} labels)", fontsize=12)
        ax.set_xticks(range(R))
        ax.set_yticks(range(R))
        ax.set_xticklabels(ROI_NAMES_SHORT, fontsize=9)
        ax.set_yticklabels(ROI_NAMES_SHORT, fontsize=9)
        ax.set_xlabel("Source ROI $j$", fontsize=11)
        ax.set_ylabel("Target ROI $i$", fontsize=11)
        for i in range(R):
            for j in range(R):
                v = data[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
                else:
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                            color="white" if v > (vmax * 0.55) else "black")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Mean α")

    plt.suptitle(f"DORGA Attention: Same vs Cross Grade {tag}", fontsize=14, y=1.02)
    plt.tight_layout()

    suffix = "gt" if use_true_labels else "pred"
    fname = save_dir / f"attention_same_vs_cross_{suffix}{('_' + tag) if tag else ''}.png"
    plt.savefig(str(fname), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[saved] {fname}")

    s_vals = same_avg[~np.isnan(same_avg)]
    c_vals = cross_avg[~np.isnan(cross_avg)]
    print(f"\n{'='*50}")
    print(f"Attention Concentration ({label_type})")
    print(f"  Same-grade  α: mean={s_vals.mean():.4f}  std={s_vals.std():.4f}")
    print(f"  Cross-grade α: mean={c_vals.mean():.4f}  std={c_vals.std():.4f}")
    ratio = s_vals.mean() / c_vals.mean() if c_vals.mean() > 0 else float("inf")
    print(f"  Ratio (same/cross): {ratio:.3f}")
    print(f"{'='*50}")

    return same_avg, cross_avg


def plot_attention_avg(attn_res: dict, save_dir: Path, tag: str = ""):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    avg_a = attn_res["alphas"].mean(axis=0)
    R = avg_a.shape[0]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(avg_a, cmap="YlOrRd", vmin=0, aspect="equal")
    ax.set_title(f"Average Attention α {tag}", fontsize=13)
    ax.set_xticks(range(R))
    ax.set_yticks(range(R))
    ax.set_xticklabels(ROI_NAMES_SHORT, fontsize=9)
    ax.set_yticklabels(ROI_NAMES_SHORT, fontsize=9)
    ax.set_xlabel("Source ROI $j$", fontsize=11)
    ax.set_ylabel("Target ROI $i$", fontsize=11)

    for i in range(R):
        for j in range(R):
            ax.text(j, i, f"{avg_a[i,j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if avg_a[i,j] > avg_a.max()*0.6 else "black")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean α")
    plt.tight_layout()

    fname = save_dir / f"attention_avg_heatmap{('_' + tag) if tag else ''}.png"
    plt.savefig(str(fname), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[saved] {fname}")


def plot_grade_pair_attention(
    attn_res: dict,
    save_dir: Path,
    use_true_labels: bool = True,
    K: int = 4,
    tag: str = "",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    alphas = attn_res["alphas"]
    labels = attn_res["y_true"] if use_true_labels else attn_res["y_pred"]
    N, R, _ = alphas.shape

    pair_sum = np.zeros((K, K), dtype=np.float64)
    pair_cnt = np.zeros((K, K), dtype=np.int64)

    for i in range(R):
        for j in range(R):
            if i == j:
                continue
            gi = labels[:, i]
            gj = labels[:, j]
            a  = alphas[:, i, j]
            for ci in range(K):
                for cj in range(K):
                    mask = (gi == ci) & (gj == cj)
                    pair_sum[ci, cj] += a[mask].sum()
                    pair_cnt[ci, cj] += mask.sum()

    pair_avg = np.divide(
        pair_sum, pair_cnt,
        where=pair_cnt > 0,
        out=np.full((K, K), np.nan),
    )

    label_type = "Ground-Truth" if use_true_labels else "Predicted"
    grade_labels = [f"Grade {c}" for c in range(K)]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    vmin = 0
    valid = pair_avg[~np.isnan(pair_avg)]
    vmax = valid.max() if len(valid) > 0 else 0.3

    im = ax.imshow(pair_avg, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(f"Grade-Pair Attention ({label_type})\naveraged over all ROI pairs (i≠j)",
                 fontsize=12)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(grade_labels, fontsize=10)
    ax.set_yticklabels(grade_labels, fontsize=10)
    ax.set_xlabel("Source ROI grade ($g_j$)", fontsize=11)
    ax.set_ylabel("Target ROI grade ($g_i$)", fontsize=11)

    for ci in range(K):
        for cj in range(K):
            v = pair_avg[ci, cj]
            if np.isnan(v):
                ax.text(cj, ci, "N/A", ha="center", va="center", fontsize=9, color="gray")
            else:
                cnt = pair_cnt[ci, cj]
                ax.text(cj, ci, f"{v:.4f}\n(n={cnt})",
                        ha="center", va="center", fontsize=8,
                        color="white" if v > (vmax * 0.55) else "black")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean α")
    plt.tight_layout()

    suffix = "gt" if use_true_labels else "pred"
    fname = save_dir / f"attention_grade_pair_{suffix}{('_' + tag) if tag else ''}.png"
    plt.savefig(str(fname), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[saved] {fname}")

    df = pd.DataFrame(pair_avg, index=grade_labels, columns=grade_labels).round(4)
    df_cnt = pd.DataFrame(pair_cnt, index=grade_labels, columns=grade_labels)

    print(f"\n{'='*55}")
    print(f"Grade-Pair Attention Summary ({label_type})")
    print(f"{'='*55}")
    print("\nMean α per (target_grade, source_grade):")
    print(df.to_string())
    print("\nSample count per (target_grade, source_grade):")
    print(df_cnt.to_string())

    diag_vals = np.array([pair_avg[c, c] for c in range(K) if not np.isnan(pair_avg[c, c])])
    off_vals = np.array([pair_avg[ci, cj]
                         for ci in range(K) for cj in range(K)
                         if ci != cj and not np.isnan(pair_avg[ci, cj])])

    if len(diag_vals) > 0 and len(off_vals) > 0:
        print(f"\nSame-grade (diagonal)  α: mean={diag_vals.mean():.4f}")
        print(f"Cross-grade (off-diag) α: mean={off_vals.mean():.4f}")
        ratio = diag_vals.mean() / off_vals.mean() if off_vals.mean() > 0 else float("inf")
        print(f"Ratio (same/cross): {ratio:.3f}")
    print(f"{'='*55}")

    return pair_avg, pair_cnt


if __name__ == "__main__":
    import argparse, yaml

    ap = argparse.ArgumentParser(description="Brixia consensus test set evaluation")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    ap.add_argument("--config", type=str, default="configs/default.yaml", help="YAML config path")
    ap.add_argument("--batch_size", type=int, default=64)
    _args = ap.parse_args()

    with open(_args.config) as f:
        cfg = yaml.safe_load(f)

    d = cfg["data"]
    norm_dir = Path(d["normalized_dir"])
    img_dir = str(norm_dir / "images")
    mask_dir = str(norm_dir / "masks")
    label_cols = d.get("label_cols", [f"brixia{i}" for i in range(1, 7)])

    RUN_DIR = Path(_args.checkpoint).parent
    CKPT = Path(_args.checkpoint)
    BS = _args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    meta = pd.read_csv(d["meta_csv"])
    test_df = meta[meta.split == d.get("test_split", "consensus_test")]
    print(f"Test samples: {test_df.shape[0]}")

    model = load_model_from_checkpoint(CKPT, device)
    loader_test = _make_loader(test_df, img_dir=img_dir, mask_dir=mask_dir,
                               label_cols=label_cols, mode="eval", bs=BS)

    res = test(model, loader_test=loader_test, run_dir=RUN_DIR)

    print("\n\n" + "="*60)
    print(" ATTENTION ANALYSIS")
    print("="*60)

    attn_res = extract_attention(model, loader_test, device)

    plot_attention_avg(attn_res, save_dir=RUN_DIR)

    plot_attention_concentration(attn_res, save_dir=RUN_DIR, use_true_labels=True)

    plot_attention_concentration(attn_res, save_dir=RUN_DIR, use_true_labels=False)

    plot_grade_pair_attention(attn_res, save_dir=RUN_DIR, use_true_labels=True)

    plot_grade_pair_attention(attn_res, save_dir=RUN_DIR, use_true_labels=False)

    print("\nDone.")
