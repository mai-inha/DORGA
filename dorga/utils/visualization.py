import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict, Optional, Sequence, Union
from pathlib import Path
import pandas as pd
import math
import random
import torch.nn.functional as F
import seaborn as sns

ROI_NAMES = ["1","2","3","4","5","6"]
ROI_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#5fa2dd", "#ffb366", "#74c476"]
CLASS_MARKERS = ["o", "s", "^", "*"]

def _has_data(ax):
    return len(ax.lines) > 0 or len(ax.collections) > 0 or len(ax.patches) > 0


def plot_sample_overlay(
    img: torch.Tensor,
    roi_masks: torch.Tensor,
    rel: torch.Tensor,
    lab: torch.Tensor,
    logits: torch.Tensor,
    w: torch.Tensor,
    run_dir,
    epoch: int,
    stem: str = "",
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pathlib import Path

    ROI_NAMES_LOCAL  = ["ROI1","ROI2","ROI3","ROI4","ROI5","ROI6"]
    CLASS_CMAP = ["#3b82f6","#22c55e","#f59e0b","#ef4444"]

    run_dir = Path(run_dir) / "diag"
    run_dir.mkdir(parents=True, exist_ok=True)

    img_np   = img[0].detach().cpu().numpy()
    masks_np = roi_masks.detach().cpu().numpy()
    rel_np   = rel.detach().cpu().numpy()
    lab_np   = lab.detach().cpu().long().numpy()
    pred_np  = logits.detach().cpu().argmax(-1).numpy()
    w_np     = w.detach().cpu().numpy()

    H, W = img_np.shape
    grid = int(np.sqrt(w_np.shape[-1]))

    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    fig.suptitle(
        f"Epoch {epoch:03d} | {stem} | GT={lab_np.tolist()} Pred={pred_np.tolist()}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    axes[0,0].imshow(img_np, cmap="gray")
    axes[0,0].set_title("Image", fontsize=10)

    axes[0,1].imshow(img_np, cmap="gray")
    overlay = np.zeros((*img_np.shape, 4))
    for r in range(6):
        rgba = plt.matplotlib.colors.to_rgba(ROI_COLORS[r], alpha=0.35)
        m = masks_np[r] > 0.5
        for c in range(4): overlay[m, c] = rgba[c]
    axes[0,1].imshow(overlay)
    axes[0,1].set_title("Mask Overlay", fontsize=10)

    axes[0,2].imshow(img_np, cmap="gray")
    for r in range(6):
        y0, x0, y1, x1 = rel_np[r] * H
        rect = mpatches.FancyBboxPatch(
            (x0, y0), x1-x0, y1-y0, boxstyle="round,pad=0",
            linewidth=2, edgecolor=ROI_COLORS[r], facecolor="none")
        axes[0,2].add_patch(rect)
        axes[0,2].text(x0+3, y0+14, ROI_NAMES_LOCAL[r], fontsize=7, color="white",
                       fontweight="bold",
                       bbox=dict(facecolor=ROI_COLORS[r], alpha=0.7, pad=1, edgecolor="none"))
    axes[0,2].set_title("ROI Boxes", fontsize=10)

    bar = np.zeros((2, 6, 3))
    for r in range(6):
        bar[0, r] = plt.matplotlib.colors.to_rgb(CLASS_CMAP[lab_np[r]])
        bar[1, r] = plt.matplotlib.colors.to_rgb(CLASS_CMAP[pred_np[r]])
    axes[0,3].imshow(bar, aspect="auto", interpolation="nearest")
    axes[0,3].set_yticks([0,1]); axes[0,3].set_yticklabels(["GT","Pred"], fontsize=9)
    axes[0,3].set_xticks(range(6)); axes[0,3].set_xticklabels(ROI_NAMES_LOCAL, fontsize=8)
    for r in range(6):
        axes[0,3].text(r, 0, str(lab_np[r]), ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        axes[0,3].text(r, 1, str(pred_np[r]), ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    match = (lab_np == pred_np).sum()
    axes[0,3].set_title(f"GT vs Pred ({match}/6)", fontsize=10)

    counts = [(masks_np[r] > 0.5).sum() for r in range(6)]
    axes[0,4].bar(range(6), counts, color=ROI_COLORS)
    axes[0,4].set_xticks(range(6)); axes[0,4].set_xticklabels(ROI_NAMES_LOCAL, fontsize=8)
    axes[0,4].set_title("Mask Pixels", fontsize=10)

    w_max   = w_np.max(axis=-1)
    w_ent   = -(w_np * np.log(np.clip(w_np, 1e-10, None))).sum(axis=-1)
    w_top10 = np.sort(w_np, axis=-1)[:, -10:].sum(axis=-1)
    x_pos = np.arange(6); bw = 0.25
    axes[0,5].bar(x_pos-bw, w_max, bw, label="max_w", color="#60a5fa")
    axes[0,5].bar(x_pos,    w_top10, bw, label="top10", color="#f97316")
    axes[0,5].bar(x_pos+bw, w_ent/max(w_ent.max(),1e-8), bw, label="ent(n)", color="#a78bfa")
    axes[0,5].set_xticks(range(6)); axes[0,5].set_xticklabels(ROI_NAMES_LOCAL, fontsize=8)
    axes[0,5].legend(fontsize=7); axes[0,5].set_title("Pooler Stats", fontsize=10)

    for r in range(6):
        wmap = w_np[r].reshape(grid, grid)
        im = axes[1,r].imshow(wmap, cmap="hot", interpolation="nearest")
        axes[1,r].set_title(f"{ROI_NAMES_LOCAL[r]} (max={w_np[r].max():.4f})", fontsize=9)
        fig.colorbar(im, ax=axes[1,r], fraction=0.046, pad=0.04)

    for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(run_dir / f"ep{epoch:03d}_{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_losses(
    run_dir: Union[str, Path],
    csv_name: str = "loss_metrics.csv",
    out_dirname: str = "plots",
    out_name: str = "loss.png",
) -> Path:
    run_dir = Path(run_dir)
    csv_path = run_dir / csv_name
    out_dir = run_dir / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists(): return out_dir

    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns or len(df) == 0: return out_dir

    x = df["epoch"].values
    EXCLUDE_KEYWORDS = ["Acc", "MAE", "MEr", "SD", "CC", "pattern_acc", "epoch"]
    
    loss_keys = []
    for col in df.columns:
        if col.startswith("tr_"):
            is_metric = any(k in col for k in EXCLUDE_KEYWORDS)
            if not is_metric:
                loss_keys.append(col[3:])
    
    if not loss_keys: return out_dir

    n_losses = len(loss_keys)
    cols = min(4, n_losses)
    rows = math.ceil(n_losses / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=150)
    if rows * cols == 1: axes = [axes]
    else: axes = axes.flatten()
        
    for i, key in enumerate(loss_keys):
        ax = axes[i]
        tr_col = f"tr_{key}"
        vl_col = f"vl_{key}"
        
        ax.set_title(key.upper(), fontsize=11, fontweight="bold")
        if tr_col in df.columns:
            ax.plot(x, df[tr_col], label="Train", color="tab:blue", linestyle="-", linewidth=1.5)
        if vl_col in df.columns:
            ax.plot(x, df[vl_col], label="Valid", color="tab:orange", linestyle="--", linewidth=1.5)
            
        ax.set_xlabel("epoch")
        ax.set_ylabel("Loss")
        if _has_data(ax): ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
    for j in range(i + 1, len(axes)): axes[j].axis("off")

    fig.suptitle("Training Losses", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / out_name, bbox_inches="tight")
    plt.close(fig)
    return out_dir


def plot_stage1(
    run_dir: Union[str, Path],
    csv_name: str = "stage1_metrics.csv",
    out_dirname: str = "plots",
    out_name: str = "stage1.png",
) -> Path:
    run_dir = Path(run_dir)
    out_dir = run_dir / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = run_dir / csv_name
    if not csv_path.exists(): return out_dir
    
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns or len(df) == 0: return out_dir
    
    x = df["epoch"].values
    fig, axes = plt.subplots(2, 5, figsize=(25, 9), dpi=150)
    
    def _plot_tr_vl(ax, key_base, title, ylab, ylim=None):
        ax.set_title(title, fontsize=10, fontweight="bold")
        tr_key, vl_key = f"tr_{key_base}", f"vl_{key_base}"
        if tr_key in df.columns: ax.plot(x, df[tr_key], label="Train", color="tab:blue")
        if vl_key in df.columns: ax.plot(x, df[vl_key], label="Valid", color="tab:orange", linestyle="--")
        ax.set_xlabel("epoch"); ax.set_ylabel(ylab)
        if ylim: ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        if _has_data(ax): ax.legend(fontsize=8)

    def _plot_roi(ax, prefix, title, ylab):
        ax.set_title(title, fontsize=10, fontweight="bold")
        for r in range(6):
            name = f"ROI{r+1}"
            col = f"vl_{prefix}_{name}" 
            if col in df.columns:
                ax.plot(x, df[col], label=name, color=ROI_COLORS[r], alpha=0.9)

        ax.set_xlabel("epoch"); ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        if _has_data(ax): ax.legend(fontsize=7, ncol=2)

    _plot_tr_vl(axes[0, 0], "pi_entropy", "PI Entropy", "Entropy")
    _plot_tr_vl(axes[0, 1], "pi_neff", "Neff (Global)", "Neff")
    _plot_roi(axes[0, 2], "pi_neff", "Neff (Per ROI)", "Neff")
    _plot_tr_vl(axes[0, 3], "w_max", "Max(w) (Global)", "max(w)")
    _plot_roi(axes[0, 4], "w_max", "Max(w) (Per ROI)", "max(w)")
    
    l = 0
    _plot_tr_vl(axes[1, 0], f"gnn_gamma_mean_l{l}", f"Gamma Mean (L{l})", "Mean", ylim=(0, 1))
    _plot_tr_vl(axes[1, 1], f"gnn_gamma_std_l{l}", f"Gamma Std (L{l})", "Std")

    ax = axes[1, 2]
    ax.set_title(f"Gamma Min/Max (Valid, L{l})", fontsize=10, fontweight="bold")
    min_k, max_k = f"vl_gnn_gamma_min_l{l}", f"vl_gnn_gamma_max_l{l}"
    if min_k in df.columns: ax.plot(x, df[min_k], label="Min", color="tab:blue", linestyle=":")
    if max_k in df.columns: ax.plot(x, df[max_k], label="Max", color="tab:red", linestyle=":")
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    if _has_data(ax): ax.legend(fontsize=8)

    _plot_tr_vl(axes[1, 3], f"gnn_gamma_low_rate_l{l}", "Low Gamma Rate (<0.5)", "Rate", ylim=(0, 1))

    fig.suptitle("Stage 1 Analysis (PI & Gamma)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / out_name, bbox_inches="tight")
    plt.close(fig)
    return out_dir


def plot_stage2(
    run_dir: Union[str, Path],
    csv_name: str = "stage2_metrics.csv",
    out_dirname: str = "plots",
    out_name: str = "stage2.png",
) -> Path:
    run_dir = Path(run_dir)
    out_dir = run_dir / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / csv_name

    if not csv_path.exists(): return out_dir

    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns or len(df) == 0: return out_dir

    x = df["epoch"].values
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150)

    def _plot(ax, cols, title, ylim=None):
        ax.set_title(title, fontsize=11, fontweight="bold")
        for col, label, color in cols:
            if col in df.columns:
                ax.plot(x, df[col], label=label, linewidth=1.5, color=color)
        if ylim: ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        if _has_data(ax): ax.legend(fontsize=8)

    _plot(axes[0,0], [("vl_d_intra", "d_intra", "tab:blue"), ("vl_d_inter", "d_inter", "tab:orange")], "Intra/Inter Dist")
    _plot(axes[0,1], [("vl_separability", "separability", "tab:green")], "Separability")
    if "vl_separability" in df.columns: axes[0,1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    _plot(axes[0,2], [("vl_sep_dist", "sep_dist", "tab:brown")], "Sep Dist Penalty")

    _plot(axes[1,0], [("vl_roi_mixing", "roi_mixing", "tab:purple")], "ROI Mixing", ylim=(0, 1.1))
    _plot(axes[1,1], [("vl_center_var", "center_var", "tab:red")], "Center Var")
    _plot(axes[1,2], [("vl_ord", "ord", "tab:cyan")], "Ordinal Penalty")

    fig.suptitle("Stage 2: Anchor Space Quality", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / out_name, bbox_inches="tight")
    plt.close(fig)

    plot_dual_projector(run_dir, csv_name=csv_name, out_dirname=out_dirname)
    return out_dir


def plot_dual_projector(
    run_dir: Union[str, Path],
    csv_name: str = "stage2_metrics.csv",
    out_dirname: str = "plots",
    out_name: str = "dual_projector.png",
) -> Path:
    run_dir = Path(run_dir)
    out_dir = run_dir / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(run_dir / csv_name)
    if "epoch" not in df.columns: return out_dir
    x = df["epoch"].values

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    
    def _plot_band(ax, base, color, ylim=None):
        mean, mn, mx = f"vl_{base}", f"vl_{base}_min", f"vl_{base}_max"
        ax.set_title(base, fontsize=11, fontweight="bold")
        if mean in df.columns:
            ax.plot(x, df[mean], color=color, linewidth=2)
            if mn in df.columns and mx in df.columns:
                ax.fill_between(x, df[mn], df[mx], color=color, alpha=0.2)
        if ylim: ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)

    _plot_band(axes[0,0], "u_severity_align", "tab:blue", (0, 1.1))
    _plot_band(axes[0,1], "u_anchor_align", "tab:green", (-0.1, 1.1))
    _plot_band(axes[0,2], "uv_decorr", "tab:purple", (0, 1.1))
    _plot_band(axes[1,0], "v_severity_orth", "tab:orange", (0, 1.1))
    _plot_band(axes[1,1], "v_diversity", "tab:red", (0, 1.1))
    
    ax = axes[1,2]; ax.set_title("Summary", fontsize=11, fontweight="bold")
    for k, c in [("vl_u_severity_align", "tab:blue"), ("vl_u_anchor_align", "tab:green"), ("vl_v_diversity", "tab:red")]:
        if k in df.columns: ax.plot(x, df[k], color=c, label=k.replace("vl_", ""))
    ax.grid(True, alpha=0.3)
    if _has_data(ax): ax.legend(fontsize=7)

    fig.suptitle("Dual Projector Analysis", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / out_name, bbox_inches="tight")
    plt.close(fig)
    return out_dir

def plot_stage3(
    run_dir: Union[str, Path],
    csv_name: str = "stage3_metrics.csv",
    out_dirname: str = "plots",
    out_name: str = "stage3.png",
    num_regions: int = 6,
    num_classes: int = 4,
) -> Path:
    run_dir = Path(run_dir)
    out_dir = run_dir / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / csv_name
    if not csv_path.exists():
        return out_dir
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns or len(df) == 0:
        return out_dir
    x = df["epoch"].values

    has_oracle = any(c.startswith("vl_oracle_") for c in df.columns)
    n_rows = 3 if has_oracle else 2
    fig, axes = plt.subplots(n_rows, 6, figsize=(36, 5.5 * n_rows), dpi=150)

    def _p(ax, title, items, ylim=None):
        ax.set_title(title, fontsize=10, fontweight="bold")
        for col, label, color, ls in items:
            if col in df.columns:
                ax.plot(x, df[col], label=label, color=color, linestyle=ls, linewidth=1.5)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        if _has_data(ax):
            ax.legend(fontsize=7)


    _p(axes[0, 0], "α Entropy", [
        ("tr_alpha_entropy_mean", "Train", "tab:blue", "--"),
        ("vl_alpha_entropy_mean", "Valid", "tab:orange", "-"),
    ])

    _p(axes[0, 1], "α Concentration", [
        ("vl_alpha_top1_mass_mean", "Top-1 Mass", "tab:blue", "-"),
        ("vl_alpha_collapse_rate", "Collapse (>0.9)", "tab:red", "--"),
        ("vl_alpha_self_mean", "Self-Attn", "tab:green", ":"),
    ], ylim=(0, 1.05))

    _p(axes[0, 2], "Gate Open Rates", [
        ("vl_gate_open_rate", "Loose (>0.3)", "tab:blue", "-"),
        ("vl_gate_open_rate_strict", "Strict (>0.7)", "tab:orange", "--"),
        ("vl_gate_closed_rate", "Closed (<0.1)", "tab:red", ":"),
    ], ylim=(0, 1.05))

    ax = axes[0, 3]
    ax.set_title("Gate Value Distribution", fontsize=10, fontweight="bold")
    if "vl_gate_mean" in df.columns:
        ax.plot(x, df["vl_gate_mean"], label="Mean", color="tab:blue", linewidth=2)
    if "vl_gate_min" in df.columns and "vl_gate_max" in df.columns:
        ax.fill_between(x, df["vl_gate_min"], df["vl_gate_max"],
                        color="tab:blue", alpha=0.15, label="Min-Max")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    if _has_data(ax):
        ax.legend(fontsize=7)

    _p(axes[0, 4], "S2 vs S3 Accuracy", [
        ("vl_s2_acc", "S2 Acc", "tab:blue", "--"),
        ("vl_s3_acc", "S3 Acc", "tab:orange", "-"),
        ("vl_s3_net_gain", "Net Gain", "tab:green", ":"),
    ])

    ax = axes[0, 5]
    ax.set_title("Qbar Prediction", fontsize=10, fontweight="bold")
    ax2 = ax.twinx()
    if "vl_qbar_acc" in df.columns:
        ax.plot(x, df["vl_qbar_acc"], label="Acc", color="tab:blue", linewidth=1.5)
    if "vl_qbar_mae" in df.columns:
        ax2.plot(x, df["vl_qbar_mae"], label="MAE", color="tab:red", linewidth=1.5, linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=8)
    ax2.set_ylabel("MAE", fontsize=8)
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)


    _p(axes[1, 0], "Message Ordinal Quality", [
        ("vl_qbar_ord_health", "Ord Health", "tab:green", "-"),
        ("vl_qbar_bimodal_rate", "Bimodal Rate", "tab:red", "--"),
        ("vl_qbar_mode_strength", "Mode Strength", "tab:purple", ":"),
    ], ylim=(0, 1.05))

    ax = axes[1, 1]
    ax.set_title("Qbar Acc by Class", fontsize=10, fontweight="bold")
    class_colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]
    for c in range(num_classes):
        col = f"vl_qbar_acc_c{c}"
        if col in df.columns:
            ax.plot(x, df[col], label=f"C{c}", color=class_colors[c], linewidth=1.5)
    if "vl_qbar_acc" in df.columns:
        ax.plot(x, df["vl_qbar_acc"], label="All", color="black", linewidth=2, linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    if _has_data(ax):
        ax.legend(fontsize=7, ncol=2)

    last_row = df.iloc[-1]
    for c in range(num_classes):
        ax = axes[1, 2 + c]
        mat = np.zeros((num_regions, num_regions))
        for r_src in range(num_regions):
            for r_tgt in range(num_regions):
                k = f"vl_att_map_sum_c{c}_src{r_src}_tgt{r_tgt}"
                if k in last_row and pd.notna(last_row[k]):
                    mat[r_src, r_tgt] = float(last_row[k])

        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=max(mat.max(), 0.01))
        ax.set_title(f"Class {c} α Map", fontsize=10, fontweight="bold")
        ax.set_xlabel("Target")
        if c == 0:
            ax.set_ylabel("Source")
        ax.set_xticks(range(num_regions))
        ax.set_yticks(range(num_regions))
        for i in range(num_regions):
            for j in range(num_regions):
                val = mat[i, j]
                clr = "white" if val < 0.5 else "black"
                if val > 0.01:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=clr)


    if has_oracle:
        def _plot_compare(ax, oracle_col, agg_col, title, ylim):
            ax.set_title(title, fontsize=10, fontweight="bold")
            oc, ac = f"vl_{oracle_col}", f"vl_{agg_col}"
            if oc in df.columns:
                ax.plot(x, df[oc], color="tab:blue", linewidth=2, label="Oracle")
            if ac in df.columns:
                ax.plot(x, df[ac], color="tab:red", linewidth=2, label="Current", linestyle="--")
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)
            if _has_data(ax):
                ax.legend(fontsize=8)

        def _plot_roi(ax, prefix, metric, title, ylim):
            ax.set_title(title, fontsize=10, fontweight="bold")
            for r in range(6):
                name = f"ROI{r+1}"
                col = f"vl_{prefix}_{metric}_{name}"
                if col in df.columns:
                    ax.plot(x, df[col], color=ROI_COLORS[r], label=name, alpha=0.8)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)
            if _has_data(ax):
                ax.legend(fontsize=7, ncol=2)

        _plot_compare(axes[2, 0], "oracle_acc", "agg_acc", "Oracle vs Agg ACC (w/ self)", (0, 1.05))
        _plot_compare(axes[2, 1], "oracle_mae", "agg_mae", "Oracle vs Agg MAE (w/ self)", (0, 1.0))
        _plot_roi(axes[2, 2], "oracle", "acc", "Oracle ACC/ROI", (0, 1.05))
        _plot_roi(axes[2, 3], "oracle", "mae", "Oracle MAE/ROI", (0, 1.0))
        _plot_roi(axes[2, 4], "agg", "acc", "Agg ACC/ROI", (0, 1.05))
        _plot_roi(axes[2, 5], "agg", "mae", "Agg MAE/ROI", (0, 1.0))

    title = "Stage 3: GNN Overview"
    if has_oracle:
        title += " (with Oracle Feasibility)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / out_name, bbox_inches="tight")
    plt.close(fig)
    return out_dir

def plot_attention_analysis(
    run_dir: Union[str, Path],
    csv_name: str = "stage3_metrics.csv",
    out_dirname: str = "plots",
    out_name: str = "attention_analysis.png",
    num_regions: int = 6,
    num_classes: int = 4,
) -> Path:
    run_dir = Path(run_dir)
    csv_path = run_dir / csv_name
    out_dir = run_dir / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        return out_dir
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns or len(df) == 0:
        return out_dir
    x = df["epoch"].values

    fig = plt.figure(figsize=(28, 20), dpi=150)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.2])

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax03 = fig.add_subplot(gs[0, 3])

    def _plot_metric(ax, title, col_suffix, ylim=None, dual=False):
        prefix = "vl"
        candidates = [
            f"{prefix}_{col_suffix}",
            f"{prefix}_gnn_{col_suffix}",
            f"{prefix}_{col_suffix}_mean",
        ]
        col = next((c for c in candidates if c in df.columns), None)

        if dual:
            tr_candidates = [f"tr_{col_suffix}", f"tr_gnn_{col_suffix}", f"tr_{col_suffix}_mean"]
            tr_col = next((c for c in tr_candidates if c in df.columns), None)
            if tr_col is not None:
                ax.plot(x, df[tr_col], color="tab:blue", linestyle="--", linewidth=1.0, alpha=0.6, label="train")

        if col is not None:
            ax.plot(x, df[col], color="tab:orange", linewidth=1.5, label="valid")
            ax.set_title(title, fontsize=10, fontweight="bold")
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.3)
            if _has_data(ax):
                ax.legend(fontsize=8)
        else:
            ax.set_title(title, fontsize=10)
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=14, color="gray")

    _plot_metric(ax00, "α Entropy", "alpha_entropy_mean")
    _plot_metric(ax01, "α Top-1 Mass", "alpha_top1_mass_mean", (0, 1))
    _plot_metric(ax02, "α Collapse Rate", "alpha_collapse_rate", (0, 1))
    _plot_metric(ax03, "α Self-Attention", "alpha_self_mean", (0, 1))

    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])
    ax13 = fig.add_subplot(gs[1, 3])

    ax10.set_title("Gate Open Rates", fontsize=10, fontweight="bold")
    for col, label, color, ls in [
        ("vl_gate_open_rate", "Loose (>0.3)", "tab:blue", "-"),
        ("vl_gate_open_rate_strict", "Strict (>0.7)", "tab:orange", "--"),
        ("vl_gate_closed_rate", "Closed (<0.1)", "tab:red", ":"),
    ]:
        if col in df.columns:
            ax10.plot(x, df[col], label=label, color=color, linestyle=ls, linewidth=1.5)
    ax10.set_ylim(0, 1.05)
    ax10.grid(True, alpha=0.3)
    if _has_data(ax10):
        ax10.legend(fontsize=8)

    ax11.set_title("Message Ordinal Quality", fontsize=10, fontweight="bold")
    for col, label, color in [
        ("vl_qbar_ord_health", "Ord Health", "tab:green"),
        ("vl_qbar_bimodal_rate", "Bimodal Rate", "tab:red"),
        ("vl_qbar_mode_strength", "Mode Strength", "tab:purple"),
    ]:
        if col in df.columns:
            ax11.plot(x, df[col], label=label, color=color, linewidth=1.5)
    ax11.set_ylim(0, 1.05)
    ax11.grid(True, alpha=0.3)
    if _has_data(ax11):
        ax11.legend(fontsize=8)

    ax12.set_title("Qbar Prediction Quality", fontsize=10, fontweight="bold")
    ax12_twin = ax12.twinx()
    for col, label, color, target_ax in [
        ("vl_qbar_acc", "Qbar Acc", "tab:blue", ax12),
        ("tr_qbar_acc", "Qbar Acc (tr)", "tab:blue", ax12),
        ("vl_qbar_mae", "Qbar MAE", "tab:red", ax12_twin),
    ]:
        if col in df.columns:
            ls = "--" if "tr_" in col else "-"
            alpha_val = 0.5 if "tr_" in col else 1.0
            target_ax.plot(x, df[col], label=label, color=color, linestyle=ls, linewidth=1.5, alpha=alpha_val)
    ax12.set_ylim(0, 1.05)
    ax12.set_ylabel("Accuracy")
    ax12_twin.set_ylabel("MAE")
    ax12.grid(True, alpha=0.3)
    lines1, labels1 = ax12.get_legend_handles_labels()
    lines2, labels2 = ax12_twin.get_legend_handles_labels()
    ax12.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    ax13.set_title("Effective Step Size", fontsize=10, fontweight="bold")
    for col, label, color in [
        ("vl_t_eff_mean", "t_eff mean", "tab:blue"),
        ("vl_t_eff_max", "t_eff max", "tab:orange"),
        ("vl_gate_mean", "Gate mean", "tab:green"),
    ]:
        if col in df.columns:
            ax13.plot(x, df[col], label=label, color=color, linewidth=1.5)
    ax13.set_ylim(0, 1.05)
    ax13.grid(True, alpha=0.3)
    if _has_data(ax13):
        ax13.legend(fontsize=8)

    last_row = df.iloc[-1]

    for c in range(num_classes):
        ax = fig.add_subplot(gs[2, c])
        mat = np.zeros((num_regions, num_regions))
        for r_src in range(num_regions):
            for r_tgt in range(num_regions):
                k = f"vl_gnn_att_map_sum_c{c}_src{r_src}_tgt{r_tgt}"
                k_alt = f"vl_att_map_sum_c{c}_src{r_src}_tgt{r_tgt}"
                if k in last_row:
                    mat[r_src, r_tgt] = float(last_row[k])
                elif k_alt in last_row:
                    mat[r_src, r_tgt] = float(last_row[k_alt])

        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1.0)
        ax.set_title(f"Class {c} Attention\n(Src→Tgt)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Target ROI")
        if c == 0:
            ax.set_ylabel("Source ROI")
        ax.set_xticks(range(num_regions))
        ax.set_yticks(range(num_regions))

        for i in range(num_regions):
            for j in range(num_regions):
                val = mat[i, j]
                color = "white" if val < 0.5 else "black"
                if val > 0.01:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.suptitle("Attention Analysis & Ordinal Gate Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / out_name, bbox_inches="tight")
    plt.close(fig)
    return out_dir / out_name

def plot_move_stats(
    run_dir: str | Path,
    csv_name: str = "stage3_metrics.csv",
    out_dirname: str = "plots",
    out_name: str = "move_stats.png",
    prefix: str = "vl_move", 
    num_regions: int = 6,
    num_classes: int = 4,
    annotate_min_count: int = 1,
) -> Path:
    run_dir = Path(run_dir)
    out_dir = run_dir / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    csv_path = run_dir / csv_name
    if not csv_path.exists(): return out_dir
    df = pd.read_csv(csv_path)
    if len(df) == 0: return out_dir

    row = df.iloc[-1]

    def _get(col_base: str):
        k = f"{prefix}_{col_base}"
        if k in df.columns:
            v = row[k]
            return None if pd.isna(v) else float(v)
        return None

    cmap_w2b = LinearSegmentedColormap.from_list("white_to_blue", ["white", "blue"])
    count_mats = [[None for _ in range(num_classes)] for __ in range(num_regions)]
    has_any = False
    global_vmax = 1

    for r in range(num_regions):
        for g in range(num_classes):
            pre_cnt = np.zeros((num_classes,), dtype=float)
            for x in range(num_classes):
                v = _get(f"precount_r{r}_g{g}_x{x}")
                pre_cnt[x] = 0.0 if v is None else float(v)

            M = np.zeros((num_classes, num_classes), dtype=float)
            for x in range(num_classes):
                denom = pre_cnt[x]
                if denom <= 0: continue
                for y in range(num_classes):
                    ratio = _get(f"ratio_r{r}_g{g}_x{x}_y{y}")
                    if ratio is None: continue
                    M[x, y] = float(ratio) * float(denom)
                    has_any = True

            M = np.rint(M).astype(int)
            count_mats[r][g] = M
            if M.size > 0:
                global_vmax = max(global_vmax, int(M.max()))

    if not has_any: return out_dir

    fig_w = 4.2 * num_classes
    fig_h = 3.2 * num_regions
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    gs = fig.add_gridspec(num_regions, num_classes, wspace=0.25, hspace=0.55)

    for r in range(num_regions):
        for g in range(num_classes):
            ax = fig.add_subplot(gs[r, g])
            M = count_mats[r][g]
            if M is None: M = np.zeros((num_classes, num_classes), dtype=int)

            im = ax.imshow(M, aspect="auto", vmin=0, vmax=global_vmax, cmap=cmap_w2b, origin="lower")
            ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
            ax.set_yticklabels(list(range(num_classes)))

            rect = patches.Rectangle((g - 0.5, -0.5), 1.0, float(num_classes), fill=False, edgecolor="black", linewidth=4.0, zorder=50, clip_on=False)
            ax.add_patch(rect)

            for x in range(num_classes):
                for y in range(num_classes):
                    c = int(M[x, y])
                    if c >= annotate_min_count:
                        tag = "stay" if x == y else "move"
                        ax.text(y, x, f"{c}\n{tag}", ha="center", va="center", fontsize=7.5, color="black")

    fig.suptitle("ROI (rows) × GT (cols): Class Transition Counts (u → h)\nEach subplot: pre=X (rows) -> post=Y (cols)", fontsize=14, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_gnn_dynamics(run_dir):
    run_dir = Path(run_dir)
    csv_path = run_dir / "stage3_metrics.csv"

    if not csv_path.exists():
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    if "epoch" not in df.columns or len(df) == 0:
        return

    epochs = df["epoch"]

    fig, axes = plt.subplots(3, 4, figsize=(28, 18), dpi=150)
    axes = axes.flatten()


    ax = axes[0]
    if "tr_dyn_TUR" in df.columns:
        ax.plot(epochs, df["tr_dyn_TUR"], label="Tr TUR (Toxic)", color="red", linestyle="--")
        ax.plot(epochs, df["tr_dyn_BFR"], label="Tr BFR (Beneficial)", color="blue", linestyle="--")
    if "vl_dyn_TUR" in df.columns:
        ax.plot(epochs, df["vl_dyn_TUR"], label="Vl TUR", color="red")
        ax.plot(epochs, df["vl_dyn_BFR"], label="Vl BFR", color="blue")
    ax.set_title("GNN Update Quality")
    ax.set_ylabel("Rate")
    ax.grid(True, alpha=0.3)
    if _has_data(ax):
        ax.legend(fontsize=8)

    ax = axes[1]
    if "tr_dyn_NC" in df.columns:
        ax.plot(epochs, df["tr_dyn_NC"], label="Train NC", color="green", linestyle="--")
    if "vl_dyn_NC" in df.columns:
        ax.plot(epochs, df["vl_dyn_NC"], label="Valid NC", color="green")
    ax.axhline(0, color="black", linestyle=":")
    ax.set_title("Net Correction (>0 better)")
    ax.grid(True, alpha=0.3)
    if _has_data(ax):
        ax.legend(fontsize=8)

    ax = axes[2]
    if "tr_struct_AvgSim" in df.columns:
        ax.plot(epochs, df["tr_struct_AvgSim"], label="Train Sim", color="orange", linestyle="--")
    if "vl_struct_AvgSim" in df.columns:
        ax.plot(epochs, df["vl_struct_AvgSim"], label="Valid Sim", color="orange")
    ax.set_title("Structure Embedding Similarity")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    if _has_data(ax):
        ax.legend(fontsize=8)

    ax = axes[3]
    if "tr_prior_PLC" in df.columns:
        ax.plot(epochs, df["tr_prior_PLC"], label="Train PLC", color="purple", linestyle="--")
    if "vl_prior_PLC" in df.columns:
        ax.plot(epochs, df["vl_prior_PLC"], label="Valid PLC", color="purple")
    ax.set_title("Prior-Label Consistency (PLC)")
    ax.grid(True, alpha=0.3)
    if _has_data(ax):
        ax.legend(fontsize=8)


    ax = axes[4]
    ax2 = ax.twinx()
    if "tr_tf_ratio" in df.columns:
        ax.fill_between(epochs, 0, df["tr_tf_ratio"], color="gray", alpha=0.1, label="TF Ratio")
    if "tr_tf_p_argmax_acc" in df.columns:
        ax2.plot(epochs, df["tr_tf_p_argmax_acc"], color="tab:blue", label="P Acc (Model)", linewidth=1.5)
    if "tr_tf_qbar_model_acc" in df.columns:
        ax2.plot(epochs, df["tr_tf_qbar_model_acc"], color="tab:green", label="Qbar Acc (Model)", linewidth=1.5)
    ax.set_title("Teacher Forcing & Model Autonomy")
    ax.set_ylabel("TF Ratio")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper left")

    ax = axes[5]
    if "tr_s2s3_pred_agreement" in df.columns:
        ax.plot(epochs, df["tr_s2s3_pred_agreement"], label="Agreement (h=u)", color="gray", linestyle=":")
    if "tr_s3_net_gain" in df.columns:
        ax.plot(epochs, df["tr_s3_net_gain"], label="Net Gain (Imp-Deg)", color="tab:red", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_title("S2 vs S3 Interaction")
    ax.grid(True, alpha=0.3)
    if _has_data(ax):
        ax.legend(fontsize=8)

    ax = axes[6]
    if "tr_s3_mae_delta" in df.columns:
        ax.plot(epochs, df["tr_s3_mae_delta"], label="MAE Delta (S2-S3)", color="tab:green")
        ax.fill_between(epochs, 0, df["tr_s3_mae_delta"],
                        where=(df["tr_s3_mae_delta"] > 0), color="tab:green", alpha=0.2)
        ax.fill_between(epochs, 0, df["tr_s3_mae_delta"],
                        where=(df["tr_s3_mae_delta"] < 0), color="tab:red", alpha=0.2)
    ax.axhline(0, color="black", linestyle="-")
    ax.set_title("S3 Benefit (>0 means S3 better)")
    ax.set_ylabel("MAE Reduction")
    ax.grid(True, alpha=0.3)
    if _has_data(ax):
        ax.legend(fontsize=8)

    ax = axes[7]
    if "tr_scale_s2" in df.columns:
        ax.plot(epochs, df["tr_scale_s2"], label="S2 (tr)", color="tab:blue", linestyle="--")
    if "tr_scale_s3" in df.columns:
        ax.plot(epochs, df["tr_scale_s3"], label="S3 (tr)", color="tab:orange", linestyle="--")
    if "vl_scale_s2" in df.columns:
        ax.plot(epochs, df["vl_scale_s2"], label="S2 (vl)", color="tab:blue")
    if "vl_scale_s3" in df.columns:
        ax.plot(epochs, df["vl_scale_s3"], label="S3 (vl)", color="tab:orange")
    ax.set_title("Logit Scales (Sharpness)")
    ax.grid(True, alpha=0.3)
    if _has_data(ax):
        ax.legend(fontsize=8)


    ax = axes[8]
    ax.set_title("Gate Statistics", fontsize=10, fontweight="bold")
    for col, label, color, ls in [
        ("vl_gate_mean", "Mean", "tab:blue", "-"),
        ("vl_gate_min", "Min", "tab:cyan", ":"),
        ("vl_gate_max", "Max", "#000080", ":"),
    ]:
        if col in df.columns:
            ax.plot(epochs, df[col], label=label, color=color, linestyle=ls, linewidth=1.5)
    if "vl_gate_min" in df.columns and "vl_gate_max" in df.columns:
        ax.fill_between(epochs, df["vl_gate_min"], df["vl_gate_max"], color="tab:blue", alpha=0.1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    if _has_data(ax):
        ax.legend(fontsize=8)

    ax = axes[9]
    ax.set_title("Message Ordinal Quality", fontsize=10, fontweight="bold")
    for prefix, ls, alpha_val in [("tr", "--", 0.5), ("vl", "-", 1.0)]:
        for col_base, label_base, color in [
            ("qbar_ord_health", "Ord Health", "tab:green"),
            ("qbar_bimodal_rate", "Bimodal Rate", "tab:red"),
        ]:
            col = f"{prefix}_{col_base}"
            if col in df.columns:
                label = f"{label_base} ({'tr' if prefix == 'tr' else 'vl'})"
                ax.plot(epochs, df[col], label=label, color=color, linestyle=ls, linewidth=1.5, alpha=alpha_val)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    if _has_data(ax):
        ax.legend(fontsize=7, ncol=2)

    ax = axes[10]
    ax.set_title("Qbar Accuracy by Class", fontsize=10, fontweight="bold")
    class_colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]
    for c in range(4):
        col = f"vl_qbar_acc_c{c}"
        if col in df.columns:
            ax.plot(epochs, df[col], label=f"C{c}", color=class_colors[c], linewidth=1.5)
    if "vl_qbar_acc" in df.columns:
        ax.plot(epochs, df["vl_qbar_acc"], label="Overall", color="black", linewidth=2, linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    if _has_data(ax):
        ax.legend(fontsize=8)

    ax = axes[11]
    ax.set_title("Update Magnitude", fontsize=10, fontweight="bold")
    ax2 = ax.twinx()
    for col, label, color, target_ax, ls in [
        ("vl_t_eff_mean", "t_eff mean", "tab:blue", ax, "-"),
        ("vl_t_eff_max", "t_eff max", "tab:cyan", ax, ":"),
        ("vl_dyn_angular_move", "Angular Move", "tab:orange", ax2, "-"),
    ]:
        if col in df.columns:
            target_ax.plot(epochs, df[col], label=label, color=color, linestyle=ls, linewidth=1.5)
    ax.set_ylabel("t_eff")
    ax.set_ylim(0, 1.05)
    ax2.set_ylabel("Angular Move (rad)")
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.suptitle("GNN Training Dynamics (with Ordinal Gate Diagnostics)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / "stage3_dynamics.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_stage4(
    run_dir: Union[str, Path],
    csv_name: str = "stage4_metrics.csv",
    gnn_csv_name: str = "stage3_metrics.csv",
    out_dirname: str = "plots",
    out_name: str = "stage4.png",
) -> Path:
    run_dir = Path(run_dir)
    out_dir = run_dir / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = run_dir / csv_name
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "epoch" in df.columns and len(df) > 0:
            x = df["epoch"].values
            METRICS = ["Acc", "MAE", "MEr", "SD", "CC"]
            fig, axes = plt.subplots(3, 5, figsize=(20, 12), dpi=150)
            
            def _plot_metric(row_idx, metric, prefix_list):
                ax = axes[row_idx, METRICS.index(metric)]
                title_prefix = ["Global", "Avg", "ROI"][row_idx]
                ax.set_title(f"{title_prefix} {metric}", fontsize=11, fontweight="bold")
                
                for prefix, col_pattern, color, label in prefix_list:
                    col = col_pattern.format(metric=metric)
                    if col in df.columns:
                        ls = "--" if "vl" in prefix else "-"
                        ax.plot(x, df[col], color=color, linestyle=ls, label=label, linewidth=1.5)
                ax.grid(True, alpha=0.3)
                if _has_data(ax): ax.legend(fontsize=8)

            for m in METRICS:
                _plot_metric(0, m, [("tr", "tr_global_{metric}", "tab:blue", "tr"), ("vl", "vl_global_{metric}", "tab:orange", "vl")])

            for m in METRICS:
                _plot_metric(1, m, [("tr", "tr_avg_{metric}", "tab:blue", "tr"), ("vl", "vl_avg_{metric}", "tab:orange", "vl")])

            for j, metric in enumerate(METRICS):
                ax = axes[2, j]
                ax.set_title(f"ROI {metric} (Valid)", fontsize=11, fontweight="bold")
                for r, name in enumerate(["ROI1", "ROI2", "ROI3", "ROI4", "ROI5", "ROI6"]):
                    col = f"vl_{name}_{metric}"
                    if col in df.columns:
                        ax.plot(x, df[col], color=ROI_COLORS[r%6], label=name, linewidth=1.5, alpha=0.8)
                ax.grid(True, alpha=0.3)
                if _has_data(ax): ax.legend(fontsize=7, ncol=2)

            fig.suptitle("Stage 4: BSN Metrics", fontsize=14, fontweight="bold", y=1.01)
            fig.tight_layout()
            fig.savefig(out_dir / out_name, bbox_inches="tight")
            plt.close(fig)

    plot_move_stats(run_dir, csv_name=gnn_csv_name, out_dirname=out_dirname)
    plot_attention_analysis(run_dir, csv_name=gnn_csv_name, out_dirname=out_dirname)
    plot_gnn_dynamics(run_dir)

    return out_dir

def plot_attention_sample(
    q_ij: torch.Tensor,
    alpha_agg: torch.Tensor,
    label: torch.Tensor,
    run_dir: Path,
    epoch: int,
    alpha_target: Optional[torch.Tensor] = None,
    pi_oracle_single: Optional[torch.Tensor] = None,
    roi_names: list = ["ROI1", "ROI2", "ROI3", "ROI4", "ROI5", "ROI6"],
    oracle_tau: float = 0.1,
):
    q_ij = q_ij.detach().float().cpu()
    alpha_agg = alpha_agg.detach().float().cpu()
    label = label.detach().cpu()

    R, _, C = q_ij.shape
    class_idx = torch.arange(C).float()

    if alpha_target is not None:
        alpha_oracle = alpha_target.detach().float().cpu()
    else:
        if pi_oracle_single is not None:
            pi_s = pi_oracle_single.detach().float().cpu()
            p_gt = F.one_hot(label.long(), num_classes=C).float()
            q_ij_gt = torch.einsum("ijcd,jd->ijc", pi_s, p_gt)
        else:
            q_ij_gt = q_ij

        E_q_gt = (q_ij_gt * class_idx).sum(dim=-1)
        dist_gt = (E_q_gt - label.float().unsqueeze(-1)).abs()
        alpha_oracle = F.softmax(-dist_gt / oracle_tau, dim=-1)

    q_bar_model = (q_ij * alpha_agg.unsqueeze(-1)).sum(dim=0)
    E_q_model = (q_bar_model * class_idx).sum(dim=-1)

    q_bar_oracle = (q_ij * alpha_oracle.unsqueeze(-1)).sum(dim=0)
    E_q_oracle = (q_bar_oracle * class_idx).sum(dim=-1)

    alpha_agg_np = alpha_agg.numpy()
    alpha_oracle_np = alpha_oracle.numpy()
    stats_np = torch.stack([E_q_model, E_q_oracle, label.float()], dim=1).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), gridspec_kw={'width_ratios': [1, 1, 0.4]})

    sns.heatmap(alpha_agg_np, ax=axes[0], cmap="Blues", vmin=0, vmax=1.0, annot=True, fmt=".2f", cbar=False, xticklabels=roi_names, yticklabels=roi_names)
    axes[0].set_title("Model Attention (alpha)")

    sns.heatmap(alpha_oracle_np, ax=axes[1], cmap="Blues", vmin=0, vmax=1.0, annot=True, fmt=".2f", cbar=False, xticklabels=roi_names, yticklabels=roi_names)
    axes[1].set_title("Oracle Attention (GT π + GT p)")

    sns.heatmap(stats_np, ax=axes[2], cmap="Reds", annot=True, fmt=".2f", xticklabels=["Model", "Oracle", "GT"], yticklabels=roi_names, cbar=False)
    axes[2].set_title("E[q_bar] Comparison")

    save_dir = Path(run_dir) / "attention_samples"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"attn_ep{epoch:03d}.png", bbox_inches="tight")
    plt.close()


def plot_qbar_roi(
    qbar: torch.Tensor,
    labels: torch.Tensor,
    run_dir: Union[str, Path],
    epoch: int,
    num_classes: int = 4,
    roi_names: list = None,
):
    run_dir = Path(run_dir)
    save_dir = run_dir / "qbar_roi"
    save_dir.mkdir(parents=True, exist_ok=True)

    qbar = qbar.detach().float().cpu()
    labels = labels.detach().long().cpu()
    N, R, C = qbar.shape
    C = min(C, num_classes)

    if roi_names is None:
        roi_names = [f"ROI{r+1}" for r in range(R)]

    class_colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]
    if len(class_colors) < C:
        class_colors = class_colors + ["gray"] * (C - len(class_colors))

    class_idx = torch.arange(C).float()
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=150)

    for r in range(min(R, 6)):
        ax = axes[r // 3, r % 3]
        q_r = qbar[:, r, :C]
        y_r = labels[:, r]

        pred_r = q_r.argmax(-1)
        acc = (pred_r == y_r).float().mean().item()
        E_q = (q_r * class_idx).sum(-1)
        mae = (E_q - y_r.float()).abs().mean().item()

        bar_width = 0.18
        x_pos = np.arange(C)
        per_class_acc = []

        for gt_c in range(C):
            mask = (y_r == gt_c)
            if mask.sum() == 0:
                for pred_c in range(C):
                    ax.bar(gt_c + (pred_c - (C-1)/2) * bar_width, 0,
                           width=bar_width, color=class_colors[pred_c],
                           edgecolor="white", linewidth=0.5)
                per_class_acc.append(0.0)
                continue

            avg_dist = q_r[mask].mean(dim=0).numpy()
            for pred_c in range(C):
                ax.bar(gt_c + (pred_c - (C-1)/2) * bar_width, avg_dist[pred_c],
                       width=bar_width, color=class_colors[pred_c],
                       edgecolor="white", linewidth=0.5,
                       label=f"P(pred={pred_c})" if (r == 0 and gt_c == 0) else "")

            c_acc = (pred_r[mask] == gt_c).float().mean().item()
            per_class_acc.append(c_acc)

        ax2 = ax.twinx()
        ax2.plot(x_pos, per_class_acc, "k-o", markersize=5, linewidth=1.5, alpha=0.7, label="Acc")
        ax2.set_ylim(0, 1.15)
        ax2.set_ylabel("Accuracy", fontsize=8)

        n_per_class = [(y_r == c).sum().item() for c in range(C)]
        n_str = "/".join(str(n) for n in n_per_class)

        ax.set_title(f"{roi_names[r]}  Acc={acc:.3f}  MAE={mae:.3f}\nn=[{n_str}]",
                     fontsize=10, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"GT={c}" for c in range(C)])
        ax.set_ylabel("Avg P(pred)", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2, axis="y")

    handles = [plt.Rectangle((0,0), 1, 1, facecolor=class_colors[c]) for c in range(C)]
    handle_labels = [f"P(pred={c})" for c in range(C)]
    axes[0, 0].legend(handles, handle_labels, fontsize=7, loc="upper right", ncol=2)

    fig.suptitle(f"Q̄ Per-ROI Analysis (Epoch {epoch})", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_dir / f"qbar_roi_ep{epoch:03d}.png", bbox_inches="tight")
    plt.close(fig)


def plot_tsne_stages(
    stage_dict: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    save_path: Path,
    epoch: int,
    anchors: Optional[torch.Tensor] = None,
    max_points_per_stage: int = 2000,
):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if not stage_dict: return
    labels = labels.detach().cpu().long()

    def _prio(k: str) -> int:
        s = k.lower()
        if "(z" in s or "stage1" in s: return 0
        if "(h" in s or "stage2" in s: return 1
        if "(h'" in s or "h'" in s or "stage3" in s: return 2
        return 9

    stage_names = sorted(list(stage_dict.keys()), key=lambda k: (_prio(k), k))
    first = stage_dict[stage_names[0]]
    N, R, _ = first.shape
    total = N * R
    n_take = min(max_points_per_stage, total)

    perm = torch.randperm(total)[:n_take]
    roi_ids_all = (perm % R).cpu()
    sample_ids = (perm // R).cpu()
    cls_ids_all = labels[sample_ids, roi_ids_all].cpu()

    feats = {}
    dims = {}
    for name in stage_names:
        x = stage_dict[name].detach().cpu()
        dims[name] = x.shape[2]
        feats[name] = x.reshape(total, x.shape[2])[perm]

    z_name = stage_names[0]
    other_names = stage_names[1:]
    can_anchor_other = (anchors is not None and len(other_names) > 0 and len(set(dims[n] for n in other_names)) == 1 and anchors.shape[-1] == dims[other_names[0]])

    Ys_tsne = {}
    tsne_z = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=min(30, max(5, (feats[z_name].shape[0] - 1) // 10)), random_state=0)
    Ys_tsne[z_name] = tsne_z.fit_transform(feats[z_name].numpy())

    Y_anchors_tsne = None
    if len(other_names) > 0:
        X_other = torch.cat([feats[n] for n in other_names], dim=0).numpy()
        if can_anchor_other:
            A = anchors.detach().cpu().float().numpy()
            X_fit = np.concatenate([X_other, A], axis=0)
        else:
            X_fit = X_other

        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=min(30, max(5, (X_fit.shape[0] - 1) // 10)), random_state=0)
        Y_fit = tsne.fit_transform(X_fit)
        Y_other = Y_fit[: X_other.shape[0]]
        if can_anchor_other:
            Y_anchors_tsne = Y_fit[X_other.shape[0] :]

        off = 0
        for n in other_names:
            Ys_tsne[n] = Y_other[off : off + n_take]
            off += n_take

    Ys_pca = {}
    pca_z = PCA(n_components=2, random_state=0)
    Ys_pca[z_name] = pca_z.fit_transform(feats[z_name].numpy())

    Y_anchors_pca = None
    if len(other_names) > 0:
        X_other = torch.cat([feats[n] for n in other_names], dim=0).numpy()
        if can_anchor_other:
            A = anchors.detach().cpu().float().numpy()
            X_fit = np.concatenate([X_other, A], axis=0)
        else:
            X_fit = X_other

        pca = PCA(n_components=2, random_state=0)
        Y_fit = pca.fit_transform(X_fit)
        Y_other = Y_fit[: X_other.shape[0]]
        if can_anchor_other:
            Y_anchors_pca = Y_fit[X_other.shape[0] :]

        off = 0
        for n in other_names:
            Ys_pca[n] = Y_other[off : off + n_take]
            off += n_take

    roi_colors = [ROI_COLORS[i % len(ROI_COLORS)] for i in range(R)]
    class_markers = CLASS_MARKERS
    n_classes = int(cls_ids_all.max().item()) + 1 if cls_ids_all.numel() else 1
    n_classes = max(1, n_classes)

    def _draw_row(axes_row, Ys, Y_anchors, row_title: str):
        for ax, name in zip(axes_row, stage_names):
            Y = Ys[name]
            for r in range(R):
                rmask = (roi_ids_all == r).numpy()
                if not rmask.any(): continue
                for c in range(n_classes):
                    cmask = (cls_ids_all == c).numpy()
                    mask = rmask & cmask
                    if not mask.any(): continue
                    mk = class_markers[c % len(class_markers)]
                    ax.scatter(Y[mask, 0], Y[mask, 1], s=18, marker=mk, alpha=0.80, facecolors=roi_colors[r], edgecolors="k", linewidths=0.25)

            if (Y_anchors is not None) and (name != z_name):
                ax.scatter(Y_anchors[:, 0], Y_anchors[:, 1], s=140, marker="X", facecolors="k", edgecolors="k", alpha=0.95, linewidths=0.8, zorder=5)
                for i in range(Y_anchors.shape[0]):
                    ax.text(Y_anchors[i, 0], Y_anchors[i, 1], f"C{i}", fontsize=11, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.9), zorder=6)

            ax.set_title(f"{name}\n{row_title}", fontsize=11, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    fig, axes = plt.subplots(2, len(stage_names), figsize=(6.5 * len(stage_names), 12), squeeze=False)
    _draw_row(axes[0], Ys_tsne, Y_anchors_tsne, f"t-SNE (Epoch {epoch})")
    _draw_row(axes[1], Ys_pca,  Y_anchors_pca,  f"PCA (Epoch {epoch})")

    roi_handles = [Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=roi_colors[r], markeredgecolor="k", markersize=8, label=ROI_NAMES[r]) for r in range(R)]
    cls_handles = [Line2D([0], [0], marker=class_markers[c % len(class_markers)], linestyle="None", markerfacecolor="white", markeredgecolor="k", markersize=8, label=f"C{c}") for c in range(n_classes)]

    ax_leg = axes[0, -1]
    leg1 = ax_leg.legend(handles=roi_handles, title="ROI", loc="upper left", bbox_to_anchor=(1.02, 1.00), borderaxespad=0.0, fontsize=9, title_fontsize=10, frameon=True)
    ax_leg.add_artist(leg1)
    ax_leg.legend(handles=cls_handles, title="Class", loc="upper left", bbox_to_anchor=(1.02, 0.55), borderaxespad=0.0, fontsize=9, title_fontsize=10, frameon=True)

    fig.tight_layout()
    fig.savefig(save_path / f"tsne_ep{epoch:03d}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
