from __future__ import annotations

import math
import time
import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import pandas as pd
import numpy as np

from dorga.models.backbone import load_mae_ckpt_to_512
from dorga.models.pattern_prior import PatternPredictor, DynamicPrior, pattern_loss
from dorga.models.patch_importance import ConvPatchImportance, Pooler_Box
from dorga.models.projectors import SharedProjector, ROISpecificProjector
from dorga.models.graph_attention import DecoupledOrdinalGraphAttentionNet
from dorga.models.classifier import AngularClassifier
from dorga.models.dorga_model import BrixiaViT512Dynamic
from dorga.losses.losses import (
    compute_alpha_oracle, loss_function, loss_function_projection,
    compute_class_counts_from_df, kl_attention_loss,
)
from dorga.utils.training import _make_optimizer, _set_trainable
from dorga.utils.monitor import (
    StageMonitor, compute_move_stats, compute_dual_proj_stats,
    compute_anchor_metrics, bsn_metrics, compute_pi_stats,
    compute_gnn_stats,
    compute_pattern_stats, compute_tf_stats,
    compute_update_dynamics,
    compute_s2_s3_gap,
    compute_scale_stats,
    compute_structure_diversity,
    compute_prior_consistency,
    compute_oracle_feasibility,
)
from dorga.utils.visualization import (
    plot_losses, plot_sample_overlay, plot_stage1, plot_stage2,
    plot_stage3, plot_stage4, plot_tsne_stages,
    plot_attention_sample, plot_qbar_roi,
)


def train_dynamic(
    model: BrixiaViT512Dynamic,
    loaders: dict,
    run_dir: Path,
    epochs: int,
    roi_class: torch.Tensor,
    loss_func_stage3: str = "RoiClass",
    plot_every: int = 1,
):
    device = next(model.parameters()).device
    roi_class = roi_class.to(device)
    run_dir = Path(run_dir)

    _set_trainable(model, freeze_blocks=6)
    optimizer = _make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    best_val_avg_mae = float("inf")

    mon_loss   = StageMonitor(run_dir, "loss",   plot_func=plot_losses)
    mon_stage1 = StageMonitor(run_dir, "stage1", plot_func=plot_stage1)
    mon_stage2 = StageMonitor(run_dir, "stage2", plot_func=plot_stage2)
    mon_stage3 = StageMonitor(run_dir, "stage3", plot_func=plot_stage3)
    mon_stage4 = StageMonitor(run_dir, "stage4", plot_func=plot_stage4)

    for epoch in range(1, epochs + 1):

        model.train()
        tr_preds, tr_labs = [], []

        for batch_idx, batch in enumerate(loaders["train"]):
            if len(batch) == 6:
                imgs, lab, rel, roi_masks, pattern_gt, _ = batch
            else:
                imgs, lab, rel, roi_masks, _ = batch
                pattern_gt = None

            imgs, lab, rel = imgs.to(device), lab.to(device), rel.to(device)
            roi_masks = roi_masks.to(device)
            if pattern_gt is not None:
                pattern_gt = pattern_gt.to(device)
            B = imgs.size(0)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda"):
                out = model(imgs, rel, masks=roi_masks, labels=lab)

                loss_s1 = pattern_loss(out["rho"], pattern_gt)
                loss_s2_ce = loss_function(out["logits_s2"], lab, roi_class=roi_class, option=loss_func_stage3)
                loss_s2_proj, _ = loss_function_projection(out["u"], lab, model.class_anchors, roi_class)

                v = out["v"]
                a = F.normalize(model.class_anchors, dim=-1)
                w_vec = F.normalize(a[model.C - 1] - a[0], dim=-1)
                v_w_cos = (v * w_vec).sum(dim=-1).clamp(-1.0, 1.0)
                loss_v_orth = (torch.acos(v_w_cos) - math.pi / 2).abs().mean()
                loss_s2 = loss_s2_ce + loss_s2_proj + loss_v_orth

                loss_s3 = loss_function(out["logits_s3"], lab, roi_class=roi_class, option=loss_func_stage3)

                alpha_pred = out["gnn_out"]["attached"]["alpha"][-1]
                rho_gt = F.one_hot(pattern_gt.long(), num_classes=model.K).float()
                pi_oracle_target = model.dynamic_prior(rho_gt)
                alpha_target, confidence = compute_alpha_oracle(pi_gt=pi_oracle_target.detach(), labels=lab, p_model_logits=out["logits_s2"].detach(), oracle_tau=0.1,)

                loss_att = kl_attention_loss(alpha_pred, alpha_target, confidence)
                total_loss = loss_s1 + loss_s2 + loss_s3 + loss_att
                
                tr_oracle_ent = -(alpha_target * (alpha_target + 1e-8).log()).sum(-1).mean().item()
                if batch_idx == 0 and epoch % plot_every == 0:
                    plot_sample_overlay(imgs[0], roi_masks[0], rel[0], lab[0], out["logits_s3"][0], out["w"][0], run_dir, epoch)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()


            mon_loss.update({
                "loss": total_loss, "pat": loss_s1,
                "s2_ce": loss_s2_ce, "s2_proj": loss_s2_proj,
                "v_orth": loss_v_orth,
                "s3": loss_s3, "att": loss_att,
            }, n=B, prefix="tr")

            pi_stats = compute_pi_stats(out["w"], out["entropy"])
            pat_stats, _ = compute_pattern_stats(out["gnn_out"])
            pi_stats.update(pat_stats)
            mon_stage1.update(pi_stats, n=B, prefix="tr")

            mon_stage2.update(compute_dual_proj_stats(out["u"], out["v"], lab, model.class_anchors), n=B, prefix="tr", )

            gnn_stats, gnn_counts = compute_gnn_stats(out["gnn_out"], labels=lab)
            mon_stage3.update(gnn_stats, n=B, prefix="tr", specific_n=gnn_counts)

            scale_stats = compute_scale_stats(model)        
            mon_stage3.update(scale_stats, n=1, prefix="tr",specific_n={k: 1 for k in scale_stats})

            mon_stage3.update(compute_s2_s3_gap(out["logits_s2"], out["logits_s3"], lab), n=B, prefix="tr")
            mon_stage3.update(compute_update_dynamics(out["u"], out["h"], lab, model.class_anchors), n=B, prefix="tr")
            mon_stage3.update(compute_prior_consistency(out["gnn_out"]["raw"]["pi_refined"][-1], lab), n=B, prefix="tr")
            mon_stage3.update(compute_structure_diversity(out["v"]), n=B, prefix="tr")

            tf_stats = compute_tf_stats(out["gnn_out"], out["logits_s2"], lab)
            mon_stage3.update(tf_stats, n=B, prefix="tr")

            tr_preds.append(out["logits_s3"].detach().cpu())
            tr_labs.append(lab.cpu())

        model.eval()
        vl_preds_pre, vl_preds, vl_labs = [], [], []
        vl_z_buf, vl_u_buf, vl_v_buf, vl_h_buf = [], [], [], []
        vl_qbar_buf = []

        with torch.no_grad():
            for vl_batch_idx, batch in enumerate(loaders["test"]):
                if len(batch) == 6:
                    imgs, lab, rel, roi_masks, pattern_gt, _ = batch
                else:
                    imgs, lab, rel, roi_masks, _ = batch
                    pattern_gt = None

                imgs, lab, rel = imgs.to(device), lab.to(device), rel.to(device)
                roi_masks = roi_masks.to(device)
                if pattern_gt is not None:
                    pattern_gt = pattern_gt.to(device)
                B = imgs.size(0)

                out = model(imgs, rel, masks=roi_masks, labels=lab)

                loss_s1 = pattern_loss(out["rho"], pattern_gt)
                loss_s2_ce = loss_function(out["logits_s2"], lab, roi_class=roi_class, option=loss_func_stage3)
                loss_s2_proj, _ = loss_function_projection(out["u"], lab, model.class_anchors, roi_class)

                v = out["v"]
                a = F.normalize(model.class_anchors, dim=-1)
                w_vec = F.normalize(a[model.C - 1] - a[0], dim=-1)
                v_w_cos = (v * w_vec).sum(dim=-1).clamp(-1.0, 1.0)
                loss_v_orth = (torch.acos(v_w_cos) - math.pi / 2).abs().mean()
                loss_s2 = loss_s2_ce + loss_s2_proj + loss_v_orth 

                loss_s3 = loss_function(out["logits_s3"], lab, roi_class=roi_class, option=loss_func_stage3)

                alpha_pred = out["gnn_out"]["attached"]["alpha"][-1]
                rho_gt = F.one_hot(pattern_gt.long(), num_classes=model.K).float()
                pi_oracle_target = model.dynamic_prior(rho_gt)
                alpha_target, confidence = compute_alpha_oracle(pi_gt=pi_oracle_target.detach(), labels=lab, p_model_logits=out["logits_s2"].detach(), oracle_tau=0.1,)
                vl_oracle_ent = -(alpha_target * (alpha_target + 1e-8).log()).sum(-1).mean().item()
 
                loss_att = kl_attention_loss(alpha_pred, alpha_target, confidence)
                total_loss = loss_s1 + loss_s2 + loss_s3 + loss_att


                mon_loss.update({
                    "loss": total_loss, "pat": loss_s1,
                    "s2_ce": loss_s2_ce, "s2_proj": loss_s2_proj,
                    "v_orth": loss_v_orth,
                    "s3": loss_s3, "att": loss_att,
                }, n=B, prefix="vl")

                pi_stats = compute_pi_stats(out["w"], out["entropy"])
                pat_stats, _ = compute_pattern_stats(out["gnn_out"])
                pi_stats.update(pat_stats)
                mon_stage1.update(pi_stats, n=B, prefix="vl")

                mon_stage2.update(compute_dual_proj_stats(out["u"], out["v"], lab, model.class_anchors), n=B, prefix="vl",)

                gnn_stats, gnn_counts = compute_gnn_stats(out["gnn_out"], labels=lab)
                mon_stage3.update(gnn_stats, n=B, prefix="vl", specific_n=gnn_counts)

                scale_stats = compute_scale_stats(model)    
                mon_stage3.update(scale_stats, n=1, prefix="vl", specific_n={k: 1 for k in scale_stats})

                mon_stage3.update(compute_s2_s3_gap(out["logits_s2"], out["logits_s3"], lab), n=B, prefix="vl")
                mon_stage3.update(compute_update_dynamics(out["u"], out["h"], lab, model.class_anchors), n=B, prefix="vl")
                mon_stage3.update(compute_prior_consistency(out["gnn_out"]["raw"]["pi_refined"][-1], lab), n=B, prefix="vl")
                mon_stage3.update(compute_structure_diversity(out["v"]), n=B, prefix="vl")

                tf_stats = compute_tf_stats(out["gnn_out"], out["logits_s2"], lab)
                mon_stage3.update(tf_stats, n=B, prefix="vl")

                gnn_raw = out["gnn_out"].get("raw", {})
                if "q" in gnn_raw and "qbar" in gnn_raw:
                    oracle_stats = compute_oracle_feasibility(q_ij=gnn_raw["q"][-1], q_bar=gnn_raw["qbar"][-1], labels=lab, alpha_agg=gnn_raw["alpha"][-1],    
                                                              alpha_target=alpha_target,  num_classes=model.C,)
                    mon_stage3.update(oracle_stats, n=B, prefix="vl")

                if "qbar" in gnn_raw:
                    vl_qbar_buf.append(gnn_raw["qbar"][-1].cpu())

                if vl_batch_idx == 0 and epoch % plot_every == 0:
                    gnn_raw = out["gnn_out"].get("raw", {})
                    if "q" in gnn_raw and "alpha" in gnn_raw:
                        plot_attention_sample(q_ij=gnn_raw["q"][-1][0], alpha_agg=gnn_raw["alpha"][-1][0], label=lab[0], run_dir=run_dir, epoch=epoch, alpha_target=alpha_target[0])

                vl_preds_pre.append(out["logits_s2"].cpu())
                vl_preds.append(out["logits_s3"].cpu())
                vl_labs.append(lab.cpu())
                vl_z_buf.append(out["z"].cpu())
                vl_u_buf.append(out["u"].cpu())
                vl_v_buf.append(out["v"].cpu())
                vl_h_buf.append(out["h"].cpu())

        scheduler.step()

        vl_preds_pre_cat = torch.cat(vl_preds_pre)
        vl_preds_cat, vl_labs_cat = torch.cat(vl_preds), torch.cat(vl_labs)
        tr_preds_cat, tr_labs_cat = torch.cat(tr_preds), torch.cat(tr_labs)

        anchor_stats = compute_anchor_metrics(torch.cat(vl_u_buf), vl_labs_cat,
                                              num_regions=model.R, anchors=model.class_anchors)
        mon_stage2.update(anchor_stats, n=1, prefix="vl")

        move_stats, move_counts = compute_move_stats(vl_labs_cat, vl_preds_pre_cat, vl_preds_cat, num_classes=model.C,)
        mon_stage3.update(move_stats, n=1, prefix="vl_move", specific_n=move_counts)         

        tr_bsn = bsn_metrics(tr_preds_cat, tr_labs_cat)
        mon_stage4.update(tr_bsn, n=1, prefix="tr")
        vl_bsn = bsn_metrics(vl_preds_cat, vl_labs_cat)
        mon_stage4.update(vl_bsn, n=1, prefix="vl")

        print(f"\n[Epoch {epoch:03d}] | "
              f"Train ACC: {tr_bsn['avg_Acc']:.4f}, Valid ACC: {vl_bsn['avg_Acc']:.4f} | "
              f"Train MAE: {tr_bsn['avg_MAE']:.4f}, Valid MAE: {vl_bsn['avg_MAE']:.4f} | "
              f"Train Oracle Ent:{tr_oracle_ent:.4f}, Valid Oracle Ent:{vl_oracle_ent:.4f} "
              )

        current_mae = vl_bsn["avg_MAE"]
        for mon in [mon_loss, mon_stage1, mon_stage2, mon_stage3, mon_stage4]:
            mon.save_and_plot(epoch)

        if epoch % plot_every == 0:
            tsne_dict = {
                "Stage1 (Z)": torch.cat(vl_z_buf),
                "Stage2 (u)": torch.cat(vl_u_buf),
                "Stage2 (v)": torch.cat(vl_v_buf),
                "Stage3 (h)": torch.cat(vl_h_buf),
            }
            plot_tsne_stages(tsne_dict, vl_labs_cat, run_dir / "tsne", epoch,
                             anchors=model.class_anchors)

            if vl_qbar_buf:
                vl_qbar_cat = torch.cat(vl_qbar_buf)
                plot_qbar_roi(vl_qbar_cat, vl_labs_cat, run_dir, epoch, num_classes=model.C)

        if current_mae < best_val_avg_mae:
            best_val_avg_mae = current_mae
            checkpoint = {
                "state_dict": model.state_dict(),
                "config": {
                    "num_regions": model.R,
                    "num_classes": model.C,
                    "num_patterns": model.K,
                    "proj_dim": model.proj_dim,
                    "gnn_num_heads": model.gnn.num_heads,
                    "gnn_dropout": model.gnn.att_dropout.p,
                },
                "epoch": epoch,
                "best_mae": best_val_avg_mae,
                "roi_class_counts": roi_class.cpu(),
            }
            torch.save(checkpoint, run_dir / "best_model.pth")
            print(f" >> Best Model Saved (MAE: {best_val_avg_mae:.4f})")


def main():
    import argparse
    import yaml

    from dorga.data.dataset import _make_loader, generate_severity_patterns, compute_priors_dynamic
    from dorga.losses.losses import compute_class_counts_from_df

    parser = argparse.ArgumentParser(description="DORGA Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    d = cfg["data"]
    m = cfg["model"]
    t = cfg["training"]

    meta = pd.read_csv(d["meta_csv"])
    N_PATTERNS = m["num_patterns"]
    meta = generate_severity_patterns(meta, n_patterns=N_PATTERNS)
    print(meta.head(5))

    train_df = meta[meta.split.isin(d["train_splits"])]
    test_df = meta[meta.split == d["test_split"]]

    _, roi_class_counts = compute_class_counts_from_df(train_df)
    pi_global, pi_patterns = compute_priors_dynamic(train_df, n_patterns=N_PATTERNS)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | patterns: {N_PATTERNS}")

    vit = load_mae_ckpt_to_512(
        m["backbone"]["checkpoint"],
        num_classes=m["backbone"]["num_classes"],
        in_chans=m["backbone"]["in_chans"],
        drop_path_rate=m["backbone"]["drop_path_rate"],
        verbose=False,
    )

    model = BrixiaViT512Dynamic(
        vit,
        num_regions=m["num_regions"],
        num_classes=m["num_classes"],
        num_patterns=N_PATTERNS,
        proj_dim=m["proj_dim"],
        pi_global=pi_global,
        pi_patterns=pi_patterns,
        gnn_num_heads=m["gnn_num_heads"],
        gnn_dropout=m["gnn_dropout"],
    ).to(device)

    run_dir = Path(cfg["output"]["run_dir"]) / f"{N_PATTERNS}_{m['proj_dim']}" / f"{datetime.datetime.now():%m%d_%H%M}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_dir}")

    bs = d["batch_size"]
    norm_dir = Path(d["normalized_dir"])
    img_dir = str(norm_dir / "images")
    mask_dir = str(norm_dir / "masks")
    label_cols = d.get("label_cols", [f"brixia{i}" for i in range(1, 7)])

    loaders = {
        "train": _make_loader(train_df, img_dir=img_dir, mask_dir=mask_dir,
                              mode="train", bs=bs, shuffle=True, drop_last=True,
                              label_cols=label_cols, include_pattern=True),
        "test": _make_loader(test_df, img_dir=img_dir, mask_dir=mask_dir,
                             mode="eval", bs=bs, shuffle=False, drop_last=False,
                             label_cols=label_cols, include_pattern=True),
    }

    train_dynamic(
        model=model,
        loaders=loaders,
        run_dir=run_dir,
        epochs=t["epochs"],
        roi_class=roi_class_counts,
        loss_func_stage3=t["loss_func_stage3"],
        plot_every=t.get("plot_every", 1),
    )


if __name__ == "__main__":
    main()
