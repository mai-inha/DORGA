import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List
from collections import defaultdict

ROI_NAMES = ["1","2","3","4","5","6"]

class StageMonitor:
    def __init__(self, run_dir: Union[str, Path], stage_name: str, plot_func=None):
        self.run_dir = Path(run_dir)
        self.stage_name = stage_name
        self.plot_func = plot_func
        self.csv_path = self.run_dir / f"{stage_name}_metrics.csv"
        self.reset()

    def reset(self):
        self.metrics = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, data: Dict[str, Any], n: int = 1, prefix: str = "tr", specific_n: Optional[Dict[str, int]] = None):
        if not data: return
        
        for k, v in data.items():
            if v is None: continue
            key = f"{prefix}_{k}"
            
            if hasattr(v, 'item'): val = v.item()
            elif isinstance(v, (int, float, np.number)): val = float(v)
            else: continue
            
            if np.isfinite(val):
                count = n
                if specific_n is not None and k in specific_n:
                    count = specific_n[k]
                
                if count > 0:
                    self.metrics[key] += val * count
                    self.counts[key] += count

    def save_and_plot(self, epoch: int):
        avg_metrics = {"epoch": int(epoch)}
        for k, v in self.metrics.items():
            if self.counts[k] > 0:
                avg_metrics[k] = round(v / self.counts[k], 6)
        
        df_row = pd.DataFrame([avg_metrics])
        if self.csv_path.exists():
            df_row.to_csv(self.csv_path, mode='a', header=False, index=False)
        else:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_row.to_csv(self.csv_path, mode='w', header=True, index=False)
        
        if self.plot_func:
            try:
                self.plot_func(self.run_dir)
            except Exception as e:
                print(f"[{self.stage_name} Plot Error] {e}")
        self.reset()


def compute_pi_stats(w: torch.Tensor, entropy: torch.Tensor, eps: float = 1e-8) -> Dict[str, float]:
    stats = {}
    
    stats["pi_entropy"] = entropy.mean().item()
    
    w_sum = w.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = w / w_sum
    neff = (-(p * (p.clamp_min(eps)).log()).sum(dim=-1)).exp()
    stats["pi_neff"] = neff.mean().item()
    
    w_max = w.max(dim=-1).values
    stats["w_max"] = w_max.mean().item()
    
    B, R, P = w.shape
    roi_names = ["ROI1", "ROI2", "ROI3", "ROI4", "ROI5", "ROI6"]
    
    neff_roi_mean = neff.mean(dim=0)
    w_max_roi_mean = w_max.mean(dim=0)
    
    for r in range(min(R, 6)):
        key_neff = f"pi_neff_{roi_names[r]}"
        key_wmax = f"w_max_{roi_names[r]}"
        stats[key_neff] = neff_roi_mean[r].item()
        stats[key_wmax] = w_max_roi_mean[r].item()
        
    return stats

def compute_pattern_stats(gnn_out: Dict) -> Tuple[Dict[str, float], Dict[str, int]]:
    stats = {}
    specific_n = {}

    if "raw" not in gnn_out:
        return stats, specific_n

    if "gate" in gnn_out["raw"]:
        g = gnn_out["raw"]["gate"][-1].squeeze(-1).float()

        stats["gate_mean_l0"] = g.mean().item()
        stats["gate_std_l0"] = g.std().item()
        stats["gate_min_l0"] = g.min().item()
        stats["gate_max_l0"] = g.max().item()
        stats["gate_open_rate_l0"] = (g > 0.3).float().mean().item()
        stats["gate_open_strict_l0"] = (g > 0.7).float().mean().item()
        stats["gate_closed_rate_l0"] = (g < 0.1).float().mean().item()

        stats["gnn_gamma_mean_l0"] = stats["gate_mean_l0"]
        stats["gnn_gamma_std_l0"] = stats["gate_std_l0"]
        stats["gnn_gamma_min_l0"] = stats["gate_min_l0"]
        stats["gnn_gamma_max_l0"] = stats["gate_max_l0"]
        stats["gnn_gamma_low_rate_l0"] = (g < 0.3).float().mean().item()

    if "qbar" in gnn_out["raw"]:
        qbar = gnn_out["raw"]["qbar"][-1]
        C = qbar.shape[-1]
        class_idx = torch.arange(C, device=qbar.device, dtype=qbar.dtype)
        V_max = ((C - 1) ** 2) / 4.0

        mu = (qbar * class_idx).sum(-1)
        var = (qbar * (class_idx - mu.unsqueeze(-1)).pow(2)).sum(-1)
        entropy = -(qbar * qbar.clamp_min(1e-8).log()).sum(-1)

        stats["qbar_mu_mean"] = mu.mean().item()
        stats["qbar_mu_std"] = mu.std().item()
        stats["qbar_var_mean"] = var.mean().item()
        stats["qbar_var_max"] = var.max().item()
        stats["qbar_ord_health"] = (1.0 - var / V_max).clamp(0, 1).mean().item()
        stats["qbar_bimodal_rate"] = (var > V_max * 0.7).float().mean().item()
        stats["qbar_entropy_mean"] = entropy.mean().item()

        mode_strength = qbar.max(-1).values - (1.0 / C)
        stats["qbar_mode_strength"] = mode_strength.clamp(0).mean().item()

    return stats, specific_n

def compute_dual_proj_stats(u, v, labels, anchors):
    u_unit = F.normalize(u.detach(), dim=-1)
    v_unit = F.normalize(v.detach(), dim=-1)
    a_unit = F.normalize(anchors.detach(), dim=-1)
    
    C = a_unit.shape[0]
    w = F.normalize(a_unit[C-1] - a_unit[0], dim=-1)
    
    u_w_cos = (u_unit @ w).abs()
    v_w_cos = (v_unit @ w).abs()
    uv_cos = (u_unit * v_unit).sum(dim=-1).abs()
    
    B, R, _ = v.shape
    v_sim = torch.bmm(v_unit, v_unit.transpose(1, 2)) 
    mask = ~torch.eye(R, device=v.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    v_div = v_sim[mask].abs().mean()
    
    a_target = a_unit[labels]
    u_anchor_align = (u_unit * a_target).sum(dim=-1)
    
    return {
        "u_severity_align": u_w_cos.mean().item(),
        "u_severity_align_min": u_w_cos.min().item(),
        "u_severity_align_max": u_w_cos.max().item(),
        "v_severity_orth": v_w_cos.mean().item(),
        "v_severity_orth_min": v_w_cos.min().item(),
        "v_severity_orth_max": v_w_cos.max().item(),
        "uv_decorr": uv_cos.mean().item(),
        "uv_decorr_min": uv_cos.min().item(),
        "uv_decorr_max": uv_cos.max().item(),
        "v_diversity": v_div.item(),
        "u_anchor_align": u_anchor_align.mean().item(),
    }

def _compute_anchor_penalties(anchors, margin=0.3):
    a = F.normalize(anchors.detach(), p=2, dim=-1)
    C = a.shape[0]
    
    sim = a @ a.T
    eye = torch.eye(C, device=a.device, dtype=torch.bool)
    max_off_diag = sim[~eye].max()
    sep_dist = F.relu(max_off_diag - (1.0 - margin)).item()
    
    ord_loss = 0.0
    for i in range(C-2):
        violation = sim[0, i+2] - sim[0, i+1] + 0.05
        ord_loss += F.relu(violation)
        
    return {"sep_dist": sep_dist, "ord": ord_loss.item()}

def compute_anchor_metrics(z, labels, num_regions=6, anchors=None):
    z = F.normalize(z.detach().float(), p=2, dim=-1)
    labels = labels.detach().long()
    
    if z.dim() == 3: 
        B, R, D = z.shape
        z = z.view(-1, D); labels = labels.view(-1)
        roi_ids = torch.arange(R, device=z.device).repeat(B)
    else:
        roi_ids = torch.zeros_like(labels) 

    N = z.shape[0]
    if N > 2000:
        idx = torch.randperm(N)[:2000]
        z = z[idx]; labels = labels[idx]; roi_ids = roi_ids[idx]
        N = 2000

    sim = z @ z.T
    dist = 1 - sim
    
    mask_same = (labels.unsqueeze(1) == labels.unsqueeze(0))
    mask_diff = ~mask_same
    d_intra = dist[mask_same].mean().item() if mask_same.any() else 0.0
    d_inter = dist[mask_diff].mean().item() if mask_diff.any() else 0.0
    
    k = min(10, N - 1)
    if k > 0 and num_regions > 1:
        dist_fill = dist.clone()
        dist_fill.diagonal().fill_(float('inf'))
        _, idx = dist_fill.topk(k, dim=1, largest=False)
        
        neighbor_rois = roi_ids[idx]
        
        n_unique = torch.zeros(N, device=z.device)
        for i in range(N):
            n_unique[i] = neighbor_rois[i].unique().numel()
        
        roi_mixing = (n_unique.mean() / num_regions).item()
    else:
        roi_mixing = 0.0

    center_vars = []
    num_classes = int(labels.max().item()) + 1
    for c in range(num_classes):
        c_mask = (labels == c)
        if not c_mask.any(): continue
        
        z_c = z[c_mask]
        r_c = roi_ids[c_mask]
        
        roi_centers = []
        for r in range(num_regions):
            r_mask = (r_c == r)
            if r_mask.any():
                roi_centers.append(z_c[r_mask].mean(dim=0))
        
        if len(roi_centers) > 1:
            roi_centers = torch.stack(roi_centers)
            center_mean = roi_centers.mean(dim=0)
            var = ((roi_centers - center_mean)**2).sum().item()
            center_vars.append(var)
            
    center_var = np.mean(center_vars) if center_vars else 0.0

    stats = {
        "d_intra": d_intra, "d_inter": d_inter,
        "separability": d_inter / (d_intra + 1e-8),
        "roi_mixing": roi_mixing, 
        "center_var": center_var,
        "sep_dist": 0.0, "ord": 0.0
    }
    
    if anchors is not None:
        pen = _compute_anchor_penalties(anchors)
        stats.update(pen)
        
    return stats


def compute_gnn_stats(gnn_out: Dict, labels: Optional[torch.Tensor] = None
                      ) -> Tuple[Dict[str, float], Dict[str, int]]:
    stats = {}
    counts = {}

    if "raw" not in gnn_out or "alpha" not in gnn_out["raw"]:
        return stats, counts

    alpha = gnn_out["raw"]["alpha"][-1].detach()
    B, R, _ = alpha.shape

    stats["alpha_entropy_mean"] = -(alpha * (alpha + 1e-8).log()).sum(-1).mean().item()
    stats["alpha_top1_mass_mean"] = alpha.max(-1).values.mean().item()
    stats["alpha_collapse_rate"] = (alpha.max(-1).values > 0.9).float().mean().item()

    eye = torch.eye(R, device=alpha.device).unsqueeze(0).bool()
    off = alpha.masked_select(~eye.expand(B, -1, -1))
    stats["alpha_mean"] = off.mean().item()
    stats["alpha_std"] = off.std().item()
    stats["alpha_self_mean"] = alpha.diagonal(dim1=-2, dim2=-1).mean().item()

    if "gate" in gnn_out["raw"]:
        g = gnn_out["raw"]["gate"][-1].squeeze(-1)
        stats["gate_mean"] = g.mean().item()
        stats["gate_std"] = g.std().item()
        stats["gate_min"] = g.min().item()
        stats["gate_max"] = g.max().item()
        stats["gate_open_rate"] = (g > 0.3).float().mean().item()
        stats["gate_open_rate_strict"] = (g > 0.7).float().mean().item()
        stats["gate_closed_rate"] = (g < 0.1).float().mean().item()
        stats["gating_effectiveness"] = stats["gate_open_rate_strict"]

    if "t_eff" in gnn_out["raw"]:
        t = gnn_out["raw"]["t_eff"][-1]
        stats["t_eff_mean"] = t.mean().item()
        stats["t_eff_max"] = t.max().item()
        stats["t_eff_nonzero_rate"] = (t > 1e-4).float().mean().item()

    if "qbar" in gnn_out["raw"]:
        qbar = gnn_out["raw"]["qbar"][-1]
        C = qbar.shape[-1]
        class_idx = torch.arange(C, device=qbar.device, dtype=qbar.dtype)
        V_max = ((C - 1) ** 2) / 4.0

        mu = (qbar * class_idx).sum(-1)
        var = (qbar * (class_idx - mu.unsqueeze(-1)).pow(2)).sum(-1)
        ent = -(qbar * qbar.clamp_min(1e-8).log()).sum(-1)

        stats["qbar_mu_mean"] = mu.mean().item()
        stats["qbar_mu_std"] = mu.std().item()
        stats["qbar_var_mean"] = var.mean().item()
        stats["qbar_var_max"] = var.max().item()
        stats["qbar_ord_health"] = (1.0 - var / V_max).clamp(0, 1).mean().item()
        stats["qbar_bimodal_rate"] = (var > V_max * 0.7).float().mean().item()
        stats["qbar_entropy_mean"] = ent.mean().item()

        mode_strength = qbar.max(-1).values - (1.0 / C)
        stats["qbar_mode_strength"] = mode_strength.clamp(0).mean().item()

        if labels is not None:
            qbar_pred = qbar.argmax(-1)
            stats["qbar_acc"] = (qbar_pred == labels).float().mean().item()
            qbar_mae = (mu - labels.float()).abs().mean().item()
            stats["qbar_mae"] = qbar_mae

            for c in range(C):
                mask_c = (labels == c)
                if mask_c.any():
                    stats[f"qbar_acc_c{c}"] = (qbar_pred[mask_c] == c).float().mean().item()
                    stats[f"qbar_var_c{c}"] = var[mask_c].mean().item()

    if "bias" in gnn_out["raw"]:
        stats["bias_abs_mean"] = gnn_out["raw"]["bias"][-1].abs().mean().item()
    if "bias_strength" in gnn_out["raw"]:
        stats["bias_strength"] = gnn_out["raw"]["bias_strength"][-1]

    if "alpha_per_head" in gnn_out["raw"]:
        ah = gnn_out["raw"]["alpha_per_head"][-1]
        H = ah.shape[1]
        for h in range(H):
            a_h = ah[:, h]
            stats[f"head{h}_entropy"] = -(a_h * (a_h + 1e-8).log()).sum(-1).mean().item()
            stats[f"head{h}_self"] = a_h.diagonal(dim1=-2, dim2=-1).mean().item()
        kl_s, n = 0.0, 0
        for i in range(H):
            for j in range(i + 1, H):
                kl_s += (ah[:, i].clamp_min(1e-8) * (
                    ah[:, i].clamp_min(1e-8).log() - ah[:, j].clamp_min(1e-8).log()
                )).sum(-1).mean().item()
                n += 1
        stats["head_diversity_kl"] = kl_s / max(n, 1)

    if "msg" in gnn_out["raw"]:
        stats["msg_norm_mean"] = gnn_out["raw"]["msg"][-1].norm(dim=-1).mean().item()

    if labels is not None:
        y = labels.long()
        for c in range(4):
            for rs in range(min(R, 6)):
                m = (y[:, rs] == c)
                cnt = m.sum().item()
                if cnt == 0:
                    continue
                avg = alpha[m, rs, :].mean(0)
                for rt in range(min(R, 6)):
                    k = f"att_map_sum_c{c}_src{rs}_tgt{rt}"
                    stats[k] = avg[rt].item()
                    counts[k] = cnt

    return stats, counts


def compute_oracle_feasibility(q_ij, q_bar, labels, alpha_agg=None, alpha_target=None, num_classes=4):
    device = q_ij.device
    B, R, _, C = q_ij.shape
    class_idx = torch.arange(C, device=device).float()
    y = labels.float()

    self_mask = torch.eye(R, device=device, dtype=torch.bool).unsqueeze(0)

    if alpha_target is not None:
        q_bar_oracle = torch.einsum("bij,bijc->bic", alpha_target, q_ij)
        E_oracle = (q_bar_oracle * class_idx).sum(dim=-1)
        oracle_pred = q_bar_oracle.argmax(dim=-1)
        oracle_acc = (oracle_pred == labels).float()
        oracle_err = (E_oracle - y).abs()

        alpha_target_ns = alpha_target.masked_fill(self_mask, 0.0)
        alpha_target_ns = alpha_target_ns / alpha_target_ns.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        q_bar_oracle_ns = torch.einsum("bij,bijc->bic", alpha_target_ns, q_ij)
        E_oracle_ns = (q_bar_oracle_ns * class_idx).sum(dim=-1)
        oracle_pred_ns = q_bar_oracle_ns.argmax(dim=-1)
        oracle_acc_ns = (oracle_pred_ns == labels).float()
        oracle_err_ns = (E_oracle_ns - y).abs()
    else:
        oracle_acc = torch.zeros(B, R, device=device)
        oracle_err = torch.ones(B, R, device=device)
        oracle_acc_ns = oracle_acc
        oracle_err_ns = oracle_err

    E_agg = (q_bar * class_idx).sum(dim=-1)
    agg_pred = q_bar.argmax(dim=-1)
    agg_acc = (agg_pred == labels).float()
    agg_err = (E_agg - y).abs()

    if alpha_agg is not None:
        alpha_ns = alpha_agg.masked_fill(self_mask, 0.0)
        alpha_ns = alpha_ns / alpha_ns.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        q_bar_ns = (alpha_ns.unsqueeze(-1) * q_ij).sum(dim=2)
    else:
        alpha_ns = (~self_mask).float() / (R - 1)
        q_bar_ns = (alpha_ns.unsqueeze(-1) * q_ij).sum(dim=2)

    E_agg_ns = (q_bar_ns * class_idx).sum(dim=-1)
    agg_pred_ns = q_bar_ns.argmax(dim=-1)
    agg_acc_ns = (agg_pred_ns == labels).float()
    agg_err_ns = (E_agg_ns - y).abs()

    stats = {}
    avgs = {k: [] for k in [
        "oracle_acc", "oracle_mae", "oracle_ns_acc", "oracle_ns_mae",
        "agg_acc",    "agg_mae",    "agg_ns_acc",    "agg_ns_mae"
    ]}

    for r in range(min(R, 6)):
        name = f"ROI{r+1}"
        vals = {
            "oracle_acc": oracle_acc[:, r].mean().item(),
            "oracle_mae": oracle_err[:, r].mean().item(),
            "oracle_ns_acc": oracle_acc_ns[:, r].mean().item(),
            "oracle_ns_mae": oracle_err_ns[:, r].mean().item(),
            "agg_acc": agg_acc[:, r].mean().item(),
            "agg_mae": agg_err[:, r].mean().item(),
            "agg_ns_acc": agg_acc_ns[:, r].mean().item(),
            "agg_ns_mae": agg_err_ns[:, r].mean().item(),
        }
        for k, v in vals.items():
            stats[f"{k}_{name}"] = v
            avgs[k].append(v)

    for k, v_list in avgs.items():
        stats[k] = sum(v_list) / len(v_list)

    return stats


def compute_update_dynamics(u, h, labels, anchors) -> Dict[str, float]:
    with torch.no_grad():
        a = F.normalize(anchors.detach(), dim=-1)
        u_n = F.normalize(u.detach(), dim=-1)
        h_n = F.normalize(h.detach(), dim=-1)
        t_a = a[labels.long()]

        d_pre = 1.0 - (u_n * t_a).sum(-1)
        d_post = 1.0 - (h_n * t_a).sum(-1)
        imp = d_pre - d_post
        cos_uh = (u_n * h_n).sum(-1).clamp(-1, 1)

    return {
        "dyn_TUR": (imp < -1e-5).float().mean().item(),
        "dyn_BFR": (imp > 1e-5).float().mean().item(),
        "dyn_NC": imp.mean().item(),
        "dyn_angular_move": torch.acos(cos_uh).mean().item(),
    }


def compute_prior_consistency(pi, labels) -> Dict[str, float]:
    B, R, _, C, _ = pi.shape
    d = pi.device
    with torch.no_grad():
        bi = torch.arange(B, device=d).view(B, 1, 1)
        rs = torch.arange(R, device=d).view(1, R, 1)
        rt = torch.arange(R, device=d).view(1, 1, R)
        ls = labels.unsqueeze(2).expand(B, R, R)
        lt = labels.unsqueeze(1).expand(B, R, R)
        prob = pi[bi, rs, rt, ls, lt]
        m = ~torch.eye(R, device=d, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    return {"prior_PLC": prob[m].mean().item()}


def compute_structure_diversity(v) -> Dict[str, float]:
    B, R, _ = v.shape
    with torch.no_grad():
        sim = torch.bmm(v, v.transpose(1, 2))
        mask = ~torch.eye(R, device=v.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    return {"struct_AvgSim": sim[mask].mean().item(), "struct_MaxSim": sim[mask].max().item()}


def compute_scale_stats(model) -> Dict[str, float]:
    stats = {}
    head = getattr(model, 'cls_head', None)
    if head is None: return stats
    if hasattr(head, 'logit_scale_s2'):
        stats["scale_s2"] = head.logit_scale_s2.exp().clamp(1, 20).item()
        stats["scale_s3"] = head.logit_scale_s3.exp().clamp(1, 20).item()
        stats["scale_gap"] = abs(stats["scale_s3"] - stats["scale_s2"])
    elif hasattr(head, 'logit_scale'):
        stats["scale_shared"] = head.logit_scale.exp().clamp(1, 15).item()
    return stats


def compute_s2_s3_gap(logits_s2, logits_s3, labels) -> Dict[str, float]:
    stats = {}
    with torch.no_grad():
        p2 = logits_s2.detach().argmax(-1)
        p3 = logits_s3.detach().argmax(-1)
        y = labels.detach()
        n = y.numel()

        stats["s2s3_pred_agreement"] = (p2 == p3).float().mean().item()
        imp = ((p2 != y) & (p3 == y)).float().sum().item() / n
        deg = ((p2 == y) & (p3 != y)).float().sum().item() / n
        stats["s3_improvement"] = imp
        stats["s3_degradation"] = deg
        stats["s3_net_gain"] = imp - deg
        stats["s2_acc"] = (p2 == y).float().mean().item()
        stats["s3_acc"] = (p3 == y).float().mean().item()

        C = logits_s2.shape[-1]
        ci = torch.arange(C, device=logits_s2.device).float()
        yf = y.float()
        m2 = ((F.softmax(logits_s2.float(), -1) * ci).sum(-1) - yf).abs().mean().item()
        m3 = ((F.softmax(logits_s3.float(), -1) * ci).sum(-1) - yf).abs().mean().item()
        stats["s2_mae"] = m2
        stats["s3_mae"] = m3
        stats["s3_mae_delta"] = m2 - m3
    return stats


def compute_move_stats(
    labels: torch.Tensor,
    logits_pre: torch.Tensor,
    logits_post: torch.Tensor,
    num_classes: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    stats: Dict[str, float] = {}
    custom_counts: Dict[str, int] = {}

    if labels is None or logits_pre is None or logits_post is None:
        return stats, custom_counts

    if logits_pre.dim() == 2:
        logits_pre = logits_pre.unsqueeze(1)
    if logits_post.dim() == 2:
        logits_post = logits_post.unsqueeze(1)

    if logits_pre.dim() != 3 or logits_post.dim() != 3:
        return stats, custom_counts
    if logits_pre.shape[:2] != logits_post.shape[:2]:
        return stats, custom_counts
    if logits_pre.shape[2] != logits_post.shape[2]:
        return stats, custom_counts

    B, R, C_logits = logits_pre.shape
    C = int(C_logits) if num_classes is None else int(num_classes)
    C = min(C, C_logits)

    pre = logits_pre.detach().argmax(dim=-1)
    post = logits_post.detach().argmax(dim=-1)

    labels = labels.detach()
    if labels.dim() == 1:
        labels_br = labels.view(B, 1).expand(B, R)
    elif labels.dim() == 2:
        if labels.shape[0] != B:
            return stats, custom_counts
        if labels.shape[1] == 1:
            labels_br = labels.expand(B, R)
        elif labels.shape[1] == R:
            labels_br = labels
        else:
            return stats, custom_counts
    else:
        return stats, custom_counts

    labels_br = labels_br.long()

    for r in range(R):
        pre_r = pre[:, r]
        post_r = post[:, r]
        gt_r = labels_br[:, r]

        for g in range(C):
            mask_g = (gt_r == g)
            n_g = int(mask_g.sum().item())

            k_ng = f"gtcount_r{r}_g{g}"
            stats[k_ng] = float(n_g)
            custom_counts[k_ng] = 1

            for x in range(C):
                mask_gx = mask_g & (pre_r == x)
                denom = int(mask_gx.sum().item())

                k_denom = f"precount_r{r}_g{g}_x{x}"
                stats[k_denom] = float(denom)
                custom_counts[k_denom] = 1

                if denom == 0:
                    for y in range(C):
                        k_ratio = f"ratio_r{r}_g{g}_x{x}_y{y}"
                        stats[k_ratio] = 0.0
                        custom_counts[k_ratio] = 1
                    continue

                for y in range(C):
                    cnt = int((mask_gx & (post_r == y)).sum().item())
                    ratio = float(cnt) / float(denom)

                    k_ratio = f"ratio_r{r}_g{g}_x{x}_y{y}"
                    stats[k_ratio] = ratio
                    custom_counts[k_ratio] = denom

    return stats, custom_counts

def compute_tf_stats(gnn_out, logits_s2, labels):
    stats = {}
    gnn_raw = gnn_out.get("raw", {})

    if "gt_ratio" in gnn_raw:
        gt_r = gnn_raw["gt_ratio"][-1]
        stats["tf_ratio"] = gt_r if isinstance(gt_r, (int, float)) else float(gt_r)

    with torch.no_grad():
        p_pred = logits_s2.detach().argmax(-1)
        stats["tf_p_argmax_acc"] = (p_pred == labels).float().mean().item()

    if "qbar" in gnn_raw:
        with torch.no_grad():
            qbar_pred = gnn_raw["qbar"][-1].argmax(-1)
            stats["tf_qbar_model_acc"] = (qbar_pred == labels).float().mean().item()

    return stats

def _compute_metrics_from_preds(pred: torch.Tensor, gt: torch.Tensor):
    pred = pred.float()
    gt = gt.float()
    err = pred - gt
    abs_err = err.abs()
    
    mae = abs_err.mean().item()
    acc = (pred == gt).float().mean().item()
    mer = err.mean().item()
    sd = err.std().item() if err.numel() > 1 else 0.0
    
    if pred.std() == 0 or gt.std() == 0:
        cc = 0.0
    else:
        vx = pred - pred.mean()
        vy = gt - gt.mean()
        cc = (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8)
        cc = cc.item()
        
    return {"Acc": acc, "MAE": mae, "MEr": mer, "SD": sd, "CC": cc}

def bsn_metrics(preds_or_logits: torch.Tensor, gt: torch.Tensor, is_logits: bool = True) -> Dict[str, float]:
    if is_logits: pred = preds_or_logits.argmax(dim=-1)
    else: pred = preds_or_logits.long()
    gt = gt.long()
    
    B, R = pred.shape
    results = {}
    
    pred_g = pred.sum(dim=1)
    gt_g = gt.sum(dim=1)
    global_stats = _compute_metrics_from_preds(pred_g, gt_g)
    for k, v in global_stats.items(): results[f"global_{k}"] = v
        
    roi_names = ["ROI1", "ROI2", "ROI3", "ROI4", "ROI5", "ROI6"]
    roi_metrics_list = defaultdict(list)
    
    for r in range(min(R, 6)):
        p_r = pred[:, r]; g_r = gt[:, r]
        r_stats = _compute_metrics_from_preds(p_r, g_r)
        for k, v in r_stats.items():
            roi_metrics_list[k].append(v)
            results[f"{roi_names[r]}_{k}"] = v
            
    for k, v_list in roi_metrics_list.items():
        results[f"avg_{k}"] = sum(v_list) / len(v_list)
        
    return results

def balanced_acc(preds, gt):
    return {}


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.float().view(-1)
    y = y.float().view(-1)
    x = x - x.mean()
    y = y - y.mean()
    vx = (x * x).sum()
    vy = (y * y).sum()
    denom = torch.sqrt(vx * vy).clamp_min(eps)
    return (x * y).sum() / denom

def bsn_metrics_from_preds(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    pred, gt = pred.long(), gt.long()
    N, R = pred.shape
    REGION_NAMES = ["A", "B", "C", "D", "E", "F"][:R]
    result = {}
    region_accs, region_ccs = [], []
    for r, name in enumerate(REGION_NAMES):
        p_r, g_r = pred[:, r], gt[:, r]
        err_r = (p_r - g_r).float()
        abs_err_r = err_r.abs()
        acc_r = (p_r == g_r).float().mean()
        region_accs.append(acc_r)
        region_ccs.append(_pearson_corr(p_r.float(), g_r.float()))
        result[name] = {
            "Acc": acc_r, "MEr": err_r.mean(),
            "MAE": abs_err_r.mean(), "SD": abs_err_r.std(unbiased=False),
            "CC": region_ccs[-1],
        }
    err = (pred - gt).float()
    abs_err = err.abs()
    result["avg"] = {
        "Acc": torch.stack(region_accs).mean(), "MEr": err.mean(),
        "MAE": abs_err.mean(), "SD": abs_err.std(unbiased=False),
        "CC": torch.stack(region_ccs).mean(),
    }
    pred_g, gt_g = pred.sum(dim=1).float(), gt.sum(dim=1).float()
    err_g = pred_g - gt_g
    abs_err_g = err_g.abs()
    exact_match = (pred == gt).all(dim=1).float().mean()
    result["global"] = {
        "Acc": exact_match, "MEr": err_g.mean(),
        "MAE": abs_err_g.mean(), "SD": abs_err_g.std(unbiased=False),
        "CC": _pearson_corr(pred_g, gt_g),
    }
    return result
