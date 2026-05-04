import math
from typing import Literal, Optional, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
def loss_function(  
    logits: torch.Tensor,
    labels: torch.Tensor,
    roi_class: Optional[torch.Tensor] = None,
    option: Literal["CE", "Class", "RoiClass"] = "RoiClass",
) -> torch.Tensor:
    B, R, C = logits.shape

    if option == "CE":
        logits_adj = logits

    else:
        if roi_class is None:
            raise ValueError("roi_class is required when option is 'Class' or 'RoiClass'.")

        prior = roi_class.to(device=logits.device, dtype=logits.dtype).clamp(min=1e-6)

        if option == "Class":
            cnt = roi_class.to(device=logits.device, dtype=logits.dtype).clamp(min=1e-6)
            class_cnt = cnt.sum(dim=0)
            w_class = (class_cnt.sum() / class_cnt)
            w_class = w_class / w_class.mean()
            return F.cross_entropy(logits.reshape(-1, C), labels.reshape(-1), weight=w_class)
        
        elif option == "RoiClass":
            cnt = roi_class.to(device=logits.device, dtype=logits.dtype).clamp(min=1e-6)

            w_rc = (cnt.sum(dim=1, keepdim=True) / cnt)
            w_rc = w_rc / w_rc.mean()

            loss_per = F.cross_entropy(
                logits.reshape(-1, C),
                labels.reshape(-1),
                reduction="none"
            ).view(B, R)

            y = labels.long()

            r_idx = torch.arange(R, device=logits.device).view(1, R).expand(B, R)

            w_pos = w_rc[r_idx, y.clamp(min=0)]

            return (loss_per * w_pos).sum() / (w_pos.sum().clamp_min(1e-6))


        else:
            raise ValueError(f"Unknown option: {option}")

    return F.cross_entropy(logits_adj.reshape(-1, C), labels.reshape(-1))

def loss_function_projection(
    features: torch.Tensor,
    labels: torch.Tensor,
    anchors: nn.Parameter,
    roi_class_counts: Optional[torch.Tensor] = None,
    penalty_weight: float = 10.0
) -> Tuple[torch.Tensor, dict]:
    
    device = features.device
    C = anchors.shape[0]
    
    u = F.normalize(features, p=2, dim=-1)
    a = F.normalize(anchors, p=2, dim=-1)
    target_anchors = a[labels] 

    direct_cos = (u * target_anchors).sum(dim=-1)
    L_align = 1.0 - direct_cos

    w = F.normalize(a[C-1] - a[0], p=2, dim=-1)
    
    u_proj_cos = (u * w).sum(dim=-1).clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
    a_proj_cos = (a * w).sum(dim=-1).clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
    
    u_theta = torch.acos(u_proj_cos)
    a_theta = torch.acos(a_proj_cos)
    
    L_OneD = F.l1_loss(u_theta, a_theta[labels], reduction='none')

    L_obtuse = F.relu(-direct_cos) * penalty_weight
    
    L_severity = L_align + L_OneD + L_obtuse

    if roi_class_counts is not None:
        counts = roi_class_counts.to(device=device, dtype=features.dtype)
        inv_freq = 1.0 / counts.clamp(min=1.0)
        weights_rc = (inv_freq / inv_freq.sum(dim=-1, keepdim=True)) * C
        
        r_idx = torch.arange(u.shape[1], device=device).unsqueeze(0).expand(u.shape[0], -1)
        sample_weights = weights_rc[r_idx, labels]
        
        L_severity_mean = (L_severity * sample_weights).mean()
    else:
        L_severity_mean = L_severity.mean()

    L_polar = (1.0 + (a[0] * a[C-1]).sum()).pow(2)
    
    target_cos = math.cos(math.pi / (C - 1))
    adj_cos = (a[:-1] * a[1:]).sum(dim=-1)
    L_ord = F.mse_loss(adj_cos, torch.full_like(adj_cos, target_cos))

    total_loss = L_severity_mean + L_polar + L_ord
    
    return total_loss, {
        "loss": total_loss.item(),
        "L_align": L_align.mean().item(),
        "L_1D": L_OneD.mean().item(),
        "L_obtuse": L_obtuse.mean().item(),
        "L_polar": L_polar.item(),
        "L_ord": L_ord.item(),
    }

def compute_alpha_oracle(pi_gt, labels, p_model_logits, oracle_tau=0.1):
    B, R = labels.shape
    C = pi_gt.shape[-1]
    device = labels.device

    p_gt = F.one_hot(labels.long(), C).float()
    q_all_gt = torch.einsum("bijcd,bjd->bijc", pi_gt, p_gt)

    p_self = F.softmax(p_model_logits.detach(), dim=-1)
    q_self = torch.einsum("bijcd,bjd->bijc", pi_gt, p_self)

    mask = torch.eye(R, device=device).bool().reshape(1, R, R, 1)
    q_hybrid = torch.where(mask, q_self, q_all_gt)

    class_idx = torch.arange(C, device=device).float()
    E_q = (q_hybrid * class_idx).sum(dim=-1)
    dist = (E_q - labels.float().unsqueeze(-1)).abs()
    var = (q_hybrid * (class_idx - E_q.unsqueeze(-1)).pow(2)).sum(-1)

    alpha_target = F.softmax(-dist / oracle_tau, dim=-1).detach()

    pred = p_model_logits.detach().argmax(dim=-1)
    is_wrong = (pred != labels).float()
    raw_conf = (-dist.min(dim=-1).values).exp()
    confidence = (is_wrong * raw_conf).detach()

    return alpha_target, confidence
    
def kl_attention_loss(alpha_pred, alpha_target, confidence, eps=1e-9):
    log_pred = (alpha_pred + eps).log()
    kl = F.kl_div(log_pred, alpha_target, reduction='none').sum(-1)
    denom = confidence.sum().clamp(min=1e-6)
    return (kl * confidence).sum() / denom

def mse_attention_loss(alpha_pred, alpha_target, confidence):
    mse = (alpha_pred - alpha_target).pow(2).mean(dim=-1)
    denom = confidence.sum().clamp(min=1e-6)
    return (mse * confidence).sum() / denom


def compute_class_counts(dataloader, num_classes: int = 4, num_regions: int = 6) -> tuple:
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    roi_class_counts = torch.zeros(num_regions, num_classes, dtype=torch.long)
    
    for batch in dataloader:
        labels = batch[1]
        
        for r in range(num_regions):
            for c in range(num_classes):
                count = (labels[:, r] == c).sum().item()
                class_counts[c] += count
                roi_class_counts[r, c] += count
    
    return class_counts, roi_class_counts


def compute_class_counts_from_df(df, num_classes: int = 4, num_regions: int = 6) -> tuple:
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    roi_class_counts = torch.zeros(num_regions, num_classes, dtype=torch.long)
    
    for r in range(num_regions):
        col = f"brixia{r+1}"
        if col in df.columns:
            values = df[col].values
            for c in range(num_classes):
                count = (values == c).sum()
                class_counts[c] += count
                roi_class_counts[r, c] += count
    
    return class_counts, roi_class_counts
