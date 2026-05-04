import torch
import torch.nn as nn

def _set_trainable(model, freeze_blocks: int = 8):
    for p in model.parameters():
        p.requires_grad = True

    if hasattr(model, "vit") and hasattr(model.vit, "blocks"):
        for blk in model.vit.blocks[:freeze_blocks]:
            for p in blk.parameters():
                p.requires_grad = False

    if hasattr(model, "pattern_predictor"):
        for p in model.pattern_predictor.parameters():
            p.requires_grad = True

    if hasattr(model, "cls_head"):
        for p in model.cls_head.parameters():
            p.requires_grad = True

    if getattr(model, "shared_projector", None) is not None:
        for p in model.shared_projector.parameters():
            p.requires_grad = True

    if getattr(model, "roi_specific_projector", None) is not None:
        for p in model.roi_specific_projector.parameters():
            p.requires_grad = True

    if getattr(model, "patch_importance", None) is not None:
        for p in model.patch_importance.parameters():
            p.requires_grad = True

    if getattr(model, "gnn", None) is not None:
        for p in model.gnn.parameters():
            p.requires_grad = True

    if hasattr(model, "class_anchors"):
        model.class_anchors.requires_grad = True


def _make_optimizer(model: nn.Module):
    wd_enc = 5e-5
    wd_head = 5e-4

    enc_params, enc_names = [], []
    pattern_params, pattern_names = [], []
    stage1_params, stage1_names = [], []
    shared_proj_params, shared_proj_names = [], []
    roi_specific_proj_params, roi_proj_names = [], []

    gnn_att_proj_params, gnn_att_proj_names = [], []
    gnn_att_scale_params, gnn_att_scale_names = [], []        
    gnn_step_params, gnn_step_names = [], []

    head_params, head_names = [], []

    added_params = set()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in added_params:
            continue

        if name.startswith("vit."):
            enc_params.append(p)
            enc_names.append(name)
        elif name.startswith("pattern_predictor."):
            pattern_params.append(p)
            pattern_names.append(name)
        elif name.startswith("patch_importance."):
            stage1_params.append(p)
            stage1_names.append(name)
        elif name.startswith("shared_projector."):
            shared_proj_params.append(p)
            shared_proj_names.append(name)
        elif name.startswith("roi_specific_projector."):
            roi_specific_proj_params.append(p)
            roi_proj_names.append(name)
        elif name.startswith("gnn."):
            if "step_size" in name:
                gnn_step_params.append(p)
                gnn_step_names.append(name)
            elif "q_proj" in name or "k_proj" in name:
                gnn_att_proj_params.append(p)
                gnn_att_proj_names.append(name)
            elif "att_scale" in name:
                gnn_att_scale_params.append(p)
                gnn_att_scale_names.append(name)
            else:
                head_params.append(p)
                head_names.append(name)
        else:
            head_params.append(p)
            head_names.append(name)

        added_params.add(id(p))

    param_groups = []

    if enc_params:
        param_groups.append({"params": enc_params, "lr": 1e-5, "weight_decay": wd_enc})
    if pattern_params:
        param_groups.append({"params": pattern_params, "lr": 1e-4, "weight_decay": wd_head})
    if stage1_params:
        param_groups.append({"params": stage1_params, "lr": 5e-5, "weight_decay": wd_enc})
    if shared_proj_params:
        param_groups.append({"params": shared_proj_params, "lr": 1e-4, "weight_decay": wd_head})
    if roi_specific_proj_params:
        param_groups.append({"params": roi_specific_proj_params, "lr": 1e-3, "weight_decay": wd_head})
    if gnn_att_proj_params:
        param_groups.append({"params": gnn_att_proj_params, "lr": 1e-4, "weight_decay": wd_head})
    if gnn_att_scale_params:                                
        param_groups.append({"params": gnn_att_scale_params, "lr": 1e-2, "weight_decay": 0.0})
    if gnn_step_params:
        param_groups.append({"params": gnn_step_params, "lr": 1e-4, "weight_decay": 0.0})
    if head_params:
        param_groups.append({"params": head_params, "lr": 1e-4, "weight_decay": wd_head})

    print(f"[Optimizer] {len(param_groups)} param groups configured")
    return torch.optim.AdamW(param_groups)


def _print_trainable_summary(model, stage_name: str):
    trainable = 0
    frozen = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    
    print(f"[{stage_name}] Trainable: {trainable/1e6:.2f}M, Frozen: {frozen/1e6:.2f}M")
