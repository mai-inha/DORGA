import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from functools import partial
from timm.models.vision_transformer import VisionTransformer


def vit_base_patch16_512(**kwargs) -> VisionTransformer:
    return VisionTransformer(
        img_size=512,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def _adapt_checkpoint_patch_embed(state_dict: Dict, model: VisionTransformer, mode: str = "mean"):
    key = "patch_embed.proj.weight"
    if key not in state_dict:
        return state_dict
    w_ckpt = state_dict[key]
    w_mod = model.patch_embed.proj.weight
    if w_ckpt.shape == w_mod.shape:
        return state_dict
    Cin_ckpt, Cin_mod = w_ckpt.shape[1], w_mod.shape[1]
    if Cin_ckpt == 3 and Cin_mod == 1:
        state_dict[key] = w_ckpt.mean(1, keepdim=True) if mode == "mean" else w_ckpt[:, :1]
    elif Cin_ckpt == 1 and Cin_mod == 3:
        state_dict[key] = w_ckpt.repeat(1, 3, 1, 1)
    return state_dict


def _resize_pos_embed(state_dict: Dict, model: VisionTransformer, verbose: bool = False):
    if "pos_embed" not in state_dict:
        return state_dict
    pe_ckpt = state_dict["pos_embed"]
    pe_mod = model.pos_embed
    if pe_ckpt.shape == pe_mod.shape:
        return state_dict
    cls_ckpt = pe_ckpt[:, :1]
    grid_ckpt = pe_ckpt[:, 1:]
    Nold = grid_ckpt.shape[1]
    gs_old = int(Nold**0.5)
    grid_ckpt = grid_ckpt.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    gs_new = model.patch_embed.grid_size[0]
    grid_new = F.interpolate(grid_ckpt, size=(gs_new, gs_new), mode="bicubic", align_corners=False)
    grid_new = grid_new.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    state_dict["pos_embed"] = torch.cat([cls_ckpt, grid_new], dim=1)
    return state_dict


def load_mae_ckpt_to_512(
    path: str,
    num_classes: int = 4,
    in_chans: int = 1,
    drop_path_rate: float = 0.1,
    global_pool: str = "avg",
    patch_mode: str = "mean",
    verbose: bool = False,
) -> VisionTransformer:
    model = vit_base_patch16_512(
        num_classes=num_classes,
        in_chans=in_chans,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool,
    )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    state = _adapt_checkpoint_patch_embed(state, model, mode=patch_mode)
    state = _resize_pos_embed(state, model, verbose=verbose)

    if "head.weight" in state:
        if state["head.weight"].shape != model.head.weight.shape:
            state.pop("head.weight", None)
            state.pop("head.bias", None)
            if verbose:
                print(f"Removed head.weight due to shape mismatch.")

    model.load_state_dict(state, strict=False)
    return model
