"""Attention extraction.

Two routes, because no single one covers every backbone:

1. Built-in (`output_attentions=True` with `attn_implementation="eager"`) — cleanest
   for transformers models that support it.
2. Forward hooks — a fallback that works everywhere:
     * timm Attention stores the post-softmax matrix, so hooking `attn_drop` catches it
       directly (no recomputation, exact).
     * plain q_proj/k_proj/v_proj modules (transformers Dinov2/Dinov3 attention) get
       the matrix recomputed from the module's own projections.

Shape convention throughout the package: attention is `(heads, T, T)` for one image,
where T = 1 cls + num_registers + num_patches.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class AttentionStore:
    """Collects per-layer attention matrices during a forward pass."""

    def __init__(self):
        self.mats: dict[int, torch.Tensor] = {}
        self._order: list[int] = []

    def add(self, idx: int, mat: torch.Tensor):
        self.mats[idx] = mat.detach()
        if idx not in self._order:
            self._order.append(idx)

    def stacked(self, dtype=torch.float32) -> Optional[torch.Tensor]:
        """(layers, heads, T, T) or None."""
        if not self.mats:
            return None
        keys = sorted(self.mats)
        return torch.stack([self.mats[k].to(dtype) for k in keys], dim=0)

    def clear(self):
        self.mats.clear()
        self._order.clear()

    def __len__(self):
        return len(self.mats)


def install_attention_hooks(model: nn.Module) -> tuple[AttentionStore, list]:
    """Attach hooks that populate an AttentionStore. Returns (store, handles).

    For timm backbones this also disables fused attention: timm can route through
    `F.scaled_dot_product_attention`, which never materialises the softmax matrix, so a
    hook on `attn_drop` would silently capture nothing. Forcing the explicit path makes
    the hook fire with the model's own (RoPE-included) attention.
    """
    store = AttentionStore()
    handles = []

    # --- route A: timm-style, hook the attention-probability dropout module -------
    timm_targets = []
    for name, mod in model.named_modules():
        if name.endswith("attn_drop") and isinstance(mod, (nn.Dropout, nn.Identity)):
            parent = name.rsplit(".", 1)[0] if "." in name else ""
            timm_targets.append((parent, mod))

    if timm_targets:
        seen_parents = set()
        for parent_name, _mod in timm_targets:
            if parent_name in seen_parents:
                continue
            seen_parents.add(parent_name)
            parent = model.get_submodule(parent_name) if parent_name else model
            if hasattr(parent, "fused_attn"):
                parent.fused_attn = False
        for layer_i, (parent, mod) in enumerate(timm_targets):
            def hook(_m, _inp, _out, i=layer_i):
                if _inp and torch.is_tensor(_inp[0]):
                    t = _inp[0].detach()
                    # timm hands the dropout module (B, heads, T, T); drop the batch dim
                    store.add(i, t[0] if t.dim() == 4 else t)
            handles.append(mod.register_forward_hook(hook))
        return store, handles

    # --- route B: recompute from q/k/v projections --------------------------------
    qkv_blocks = []
    for name, mod in model.named_modules():
        if all(hasattr(mod, a) for a in ("q_proj", "k_proj", "v_proj")):
            qkv_blocks.append((name, mod))

    for layer_i, (name, mod) in enumerate(qkv_blocks):
        def make_hook(m, i):
            def hook(_m, inputs, _out):
                x = inputs[0]
                try:
                    b, t, _ = x.shape
                    q = m.q_proj(x)
                    k = m.k_proj(x)
                    v = m.v_proj(x)  # noqa: F841  (kept for symmetry/debugging)
                    heads = getattr(m, "num_heads", None) or getattr(m, "num_attention_heads", None)
                    if heads is None:
                        heads = getattr(m, "heads", 1)
                    d = q.shape[-1] // heads
                    q = q.view(b, t, heads, d).transpose(1, 2)
                    k = k.view(b, t, heads, d).transpose(1, 2)
                    scale = getattr(m, "scale", d ** -0.5)
                    attn = (q @ k.transpose(-2, -1)) * scale
                    attn = attn.softmax(dim=-1)
                    store.add(layer_i, attn[0])
                except Exception:  # pragma: no cover - best effort only
                    pass
            return hook
        handles.append(mod.register_forward_hook(make_hook(mod, layer_i)))

    return store, handles


def remove_hooks(handles: list):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


# --------------------------------------------------------------------------- maths

def attention_rollout(attn: torch.Tensor, discard_ratio: float = 0.0,
                      head_fusion: str = "mean", add_residual: bool = True) -> torch.Tensor:
    """Attention rollout (Abnar & Zuidema).

    attn: (layers, heads, T, T). Returns (T_from_cls,) importance over tokens,
    normalised to sum 1, with the cls token's own entry set to 0.
    """
    if attn.dim() != 4:
        raise ValueError(f"expected (layers, heads, T, T), got {tuple(attn.shape)}")
    a = attn.detach().float().clone()

    if head_fusion == "mean":
        a = a.mean(dim=1)
    elif head_fusion == "max":
        a = a.max(dim=1).values
    elif head_fusion == "min":
        a = a.min(dim=1).values
    else:
        raise ValueError("head_fusion must be mean/max/min")

    if discard_ratio > 0:
        flat = a.flatten(1)
        k = int(flat.shape[1] * discard_ratio)
        if k > 0:
            thresh = flat.kthvalue(flat.shape[1] - k, dim=1, keepdim=True).values
            flat = torch.where(flat < thresh, torch.zeros_like(flat), flat)
            a = flat.view_as(a)

    if add_residual:
        a = a + torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)

    a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    result = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    for layer in a:
        result = layer @ result
    # row 0 = the cls token's accumulated attention to everything else
    return result[0]


def attention_entropy(attn: torch.Tensor, num_prefix: int = 1) -> np.ndarray:
    """Per-layer, per-head normalised entropy of attention from the cls token.

    attn: (layers, heads, T, T). Returns (layers, heads) in [0, 1] where 1 is
    perfectly uniform (diffuse) and 0 is focused on a single patch.
    """
    a = attn.detach().float()
    cls_to_patches = a[:, :, 0, num_prefix:]                      # (L, H, P)
    p = cls_to_patches.clamp_min(1e-12)
    p = p / p.sum(dim=-1, keepdim=True)
    ent = -(p * p.log()).sum(dim=-1)
    denom = np.log(max(p.shape[-1], 2))
    return (ent / denom).cpu().numpy()


def per_head_cls_map(attn: torch.Tensor, layer: int, head: int, grid: tuple[int, int],
                     num_prefix: int = 1) -> np.ndarray:
    """cls -> patch attention reshaped to a (h, w) heatmap for one layer/head."""
    a = attn.detach().float()[layer, head, 0, num_prefix:]
    h, w = grid
    return a[: h * w].reshape(h, w).cpu().numpy()


def head_map_mean(attn: torch.Tensor, layer: int, grid: tuple[int, int],
                  num_prefix: int = 1, mode: str = "mean") -> np.ndarray:
    """Pool the cls->patch attention of all heads at one layer into a heatmap."""
    a = attn.detach().float()[layer, :, 0, num_prefix:]           # (H, P)
    if mode == "mean":
        v = a.mean(0)
    elif mode == "max":
        v = a.max(0).values
    elif mode == "min":
        v = a.min(0).values
    else:
        raise ValueError("mode must be mean/max/min")
    h, w = grid
    return v[: h * w].reshape(h, w).cpu().numpy()


def mean_key_attention(attn: torch.Tensor, num_prefix: int = 1) -> np.ndarray:
    """Average attention received by each patch token across all query tokens,
    all heads and all layers — the 'how much is this patch attended to' signal."""
    a = attn.detach().float()
    per_layer = a[:, :, :, num_prefix:].mean(dim=(1, 2))          # (L, P)
    return per_layer.mean(dim=0).cpu().numpy()


def layer_head_summary(attn: torch.Tensor, num_prefix: int = 1) -> dict:
    """Compact per-layer stats used by the UI: entropy, focus, locality."""
    a = attn.detach().float()
    L, H, T, _ = a.shape
    ent = attention_entropy(a, num_prefix=num_prefix)             # (L, H)
    cls = a[:, :, 0, num_prefix:]                                 # (L, H, P)
    top1 = cls.max(dim=-1).values                                 # (L, H)
    return {
        "entropy": ent,
        "entropy_layer_mean": ent.mean(axis=1),
        "top1_mass": top1.cpu().numpy(),
        "top1_layer_mean": top1.mean(dim=1).cpu().numpy(),
    }
