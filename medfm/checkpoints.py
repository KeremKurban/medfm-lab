"""Loader for medical checkpoints published in the DINOv3/DINOv2 reference format.

A lot of medical foundation models are released as a bare `model.pth` produced by the
facebookresearch training code, not as a `transformers` or `timm` repository. Those state
dicts are DINO-format: fused `attn.qkv`, `ls1.gamma`/`ls2.gamma` layer scales, RoPE
periods, `storage_tokens` registers.

timm ships matching DINOv3/DINOv2 architectures key-for-key apart from two cosmetic
renames (`ls1.gamma` -> `gamma_1`, `ls2.gamma` -> `gamma_2`), so we build the timm
architecture and remap. Verified against MedDINOv3 ViT-B/16 (CT-3M): 137/137 loadable
tensors, zero shape mismatches.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

CHECKPOINT_DIR = os.path.expanduser("~/medfm-lab/data/checkpoints")


def _strip_prefix(sd: dict, prefix: Optional[str]) -> dict:
    if not prefix:
        return dict(sd)
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out or dict(sd)


def _unwrap_container(sd) -> dict:
    """Some releases wrap the weights as {'teacher': {...}} or {'model': {...}}."""
    if not isinstance(sd, dict):
        return sd
    looks_like_weights = any(torch.is_tensor(v) for v in sd.values())
    if looks_like_weights:
        return sd
    for key in ("teacher", "model", "student", "state_dict", "backbone"):
        if key in sd and isinstance(sd[key], dict):
            return _unwrap_container(sd[key])
    return sd


def remap_dino_keys(sd: dict, drop_prefixes: tuple = ("mask_token",)) -> tuple[dict, list]:
    """Map reference-implementation key names onto timm names."""
    out = {}
    dropped = []
    for k, v in sd.items():
        if any(k.startswith(p) or k.endswith(p) for p in drop_prefixes):
            dropped.append(k)
            continue
        nk = k
        nk = nk.replace(".ls1.gamma", ".gamma_1").replace(".ls2.gamma", ".gamma_2")
        # timm names the register set `reg_token` in some versions, `storage_tokens` in others
        if nk in ("storage_tokens", "reg_token"):
            nk = "reg_token"
        out[nk] = v
    return out, dropped


def download_checkpoint(repo_id: str, filename: str = "model.pth",
                        subdir: Optional[str] = None, token: Optional[str] = None) -> str:
    from huggingface_hub import hf_hub_download

    subdir = subdir or repo_id.split("/")[-1]
    local_dir = os.path.join(CHECKPOINT_DIR, subdir)
    os.makedirs(local_dir, exist_ok=True)
    return hf_hub_download(repo_id, filename, local_dir=local_dir, token=token)


def load_into_timm(arch: str, ckpt_path: str, prefix: Optional[str] = "backbone.",
                   num_classes: int = 0, verbose: bool = True) -> tuple[nn.Module, dict]:
    """Build a timm model and load a DINO-format checkpoint into it."""
    import timm

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = _unwrap_container(raw)
    stripped = _strip_prefix(raw, prefix)
    remapped, dropped = remap_dino_keys(stripped)

    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    sd = model.state_dict()

    common = set(sd) & set(remapped)
    missing = sorted(set(sd) - set(remapped))
    unexpected = sorted(set(remapped) - set(sd))
    wrong_shape = [k for k in common if tuple(sd[k].shape) != tuple(remapped[k].shape)]

    loadable = {k: remapped[k] for k in common if k not in wrong_shape}
    info = model.load_state_dict(loadable, strict=False)

    report = {
        "arch": arch,
        "ckpt": ckpt_path,
        "n_checkpoint_tensors": len(remapped),
        "n_loaded": len(loadable),
        "n_model_tensors": len(sd),
        "missing_in_checkpoint": missing,
        "unexpected_in_checkpoint": unexpected,
        "shape_mismatch": wrong_shape,
        "dropped": dropped,
        "coverage": len(loadable) / max(len(sd), 1),
    }
    if verbose:
        print(f"[checkpoints] {os.path.basename(ckpt_path)} -> {arch}: "
              f"{len(loadable)}/{len(sd)} tensors "
              f"({report['coverage'] * 100:.1f}%)")
        if missing:
            print(f"  missing   : {missing[:6]}{' ...' if len(missing) > 6 else ''}")
        if wrong_shape:
            print(f"  shape diff: {wrong_shape[:6]}")
        if unexpected:
            print(f"  unexpected: {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}")

    if report["coverage"] < 0.9:
        raise RuntimeError(
            f"checkpoint only covers {report['coverage'] * 100:.1f}% of {arch} — "
            f"architecture mismatch, refusing to use a half-initialised model"
        )
    return model, report
