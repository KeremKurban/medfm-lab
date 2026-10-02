"""Smoke test: environment, model loading, attention capture, MedMNIST+ download.

Run:  .venv/bin/python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

print(f"torch {torch.__version__}")
print(f"mps available : {torch.backends.mps.is_available()}")
print(f"mps built     : {torch.backends.mps.is_built()}")
if torch.backends.mps.is_available():
    print(f"mps device    : {torch.device('mps')}")

import transformers

print(f"transformers  : {transformers.__version__}")
import timm

print(f"timm          : {timm.__version__}")

# ---------------------------------------------------------------- timm DINOv3 ids
print("\n--- timm dinov3 models ---")
try:
    dinos = timm.list_models("*dinov3*", pretrained=True)
    for m in dinos[:20]:
        print("  ", m)
except Exception as e:
    print("  err:", e)

# ---------------------------------------------------------------- test image
from PIL import Image

rng = np.random.default_rng(0)
h = w = 224
yy, xx = np.mgrid[0:h, 0:w]
disk = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (h / 4) ** 2
arr = np.zeros((h, w), dtype=np.float32)
arr[disk] = 200
arr += rng.normal(0, 12, (h, w))
arr = np.clip(arr, 0, 255).astype(np.uint8)
test_img = Image.fromarray(np.stack([arr] * 3, -1))
print(f"\ntest image: {test_img.size} {test_img.mode}")

# ---------------------------------------------------------------- encoders
from medfm import encoders  # noqa: E402
from medfm.attention import attention_rollout, layer_head_summary  # noqa: E402
from medfm.residual import cka_matrix, residual_norm_profile  # noqa: E402
from medfm.embeddings import pca_rgb, outlier_tokens  # noqa: E402

CANDIDATES = [
    ("dinov2-base", "hf"),
    ("rad-dino", "hf"),
    ("meddinov3-vitb16", "hf"),
    ("dinov3-vits16plus", "timm"),
]

results = {}
for key, backend in CANDIDATES:
    print(f"\n=== {key} ({backend}) ===")
    try:
        enc = encoders.Encoder.load(key, backend=backend)
        out = enc(test_img, capture_attention=True)
        print(f"  device        : {enc.device}")
        print(f"  grid          : {out.grid}   num_prefix={out.num_prefix}")
        print(f"  pooled        : {out.pooled.shape}")
        print(f"  patch_tokens  : {out.patch_tokens.shape}")
        print(f"  hidden_states : {out.hidden_states.shape}")
        if out.attentions is not None:
            a = torch.from_numpy(out.attentions)
            print(f"  attentions    : {out.attentions.shape}")
            summ = layer_head_summary(a, num_prefix=out.num_prefix)
            print(f"  attn entropy  : mean={summ['entropy'].mean():.4f} "
                  f"(L0={summ['entropy_layer_mean'][0]:.4f} "
                  f"Llast={summ['entropy_layer_mean'][-1]:.4f})")
            roll = attention_rollout(a)
            print(f"  rollout       : {roll.shape} sum={roll.sum():.4f}")
        else:
            print("  attentions    : NONE")
        rgb, mask, info = pca_rgb(out.patch_tokens, out.grid)
        print(f"  pca rgb       : {rgb.shape} evr={[round(v,3) for v in info['explained_variance_ratio'][:3]]}")
        ot = outlier_tokens(out.hidden_states, layer=-1)
        print(f"  outlier tokens: n={ot['n_outliers']} max_z={ot['max_z']:.2f}")
        prof = residual_norm_profile(out.hidden_states, num_prefix=out.num_prefix)
        print(f"  cls norms     : first={prof['cls_norms'][0]:.2f} last={prof['cls_norms'][-1]:.2f}")
        M = cka_matrix(out.hidden_states[:4], num_prefix=out.num_prefix)
        print(f"  cka (4 layers): diag={np.diag(M)}")
        results[key] = "OK"
    except Exception as e:
        results[key] = f"FAIL: {type(e).__name__}: {e}"
        traceback.print_exc()

# ---------------------------------------------------------------- MedMNIST+
print("\n=== MedMNIST+ 224 ===")
try:
    from medfm import data as mdata

    meta = mdata.dataset_meta("pneumoniamnist")
    print("  meta:", {k: v for k, v in meta.items() if k != "label_map"})
    samples = mdata.fetch_samples("pneumoniamnist", split="test", n=4, size=224)
    print(f"  fetched {len(samples)} samples, image shape {samples[0].image.shape}, "
          f"label {samples[0].label}")
    results["medmnist"] = "OK"
except Exception as e:
    results["medmnist"] = f"FAIL: {type(e).__name__}: {e}"
    traceback.print_exc()

print("\n=== summary ===")
for k, v in results.items():
    print(f"  {k:22s} {v}")
