"""Diagnostic: load every registered model that can be loaded, run one image through it,
and verify attention + residual-stream capture actually produces sane numbers.

Run:  .venv/bin/python scripts/check_models.py [--all]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from medfm import encoders, registry
from medfm.attention import attention_rollout, layer_head_summary, mean_key_attention
from medfm.embeddings import anisotropy, effective_rank, outlier_tokens, pca_rgb
from medfm.residual import cka_matrix, representation_drift, residual_norm_profile


def make_test_image(size: int = 224):
    from PIL import Image

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:size, 0:size]
    disk = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size / 4) ** 2
    arr = np.zeros((size, size), dtype=np.float32)
    arr[disk] = 200
    arr += rng.normal(0, 12, (size, size))
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr] * 3, -1))


def check(key: str, backend: str = "auto", device: str | None = None) -> dict:
    spec = registry.get_spec(key)
    rec: dict = {"key": key, "backend": backend, "status": "?"}
    t0 = time.time()
    try:
        enc = encoders.Encoder.load(key, backend=backend, device=device)
        rec["load_s"] = round(time.time() - t0, 2)
        rec["device"] = str(enc.device)
        if enc.ckpt_report:
            rec["ckpt_coverage"] = round(enc.ckpt_report["coverage"], 4)

        img = make_test_image(spec.image_size)
        t1 = time.time()
        out = enc(img, capture_attention=True)
        rec["forward_s"] = round(time.time() - t1, 2)

        rec["grid"] = list(out.grid)
        rec["num_prefix"] = out.num_prefix
        rec["n_patches"] = int(out.patch_tokens.shape[0])
        rec["embed_dim"] = int(out.patch_tokens.shape[1])
        rec["n_layers"] = int(out.hidden_states.shape[0])
        rec["finite"] = bool(np.isfinite(out.patch_tokens).all())

        prof = residual_norm_profile(out.hidden_states, num_prefix=out.num_prefix)
        rec["cls_norm_first"] = round(float(prof["cls_norms"][0]), 3)
        rec["cls_norm_last"] = round(float(prof["cls_norms"][-1]), 3)
        rec["patch_norm_last"] = round(float(prof["patch_norm_mean"][-1]), 3)

        ot = outlier_tokens(out.hidden_states, layer=-1)
        rec["n_outlier_tokens"] = ot["n_outliers"]
        rec["max_z_norm"] = round(ot["max_z"], 2)

        rec["effective_rank"] = round(effective_rank(out.patch_tokens), 2)
        rec["anisotropy"] = round(anisotropy(out.patch_tokens), 4)

        rgb, mask, info = pca_rgb(out.patch_tokens, out.grid)
        rec["pca_evr"] = [round(v, 3) for v in info["explained_variance_ratio"][:3]]
        rec["pca_rgb_range"] = [round(float(rgb.min()), 3), round(float(rgb.max()), 3)]

        drift = representation_drift(out.hidden_states, num_prefix=out.num_prefix)
        rec["drift_mean"] = round(float(drift.mean()), 4) if drift.size else None
        rec["drift_min"] = round(float(drift.min()), 4) if drift.size else None

        if out.attentions is not None:
            a = torch.from_numpy(out.attentions)
            L, H, T, T2 = a.shape
            rec["attn_shape"] = [L, H, T, T2]
            rec["attn_rows_sum_to_1"] = bool(
                torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-3)
            )
            summ = layer_head_summary(a, num_prefix=out.num_prefix)
            rec["attn_entropy_mean"] = round(float(summ["entropy"].mean()), 4)
            rec["attn_entropy_layer0"] = round(float(summ["entropy_layer_mean"][0]), 4)
            rec["attn_entropy_layerlast"] = round(float(summ["entropy_layer_mean"][-1]), 4)
            roll = attention_rollout(a)
            rec["rollout_sum"] = round(float(roll.sum()), 4)
            rec["rollout_concentrated"] = bool(roll.max() > 1.0 / max(T - 1, 1))
            mk = mean_key_attention(a, num_prefix=out.num_prefix)
            rec["mean_key_attn_len"] = int(mk.shape[0])
        else:
            rec["attn_shape"] = None

        rec["status"] = "OK"
        return rec
    except Exception as e:
        rec["status"] = "FAIL"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-1200:]
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include gated models too")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", default="outputs/model_check.json")
    args = ap.parse_args()

    specs = registry.list_specs()
    if args.only:
        specs = [s for s in specs if s.key in args.only]
    elif not args.all:
        specs = [s for s in specs if not s.gated]

    records = []
    for s in specs:
        backend = "auto"
        print(f"\n### {s.key}  ({backend})  — {s.label}")
        rec = check(s.key, backend=backend, device=args.device)
        records.append(rec)
        if rec["status"] == "OK":
            print(f"    OK  load={rec['load_s']}s fwd={rec['forward_s']}s "
                  f"grid={rec['grid']} patches={rec['n_patches']} dim={rec['embed_dim']} "
                  f"layers={rec['n_layers']}")
            print(f"    norms cls {rec['cls_norm_first']} -> {rec['cls_norm_last']}, "
                  f"outliers={rec['n_outlier_tokens']} maxz={rec['max_z_norm']}")
            print(f"    eff_rank={rec['effective_rank']} anis={rec['anisotropy']} "
                  f"pca_evr={rec['pca_evr']} drift={rec['drift_mean']}")
            if rec.get("attn_shape"):
                print(f"    attn {rec['attn_shape']} rowsum_ok={rec['attn_rows_sum_to_1']} "
                      f"entropy L0={rec['attn_entropy_layer0']} "
                      f"Llast={rec['attn_entropy_layerlast']} "
                      f"rollout_sum={rec['rollout_sum']}")
        else:
            print(f"    FAIL {rec['error']}")

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nwrote {args.json}")
    ok = sum(1 for r in records if r["status"] == "OK")
    print(f"{ok}/{len(records)} models OK")


if __name__ == "__main__":
    main()
