# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-10-02

First tagged release. Validated end to end on `pneumoniamnist` (chest X-ray) and
`retinamnist` (fundus) with RAD-DINO on Apple Silicon / MPS.

### Added

**Backbone loading**
- Unified loader for DINOv2, DINOv3 (ViT-S/S+/B/L, ConvNeXt) and four medical
  adaptations: RAD-DINO, RAD-DINO-MAIRA-2, MedDINOv3 and CXformer.
- `medfm/checkpoints.py` — loads medical checkpoints published as raw DINO
  reference-format `.pth` files by building the matching timm architecture and remapping
  keys (`ls1.gamma` → `gamma_1`). Refuses any load under 90% tensor coverage rather than
  running a half-initialised model.
- Automatic fallback to the ungated timm mirror when a gated `facebook/dinov3-*` repo
  returns 401/403, so the lab works without a licence click.
- Per-backbone preprocessing: each model's own HF image processor by default, with a
  `uniform` mode for pixel-fair cross-model comparison.
- `Encoder.preview()` — de-normalises the tensor back to viewable RGB so you can see the
  resolution and intensity mapping the network actually received.

**Interpretability**
- Attention capture via the model's own `output_attentions`, with a forward-hook fallback
  that forces timm off its fused-attention path (which never materialises the softmax
  matrix and silently yields no attention at all).
- Per-layer/head cls→patch maps, head pooling, attention rollout with a discard ratio,
  and normalised entropy over layer × head.
- Patch embeddings: PCA→RGB, PC1 foreground mask, token UMAP with k-means clusters,
  patch-similarity maps, positional debiasing for DINOv3's coordinate bias, token-norm
  growth and high-norm outlier detection.
- Residual stream: per-block capture for both HF and timm backbones, layer-wise CKA,
  representation drift, norm profiles, and a layer-wise linear probe that reports where
  the label is linearly decodable and suggests a concrete adapter block range.
- Frozen-feature linear and kNN probes plus a cosine `FeatureIndex` for retrieval.

**UI** (`app.py`, Gradio)
- Five tabs: Inference, Attention, Patch tokens, Residual stream, Models.
- Inference laid out as three numbered steps with results on their own row; four result
  panels drawn at matching size — what the model saw, patch PCA→RGB, PC1 foreground mask,
  and kNN retrieval.
- Per-tab "What can I do on this tab?" guides, tooltips on every control, and a live
  guidance line stating the next action and what it unlocks.
- Ground truth is always stated for MedMNIST+ samples, and retrieval reports whether the
  nearest neighbour agrees with it.

**Data**
- MedMNIST+ at 224px across 12 datasets, with exact per-set download sizes shown in the
  dropdown, the list ordered smallest-first, and a warning with a time estimate before any
  first download.
- `data.ensure_intact()` removes truncated cache files, which medmnist would otherwise
  accept silently and then fail on deep inside `np.load`.

**Tooling**
- `run.sh` — start/stop the UI from a terminal (`--detach`, `--port`).
- `NOTICE.md` — third-party licence terms for every model and dataset the lab fetches,
  separated from `LICENSE` so the code's MIT licence stays machine-detectable.
- `scripts/check_models.py` — loads every registered backbone, runs one image, asserts
  sane attention and residual-stream statistics. 6/6 passing.
- `scripts/test_app.py` — 34 headless checks covering every UI callback, the dropdown
  value path, image provenance, panel geometry and the retrieval self-match guard.
- `scripts/smoke_test.py` — environment and dependency check.

### Fixed

- **Dropdown label/value swap.** Gradio validates a `Dropdown`'s `value` against a choice's
  *value* in Python but against its *name* in the frontend, so `(name, value)` tuples
  produced contradictory errors and an app that rendered but never responded. All
  dropdowns now use plain strings with the label parsed back in the callback.
- **Tracebacks in the UI.** Errors render as a short line plus a hint; full tracebacks are
  appended to `logs/errors.log` so absolute paths and library versions cannot leak into a
  screenshot or screen share.
- **Retrieval self-match.** The query could appear in its own reference gallery and score
  similarity 1.0, guaranteeing a label match and making retrieval look perfect while
  measuring nothing. Source indices are now tracked and the query is excluded.
- **Patch-grid maps rendered at native size.** A 37×37 grid at 518px input is genuinely a
  37-pixel image; maps are now upscaled with nearest-neighbour to the model-input
  resolution so they line up patch for patch.
- **Image provenance race.** Gradio re-emits an `Image` change event after our own callback
  sets it, overwriting the dataset status and mislabelling a MedMNIST+ sample as an upload.
  Provenance is now resolved inside the analysis callback by comparing against what is
  already held.
- **Layout.** The Inference left column no longer runs off the page; the reference gallery
  moved into a collapsed optional section.
- **Runaway downloads.** A fetch of a large set started a multi-gigabyte download behind a
  spinner indistinguishable from a hung server, on a serial queue that blocked the whole
  UI. Sizes are now surfaced before the download begins.

[1.0.0]: https://github.com/KeremKurban/medfm-lab/releases/tag/v1.0.0
