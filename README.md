# medfm Lab

A local workbench for **medical vision foundation models**: run inference, look at what the
model attends to, inspect patch embeddings, and read the residual stream layer by layer to
decide *where* a backbone should be adapted to a medical domain.

Everything runs on this Mac. No cloud calls, no uploads.

**v1.0.0** — validated end to end on `pneumoniamnist` (chest X-ray) and `retinamnist`
(fundus) with RAD-DINO on Apple Silicon / MPS. See [CHANGELOG.md](CHANGELOG.md).

---

## Why this exists

DINOv3-class backbones transfer to medical imaging surprisingly well out of the box — a
recent benchmark found DINOv3 beating domain-specific models like BiomedCLIP and CT-Net on
several tasks while trained only on natural images — but they degrade in specific places
(whole-slide pathology, electron microscopy, PET) and they do **not** follow clean scaling
laws in the medical domain. So the interesting questions are empirical and local:

- Where does this backbone already work, and where does it fall over?
- Which layers actually carry the label, and which are just doing generic vision?
- Are its attention maps trustworthy, or is a handful of high-norm register tokens
  absorbing all the global context?
- If adaptation is warranted, which parameter-efficient route and which blocks?

This tool exists to answer those questions with measurements rather than vibes.

---

## Install

```bash
cd ~/medfm-lab
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -r requirements.txt
```

Verified on: Apple M5, 32 GB, macOS 26.6, Metal 4 — torch 2.14, transformers 5.18,
timm 1.0.30, medmnist 3.0.2, gradio 6.29.

## Run

```bash
./run.sh                 # foreground, Ctrl-C to stop
./run.sh --detach        # background; survives closing the terminal
./run.sh --port 7861     # different port
```

Or directly, `.venv/bin/python app.py` → http://127.0.0.1:7860.

**Start it from your own terminal, not from an agent session.** A server launched as a
child of an agent is cleaned up when that session closes (SIGTERM, `agent_close`), which
presents to you as "connection to server was lost" in the middle of using the UI.

Stop it with:

```bash
kill $(lsof -t -nP -iTCP:7860 -sTCP:LISTEN)
```

Other entry points:

```bash
.venv/bin/python scripts/check_models.py          # health-check every ungated backbone
.venv/bin/python scripts/check_models.py --all    # include gated (licence) models
.venv/bin/python scripts/test_app.py              # headless exercise of every UI callback
```

---

## The five tabs

**1 · Inference** — load a backbone, pick a MedMNIST+ image or upload one, and get the
pooled embedding, patch grid, effective rank and anisotropy of the patch tokens, a PCA→RGB
pointillism map, and kNN retrieval against a labelled reference gallery you build on the
fly. `preprocessing = model` uses each backbone's own image processor; `uniform` forces one
shared resize and mean/std so cross-model comparisons are pixel-fair.

The tab is laid out as three numbered steps across the top (choose a backbone / choose an
image / analyse) with results below, so no single column runs off the page. Every tab opens
with a collapsed **"What can I do on this tab?"** guide, every control carries a tooltip, and
a live guidance line under the header always tells you the next action and what unlocked.

**2 · Attention** — cls→patch attention for any layer/head, heads pooled by mean/max/min,
attention rollout with an optional discard ratio, and a normalised-entropy heatmap over
layer × head. Entropy near 1.0 means the head is looking everywhere at once; near 0 means
it has collapsed onto one patch.

**3 · Patch tokens** — the DINO diagnostic stack: PCA→RGB, foreground masking, token UMAP
with k-means clusters, patch-similarity maps from a query patch, token-norm growth across
layers, and high-norm outlier token detection. There is a **positional debiasing** toggle:
DINOv3 features carry strong absolute-coordinate structure and similarity maps read much
better with the low-rank coordinate subspace projected out.

**4 · Residual stream** — layer-wise CKA, per-layer representation drift, norm profiles,
and the headline measurement: a **layer-wise linear probe**. If a linear model on frozen
features already reads the label, adaptation is wasted compute. If it does not, the probe
curve says which blocks to touch, and the tool prints a concrete adapter block range.

**5 · Models** — the catalogue with licences and gating status.

---

## MedMNIST+ download sizes — read before clicking fetch

MedMNIST+ upsamples every image from 28px to **224px**, which is what makes it usable by a
patch-14/16 vision transformer (at 28px the patch grid is 2×2, i.e. useless). The cost is
that the archives are enormous. These are exact Zenodo sizes, and the UI shows them in the
dropdown and warns before a first download:

| set | 224px | set | 224px |
|---|---|---|---|
| breastmnist | 31 MB | bloodmnist | 1.5 GB |
| retinamnist | 128 MB | organamnist | 1.8 GB |
| pneumoniamnist | 214 MB | tissuemnist | 3.4 GB |
| organcmnist | 760 MB | chestmnist | 3.9 GB |
| organsmnist | 803 MB | octmnist | 4.0 GB |
| dermamnist | 1.1 GB | pathmnist | **12.6 GB** |

Zenodo serves at roughly 4 MB/s, so `pathmnist` is about 50 minutes on a first run; it is
cached afterwards. A fetch that looks like a hang with a spinner is almost certainly a
download — the list is sorted smallest-first and every entry states its size.

**Truncated-cache trap.** medmnist caches purely by filename and torchvision writes straight
to the final path, so an interrupted download leaves a partial `.npz` that is accepted on
the next run and then fails inside `np.load` with a message that gives no clue why.
`medfm/data.py::ensure_intact()` compares the cached file against the known size on every
load and deletes anything short, so a killed download self-heals instead of corrupting.

## Model catalogue

| key | backbone | domain | notes |
|---|---|---|---|
| `dinov2-base` | DINOv2 ViT-B/14 | natural | ungated baseline |
| `dinov2-base-reg` | DINOv2 ViT-B/14 + registers | natural | the register-token control |
| `dinov3-vits16` / `vits16plus` / `vitb16` / `vitl16` | DINOv3 ViT-S/S+/B/L-16 | natural | gated on HF; auto-falls back to the ungated timm mirror |
| `dinov3-convnext-base` | DINOv3 ConvNeXt-B | natural | non-transformer control |
| `rad-dino` | DINOv2 ViT-B/14, chest X-ray | medical | Microsoft, ~800k CXR |
| `rad-dino-maira2` | as above, more data | medical | MSRLA licence |
| `meddinov3-vitb16` | DINOv3 ViT-B/16 on CT-3M | medical (CT) | Apache-2.0, raw `model.pth` |
| `cxformer-base` | DINOv2-based chest X-ray | medical | CC-BY-NC, `trust_remote_code` |

### Two loading quirks worth knowing

**Gated DINOv3.** The official `facebook/dinov3-*` repos require a one-time licence
acceptance while logged in. Without it the download returns 401/403. The loader detects
this and silently falls back to the ungated timm mirror (`timm/vit_*_dinov3.*`), so the
lab works either way; set `HF_TOKEN` and accept the licence if you want the official
checkpoints.

**MedDINOv3 is not a `transformers` model.** It ships a raw DINO reference-format
`model.pth` (fused `attn.qkv`, `ls1.gamma`/`ls2.gamma` layer scale, RoPE, 4 storage
tokens). `medfm/checkpoints.py` builds the matching timm architecture and remaps the keys
— 162/162 tensors, zero shape mismatches. The loader reports coverage and **refuses to
use** anything under 90% rather than quietly running a half-initialised model.

---

## Implementation notes that matter

- **Attention capture is dual-route.** Preferred: the model's own `output_attentions` with
  `attn_implementation="eager"`. Fallback: forward hooks. For timm this second route
  required disabling fused attention — timm can route through
  `F.scaled_dot_product_attention`, which never materialises the softmax matrix, so a hook
  on `attn_drop` captures *nothing* and you get a silent `None`. With fused attention off,
  the hook fires with the model's own RoPE-included attention.
- **Token prefix is derived, not declared.** `num_prefix = T - (H/patch)·(W/patch)`, so
  cls tokens, 4 register tokens and odd input sizes all resolve correctly without trusting
  per-model config flags.
- **Residual streams for timm.** timm exposes only final tokens; without help, layer-wise
  analysis is impossible for MedDINOv3. The encoder hooks every block and reconstructs
  `[block-0 input, block-1 out, …, block-L out]`, so HF and timm give comparable `L+1`
  layer stacks.
- **Memory.** Layer probes pool *inside* the extraction loop. Materialising token-level
  residual streams for a 200-image probe set would need gigabytes; pooled features are
  200×(L+1)×768 floats, i.e. megabytes.
- **MPS.** `PYTORCH_ENABLE_MPS_FALLBACK=1` is set, so unsupported ops fall back to CPU
  instead of crashing.

## Known caveats

- `cxformer-base` reports near-uniform attention entropy (~0.998) at every layer on
  synthetic images. Either its custom `modeling_*.py` attention behaves differently from
  stock DINOv2, or it needs its own preprocessing. Treat its attention tab as unverified
  until checked against a real radiograph.
- MedMNIST+ images are preprocessed and small. They are excellent for *cheap* comparative
  measurement across modalities, not for claiming clinical performance.
- Default input sizes are the registry's (224 for ViT-16 models, 518 for the CXR models).
  Change `image_size` in `medfm/registry.py` to probe resolution sensitivity — DINOv3's
  scaling behaviour in the medical domain is one of the open questions.

## UI conventions

- **Never a stack trace in the interface.** Errors render as one short line plus a hint;
  the full traceback is appended to `logs/errors.log`. A traceback leaks absolute paths,
  library versions and internal structure into screenshots, screen shares and bug reports.
- **Guidance over documentation.** Each tab opens with a collapsed *"What can I do on this
  tab?"* panel explaining every output, every control has an `info` tooltip, and a live line
  under the header states the next action and what it unlocks (`_hint()` in `app.py`).
- **Dropdowns use plain strings.** Gradio 6 validates a `Dropdown`'s `value` against a
  choice's *value* in Python but against its *name* in the frontend, so `(name, value)`
  tuples produce contradictory errors and an app that renders but never responds. Every
  dropdown here uses plain strings and parses the label back inside the callback —
  `registry.resolve_key` for backbones, `data.flag_from_option` for datasets. If you add a
  dropdown, follow that pattern; `scripts/test_app.py` will fail if you do not.

## Layout

```
medfm/
  registry.py     model catalogue: ids, patch size, mean/std, gating, licence
  encoders.py     uniform loader (HF + timm) -> pooled / patch tokens / residual / attention
  checkpoints.py  DINO reference-format .pth -> timm architecture, with key remap
  attention.py    capture hooks, rollout, per-head maps, entropy
  embeddings.py   PCA→RGB, UMAP, similarity, positional debiasing, token statistics
  residual.py     CKA, drift, norm profiles, layer-wise probes, adapter advice
  probe.py        frozen-feature linear/kNN probes and a cosine FeatureIndex
  data.py         MedMNIST+ at 224px, galleries, stratified probe sets
  viz.py          matplotlib/numpy figure helpers
app.py            the Gradio UI
scripts/
  check_models.py  load every model, run one image, assert sane attention + residual stats
  test_app.py      headless exercise of every UI callback
  smoke_test.py    environment check
```

## Licence

Code: MIT — see [LICENSE](LICENSE).

**The bundled code is not the whole story.** The model weights and datasets are downloaded
at runtime and carry their own terms, several of them non-commercial. Full detail in
[NOTICE.md](NOTICE.md).

| asset | licence |
|---|---|
| RAD-DINO / RAD-DINO-MAIRA-2 | Microsoft MSRLA (research use) |
| CXformer | CC-BY-NC-4.0 (non-commercial) |
| MedDINOv3 (CT-3M) | Apache-2.0 |
| DINOv2 / DINOv3 | Meta licence, gated on Hugging Face |
| MedMNIST+ | CC BY 4.0 (plus source-dataset terms) |

Check upstream terms before any use beyond research. Nothing here is validated for clinical
decision-making.

## Next: phase 2 — domain adaptation

The interpretability layer is deliberately built first, because it produces the evidence
that tells you *how* to adapt. Phase 2 slots in here:

1. LoRA / adapter modules injected at the block range the layer-wise probe recommends.
2. A DINOv3-FD-style dual-stream adapter (task-relevant vs task-irrelevant subspaces with
   an orthogonality loss) — a good fit for a single Mac, since only the adapter trains.
3. Baseline vs adapted comparison rendered in the same attention / embedding / residual
   views, so the effect of adaptation is visible rather than just tabulated.
4. Continual self-supervised pretraining on unlabelled slices if the probe curve says the
   frozen features are simply not there yet.
