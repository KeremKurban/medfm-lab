"""medfm Lab — a local UI for medical vision foundation models.

Run:  .venv/bin/python app.py          then open http://127.0.0.1:7860

Tabs
  1 Inference      pick a model, an image, get embeddings and kNN retrieval
  2 Attention      per-layer/head maps, rollout, entropy profile
  3 Patch tokens   PCA pointillism, UMAP, patch similarity, registers/outliers
  4 Residual       CKA, layer-wise linear probes, norm growth, adapter advice
  5 Models         registry, licences, loading status, gating notes
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

import gradio as gr

from medfm import data as mdata
from medfm import encoders, glossary as gl, guides, registry, viz
from medfm.attention import (attention_entropy, attention_rollout, head_map_mean,
                             layer_head_summary, mean_key_attention, per_head_cls_map)
from medfm.embeddings import (anisotropy, cluster_tokens, effective_rank, outlier_tokens,
                              pca_rgb, positional_debias, similarity_map, token_norms,
                              umap_2d)
from medfm.probe import FeatureIndex, fit_linear_probe, knn_probe
from medfm.residual import (adapt_layer_recommendation, cka_matrix_multi,
                            layer_probe_curve_pooled, representation_drift,
                            residual_norm_profile)

MODEL_CHOICES = registry.available_choices()          # plain label strings

# MedMNIST+ at 224px ranges from 31 MB (breastmnist) to 12.6 GB (pathmnist). Order by real
# download size and put the size in the label so nobody starts a multi-gigabyte download by
# accident and reads the spinner as a hung server.
DATASET_CHOICES = [
    mdata.option_label(flag)
    for flag, _ in sorted(mdata.MEDMNIST_2D.items(),
                          key=lambda kv: mdata.SIZE_224_GB.get(kv[0], 99.0))
]

# Gradio 6 validates a Dropdown's `value` inconsistently between its Python and frontend
# layers, so every dropdown here uses plain strings (name == value) and each callback
# normalises what it receives: registry.resolve_key for backbones, data.flag_from_option
# for datasets. Never reintroduce (name, value) tuples.
DEFAULT_MODEL = registry.label_for("rad-dino")
# Small and clinically legible: a 4 MB first download instead of pathmnist's 200 MB, so the
# very first click in a fresh install finishes in seconds.
DEFAULT_DATASET = next((d for d in DATASET_CHOICES if d.startswith("pneumoniamnist")),
                       DATASET_CHOICES[0])


# ------------------------------------------------------------------------- state

@dataclass
class LabState:
    encoder: Optional[encoders.Encoder] = None
    model_key: str = ""
    out: Optional[encoders.EncoderOutput] = None
    image: Optional[np.ndarray] = None
    dataset: str = ""
    label: Optional[int] = None
    gallery: Optional[FeatureIndex] = None
    cache: dict = field(default_factory=dict)
    image_status: str = ""
    sample_index: Optional[int] = None

    def clear_analysis(self):
        self.out = None
        self.cache = {}


LAB = LabState()


def _pil(image: np.ndarray):
    from PIL import Image

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    return Image.fromarray(arr)


LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "errors.log")


def _err(e: Exception, hint: str = "") -> str:
    """Log the full traceback to disk; return a short, safe message for the UI.

    Never render a stack trace in the interface: it leaks absolute paths, library versions
    and internal structure into screenshots, screen shares and bug reports.
    """
    import time

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}\n")

    msg = f"**Could not complete that step.**  \n`{type(e).__name__}: {e}`"
    if hint:
        msg += f"\n\n{hint}"
    msg += (f"\n\n<sub>Full details were written to `logs/errors.log` rather than shown "
            f"here, so nothing sensitive ends up in a screenshot.</sub>")
    return msg


def _hint(stage: str, spec=None) -> str:
    """A live 'what to do next' line. Keeps the UI self-teaching without a manual."""
    if stage == "start":
        return ("**Do this first →**  ① pick a backbone in the left column and press "
                "*Load model*  ·  ② fetch a MedMNIST+ sample (or upload an image) in the "
                "middle  ·  ③ press *Run inference & analyse*.")
    if stage == "no_model":
        return "**⚠ No backbone loaded yet.** Go to ① *Load model* first."
    if stage == "model_ready":
        patch = getattr(spec, "patch_size", 16)
        img = getattr(spec, "image_size", 224)
        return (f"**✓ Backbone ready.** Next → ② fetch a MedMNIST+ sample or upload an image "
                f"({img}px input, {patch}px patches).  \n"
                f"Afterwards ③ *Run inference & analyse* unlocks the other tabs.")
    if stage == "model_failed":
        return "**Backbone did not load.** Try another entry, or check `logs/errors.log`."
    if stage == "image_ready":
        return ("**✓ Image ready.** Next → ③ press **Run inference & analyse**.  \n"
                "On a medical backbone, try *RAD-DINO* with a chest X-ray set "
                "(`pneumoniamnist`) and *MedDINOv3* with a CT set (`organamnist`).")
    if stage == "analyzed":
        return ("**✓ Analysis done.** Explore further →  \n"
                "• **2 · Attention** — which patches the model looks at  \n"
                "• **3 · Patch tokens** — is the feature map anatomically meaningful?  \n"
                "• **4 · Residual stream** — which layers actually encode the label "
                "(and therefore where to adapt).  \n"
                "Optional: build a reference gallery in ③ to get kNN retrieval.")
    if stage == "analyzed_no_attn":
        return ("**✓ Analysis done** (attention was off, so tab 2 is empty). Tick "
                "*capture attention* in ③ and re-run, then open **2 · Attention**.")
    if stage == "gallery_ready":
        return ("**✓ Gallery built.** Re-run ③ *Run inference & analyse* to see nearest "
                "neighbours for the current image.")
    if stage == "no_analysis":
        return "**⚠ Nothing analysed yet.** Run ③ on the Inference tab first."
    return ""

def load_model(model_key: str, preprocess: str):
    try:
        LAB.clear_analysis()
        key = registry.resolve_key(model_key)
        enc = encoders.get_encoder(key, preprocess=preprocess)
        LAB.encoder = enc
        LAB.model_key = key
        spec = enc.spec
        ck = ""
        if enc.ckpt_report:
            ck = (f"  |  checkpoint: {enc.ckpt_report['n_loaded']}/"
                  f"{enc.ckpt_report['n_model_tensors']} tensors "
                  f"({enc.ckpt_report['coverage'] * 100:.1f}%)")
        msg = (
            f"**{spec.label}**  \n"
            f"backend `{enc.backend}`  |  device `{enc.device}`  |  "
            f"patch {spec.patch_size}px  |  dim {spec.embed_dim}  |  "
            f"blocks {spec.num_blocks}{ck}  \n"
            f"{spec.notes}"
        )
        guide = guides.model_guide_markdown(key)
        if guide:
            msg += f"\n\n{guide}"
        return msg, _hint("model_ready", spec)
    except Exception as e:
        return _err(e, "Pick a backbone from the dropdown, then press **Load model** "
                       "again."), _hint("model_failed")


def sample_index_in_gallery() -> bool:
    """True when the currently loaded sample is also present in the gallery."""
    return (LAB.sample_index is not None and LAB.gallery is not None
            and LAB.gallery.indices is not None
            and LAB.gallery.dataset == LAB.dataset
            and bool((LAB.gallery.indices == LAB.sample_index).any()))


def _abbr(term: str, label: str = None) -> str:
    """Term with a hover tooltip from the glossary."""
    tip = gl.SHORT.get(term, "")
    text = label if label is not None else term
    if not tip:
        return text
    safe = tip.replace('"', "&quot;")
    return (f'<abbr title="{safe}" style="border-bottom:1px dotted var(--muted-foreground, '
            f'#888);cursor:help;text-decoration:none">{text}</abbr>')


def _metrics_html(out, spec, pooled) -> str:
    """The numeric block, with every term carrying a hover definition.

    Rendered as HTML rather than Markdown purely so the <abbr> tooltips work — Markdown has
    no way to attach a definition to a term.
    """
    n_layers = out.hidden_states.shape[0]
    rows = [
        (_abbr("patch grid"), f"{out.grid[0]} × {out.grid[1]} = "
                              f"{out.grid[0] * out.grid[1]} patches"),
        (_abbr("patch size"), f"{spec.patch_size} px square"),
        ("input the model received", f"{out.input_size[0]} × {out.input_size[1]} px"),
        (_abbr("CLS token (and register tokens)", "prefix tokens"),
         f"{out.num_prefix} (1 cls" + (f" + {out.num_prefix - 1} registers"
                                       if out.num_prefix > 1 else " only") + ")"),
        ("embedding width", f"{out.patch_tokens.shape[1]} dims"),
        (_abbr("Residual stream", "residual stream"), f"{n_layers} layers captured"),
        (_abbr("embedding norm"), f"{np.linalg.norm(pooled):.2f}"),
        (_abbr("mean |value|"), f"{np.abs(pooled).mean():.4f}"),
        (_abbr("effective rank"),
         f"{effective_rank(out.patch_tokens):.1f} of {out.patch_tokens.shape[1]} dims"),
        (_abbr("anisotropy"), f"{anisotropy(out.patch_tokens):.3f}"),
    ]
    items = "".join(
        f'<div style="display:flex;gap:10px;padding:2px 0">'
        f'<span style="min-width:230px;opacity:.95">{k}</span>'
        f'<span style="font-variant-numeric:tabular-nums">{v}</span></div>'
        for k, v in rows
    )
    return (f'<div style="font-size:0.86rem;line-height:1.5">{items}</div>'
            f'<div style="font-size:0.78rem;opacity:.7;margin-top:6px">'
            f'Hover any dotted term for its definition · full glossary in tab 6</div>')


def _dataset_panel(dataset_flag: str) -> str:
    """Download notice plus the dataset's class guide."""
    return _dataset_notice(dataset_flag) + "\n\n" + _dataset_guide_body(dataset_flag)


def _dataset_guide_body(dataset_flag: str) -> str:
    try:
        flag = mdata.flag_from_option(dataset_flag)
    except Exception:
        return ""
    try:
        return guides.dataset_guide_markdown(flag)
    except Exception as e:
        return f"_guide unavailable: {e}_"


def _dataset_notice(dataset_flag: str) -> str:
    """Warn about the one-time download before the user commits to it."""
    try:
        flag = mdata.flag_from_option(dataset_flag)
        meta = mdata.dataset_meta(flag)
    except Exception:
        return ""
    gb = meta["size_224_gb"]
    if not gb:
        return ""
    if meta["cached"]:
        return f"✓ **{flag}** already cached locally ({gb:.2f} GB on disk) — loads instantly."
    if gb >= 1:
        return (f"⚠ **First run downloads {gb:.1f} GB** from Zenodo at roughly 4 MB/s — "
                f"expect ~{gb * 1024 / 4 / 60:.0f} minutes for this one set. It is cached "
                f"afterwards, and the size is not a mistake: MedMNIST+ upsamples every image "
                f"from 28px to 224px. Pick a smaller set above if you just want a quick run.")
    return (f"First run downloads {gb * 1000:.0f} MB from Zenodo — cached afterwards.")


def fetch_sample(dataset_flag: str, index: int):
    try:
        flag = mdata.flag_from_option(dataset_flag)
        samples = mdata.fetch_samples(flag, split="test", n=1, offset=int(index), size=224)
        if not samples:
            return None, "No sample at that index.", _hint("start")
        s = samples[0]
        LAB.image = s.image
        LAB.dataset = flag
        LAB.label = s.label_scalar
        LAB.sample_index = int(s.index)
        name = mdata.label_name(flag, s.label_scalar)
        status = (f"**{flag}** test[{s.index}] · label {s.label_scalar} ({name}) · "
                  f"{s.image.shape[1]}x{s.image.shape[0]}px")
        LAB.image_status = status
        return s.image, status, _hint("image_ready")
    except Exception as e:
        return (None,
                _err(e, "If this set has never been used, the first fetch downloads it from "
                        "Zenodo — check the connection and try again."),
                _hint("start"))


def received_image(image):
    """Unused hook kept for clarity: see analyze(), which now owns image provenance.

    Gradio re-emits a change event when we set the image ourselves from a MedMNIST+ sample,
    and it fires *after* the fetch callback, so any handler here would clobber the dataset
    status. Provenance is therefore resolved inside analyze() by comparing against what we
    already hold.
    """
    raise NotImplementedError


def analyze(image, capture_attn: bool, top_k: int):
    try:
        if LAB.encoder is None:
            return ("**No backbone loaded.** Go to ① in the left column, choose a "
                    "backbone, press **Load model**.", None, None, None, None, None,
                    _hint("no_model"), gr.update())
        if image is None and LAB.image is None:
            return ("**No image yet.** Use ② to fetch a MedMNIST+ sample or upload one.",
                    None, None, None, None, None, _hint("start"), gr.update())

        img = np.asarray(image if image is not None else LAB.image)
        # Single source of truth for provenance: if the component holds something different
        # from what we last saw, the user uploaded it. Comparing here instead of in a change
        # handler avoids the race where the change event fires after the fetch callback.
        current = LAB.image
        same = (current is not None and img.shape == current.shape
                and np.array_equal(img, current))
        if not same:
            LAB.image = img
            LAB.dataset = "upload"
            LAB.label = None
            LAB.sample_index = None
            LAB.image_status = (f"Uploaded image · {img.shape[1]}x{img.shape[0]}px "
                                f"(no ground-truth label)")
        LAB.image = img

        out = LAB.encoder(img, capture_attention=capture_attn)
        LAB.out = out
        LAB.cache = {}

        spec = LAB.encoder.spec
        pooled = out.pooled
        seen = LAB.encoder.preview(img)          # what the network actually received
        h, w = out.grid

        # Ground truth, so there is always something to check the pictures against.
        gt_txt = "none (uploaded image)"
        if LAB.label is not None:
            gt_txt = f"**{LAB.label}** ({mdata.label_name(LAB.dataset, LAB.label)})"

        summary = [
            f"### Inference",
            f"- image: {LAB.image_status or LAB.dataset}",
            f"- **ground truth: {gt_txt}**",
            f"- model: `{spec.key}` via `{LAB.encoder.backend}` on `{LAB.encoder.device}`",
            f"- the model received {out.input_size[0]}x{out.input_size[1]} and cut it into a "
            f"**{h}x{w} grid of {out.patch_tokens.shape[0]} patches** — see the left panel",
        ]
        metrics = _metrics_html(out, spec, pooled)

        rgb, mask, info = pca_rgb(out.patch_tokens, out.grid)
        LAB.cache["pca"] = (rgb, mask, info)
        LAB.cache["seen"] = seen

        ret_strip = None
        if LAB.gallery is not None and LAB.gallery.size:
            # never let the query match itself: a self-hit scores 1.0 and always agrees with
            # the label, which makes retrieval look perfect while measuring nothing
            exclude = None
            if (LAB.sample_index is not None
                    and LAB.gallery.dataset == LAB.dataset
                    and LAB.gallery.indices is not None):
                exclude = LAB.sample_index
            hits = LAB.gallery.query(pooled, k=int(top_k), exclude_index=exclude)
            imgs = [h_["image"] for h_ in hits if h_["image"] is not None]
            labs = [h_["label"] for h_ in hits if h_["image"] is not None]
            sims = [h_["similarity"] for h_ in hits if h_["image"] is not None]
            if imgs:
                top1 = hits[0]
                verdict = ""
                if LAB.label is not None:
                    agree = int(top1["label"]) == int(LAB.label)
                    verdict = (f" · nearest neighbour is a **{'MATCH' if agree else 'MISMATCH'}** "
                               f"for the ground truth")
                ret_strip = viz.gallery_strip(
                    imgs, labs, sims,
                    title=f"nearest neighbours in the {LAB.gallery.dataset} gallery "
                          f"(model {LAB.gallery.model_key})"
                          + (" · query excluded from the gallery" if exclude is not None
                             else ""))
                hist = LAB.gallery.label_histogram(pooled, k=int(top_k),
                                                   exclude_index=exclude)
                summary.append(
                    f"\n**kNN retrieval** — the {len(imgs)} reference images whose embeddings "
                    f"are closest to this one by cosine similarity. It answers \"is this "
                    f"backbone's representation of this image like other images of the same "
                    f"kind?\" If the neighbours share the label, the features are already "
                    f"clinically organised; if they are random, the backbone does not "
                    f"represent this modality well.{verdict}")
                summary.append(f"- top-1 neighbour: label {top1['label']} at "
                               f"similarity {top1['similarity']:.3f}")
                summary.append(f"- weighted label votes: " +
                               ", ".join(f"{k}:{v:.2f}" for k, v in list(hist.items())[:6]))

        summary.append(
            f"\n**Reading the three panels** (all shown at the same size so they line up "
            f"patch for patch):  \n"
            f"**left — what the model saw.** {out.input_size[0]}x{out.input_size[1]}, "
            f"resized and intensity-normalised exactly as the network received it. Your "
            f"upload was {img.shape[1]}x{img.shape[0]}.  \n"
            f"**middle — patch PCA.** Each square is one {spec.patch_size}x{spec.patch_size}px "
            f"patch, coloured by the first three principal components of its 768-dim feature "
            f"vector. It is genuinely a {h}x{w} image, one pixel per patch, which is why it is "
            f"drawn enlarged with hard edges. Patches that share a colour are represented "
            f"similarly by the model — so if the anatomy separates into coherent coloured "
            f"regions, the backbone sees structure; if it is confetti, it does not.  \n"
            f"**right — PC1 foreground mask.** The first principal component plotted as a "
            f"brightness map and thresholded. This is the DINO papers' own trick: these "
            f"backbones tend to encode \"subject versus background\" in PC1, so a threshold "
            f"on it acts as a free segmentation. Everything tinted green is what the model "
            f"considers foreground, without ever having been trained to segment anything."
        )
        summary.append(f"\n- attention captured: "
                       f"{'yes' if out.attentions is not None else 'no'}")

        hint = _hint("analyzed" if out.attentions is not None else "analyzed_no_attn")
        return ("\n".join(summary), metrics, seen,
                viz.upscale(rgb, size=seen.shape[0], nearest=True),
                viz.mask_overlay(seen, mask), ret_strip, hint,
                LAB.image_status or "_image ready_")
    except Exception as e:
        return (_err(e, "The backbone ran but something downstream failed. Try re-loading "
                        "the model."), None, None, None, None, None, _hint("start"),
                gr.update())


def build_gallery(dataset_flag: str, n: int, pool: str, progress=gr.Progress()):
    try:
        if LAB.encoder is None:
            return "**No backbone loaded.** Load one in ① first.", _hint("no_model")
        flag = mdata.flag_from_option(dataset_flag)
        progress(0.02, desc="downloading / loading MedMNIST+")
        samples = mdata.build_gallery(flag, n=int(n), size=224)
        imgs = [s.image for s in samples]
        labels = np.array([s.label_scalar for s in samples])
        progress(0.1, desc=f"embedding {len(imgs)} images")

        def cb(frac, desc=""):
            progress(0.1 + 0.85 * frac, desc=desc)

        # embed one by one so we can report progress
        vecs = []
        for i, im in enumerate(imgs):
            vecs.append(LAB.encoder.embed([im], pool=pool)[0])
            cb((i + 1) / len(imgs), f"embedding {i + 1}/{len(imgs)}")
        vecs = np.stack(vecs)

        LAB.gallery = FeatureIndex().build(vecs, labels, imgs, dataset=flag,
                                           model_key=LAB.model_key,
                                           indices=np.array([s.index for s in samples]))
        counts = np.bincount(labels.astype(int))
        return (f"**✓ {LAB.gallery.size} reference images** from `{flag}` "
                f"({pool}-pooled, model `{LAB.model_key}`).  \n"
                f"class counts: {dict(enumerate(counts.tolist()))}",
                _hint("gallery_ready"))
    except Exception as e:
        return _err(e, "Building a large gallery takes a while — try a smaller size."), \
            _hint("start")


# --------------------------------------------------------------------- tab 2

def render_attention(layer: int, head: int, mode: str, discard: float, cmap: str):
    _empty = ("**Nothing to show yet.** Go to **1 · Inference**, tick *capture attention*, "
              "press **Run inference & analyse**, then come back here.")
    try:
        if LAB.out is None or LAB.out.attentions is None:
            return _empty, None, None, None, None, None
        out = LAB.out
        a = torch.from_numpy(out.attentions)
        L, H, T, _ = a.shape
        layer = int(np.clip(layer, 0, L - 1))
        head = int(np.clip(head, 0, H - 1))
        grid = out.grid
        npfx = out.num_prefix

        per_head = per_head_cls_map(a, layer, head, grid, num_prefix=npfx)
        pooled_heads = head_map_mean(a, layer, grid, num_prefix=npfx, mode=mode)
        roll = attention_rollout(a, discard_ratio=float(discard)).numpy()[npfx:]
        roll_map = roll[: grid[0] * grid[1]].reshape(grid)

        summ = layer_head_summary(a, num_prefix=npfx)
        ent_plot = viz.line_plot(
            {"entropy (mean over heads)": summ["entropy_layer_mean"],
             "top-1 attention mass": summ["top1_layer_mean"]},
            title="attention concentration per layer",
            ylabel="normalised entropy / mass", ylim=(0, 1),
        )
        ent_plot2 = viz.heatmap_plot(
            summ["entropy"], title="cls->patch attention entropy (layer x head)",
            xtick_labels=[f"h{i}" for i in range(H)],
            ytick_labels=[f"L{i}" for i in range(L)], cmap="magma", figsize=(7.5, 4.0),
        )

        msg = (f"Layer {layer} / head {head} — cls token attention over the "
               f"{grid[0]}x{grid[1]} patch grid.  \n"
               f"entropy here: {summ['entropy'][layer, head]:.3f} "
               f"(1.0 = uniform, 0 = single patch).  \n"
               f"rollout uses {mode}-fusion over heads with discard ratio {discard}.")
        return (msg,
                viz.overlay_heatmap(LAB.image, per_head, alpha=0.55, cmap=cmap),
                viz.overlay_heatmap(LAB.image, pooled_heads, alpha=0.55, cmap=cmap),
                viz.overlay_heatmap(LAB.image, roll_map, alpha=0.55, cmap=cmap),
                ent_plot, ent_plot2)
    except Exception as e:
        return _err(e), None, None, None, None, None


# --------------------------------------------------------------------- tab 3

def render_embeddings(n_clusters: int, patch_row: int, patch_col: int,
                      debias: bool, mask_q: float):
    try:
        if LAB.out is None:
            return ("**Nothing to show yet.** Run **1 · Inference → Run inference & "
                    "analyse** first.", None, None, None, None, None, None, None)
        out = LAB.out
        grid = out.grid
        tokens = out.patch_tokens

        rgb, mask, info = pca_rgb(tokens, grid, mask_quantile=float(mask_q))
        pca_rgb_img = viz.heat_image(rgb)
        fg_img = viz.mask_overlay(LAB.image, mask)

        um = umap_2d(tokens)
        cl = cluster_tokens(tokens, n_clusters=int(n_clusters))
        umap_fig = viz.scatter_plot(um, color=cl, title="patch token UMAP, coloured by k-means cluster")

        row = int(np.clip(patch_row, 0, grid[0] - 1))
        col = int(np.clip(patch_col, 0, grid[1] - 1))
        idx = row * grid[1] + col
        use_tokens = positional_debias(tokens, grid) if debias else tokens
        sim = similarity_map(use_tokens, idx, grid)
        sim_overlay = viz.overlay_heatmap(LAB.image, sim, alpha=0.6)
        marker = LAB.image.copy()
        y0, y1 = int(row * marker.shape[0] / grid[0]), int((row + 1) * marker.shape[0] / grid[0])
        x0, x1 = int(col * marker.shape[1] / grid[1]), int((col + 1) * marker.shape[1] / grid[1])
        marker[y0:y1, x0:x1] = [255, 0, 0]

        norms = token_norms(out.hidden_states)
        ot_all = outlier_tokens(out.hidden_states, layer=-1)
        ot = outlier_tokens(out.hidden_states, layer=-1, num_prefix=out.num_prefix)
        norm_fig = viz.line_plot(
            {"patch norm (mean)": norms[:, out.num_prefix:].mean(axis=1),
             "patch norm (max)": norms[:, out.num_prefix:].max(axis=1),
             "cls norm": norms[:, 0]},
            title="token norm growth across the residual stream",
            ylabel="L2 norm",
        )
        out_fig = viz.scatter_plot(um, color=ot["z"],
                                   title="token UMAP coloured by last-layer norm z-score "
                                         "(patch tokens)",
                                   colorbar_label="norm z", cmap="plasma")

        msg = (f"PCA explains {info['explained_variance_ratio'][0]:.1%} + "
               f"{info['explained_variance_ratio'][1]:.1%} + "
               f"{info['explained_variance_ratio'][2]:.1%} of patch variance.  \n"
               f"foreground mask keeps {mask.mean():.1%} of the grid.  \n"
               f"similarity map queried at patch ({row}, {col})"
               f"{' after positional debiasing' if debias else ''}.  \n"
               f"high-norm tokens at the last layer: **{ot['n_outliers']}** among the "
               f"{len(ot['z'])} patch tokens (max z = {ot['max_z']:.1f}), "
               f"**{ot_all['n_outliers']}** counting cls and registers too.  \n"
               f"A few very high-norm patches act as global-context registers, and "
               f"attention maps that look like they ignore the anatomy are usually that.")
        return msg, pca_rgb_img, fg_img, sim_overlay, marker, umap_fig, norm_fig, out_fig
    except Exception as e:
        return _err(e), None, None, None, None, None, None, None


# --------------------------------------------------------------------- tab 4

def render_residual(progress=gr.Progress()):
    try:
        if LAB.out is None or LAB.encoder is None:
            return ("**Nothing to show yet.** Run **1 · Inference → Run inference & "
                    "analyse** first.", None, None, None)
        out = LAB.out
        H = out.hidden_states
        labels = ["embed"] + [f"b{i}" for i in range(1, H.shape[0])]

        M = cka_matrix_multi([H], num_prefix=out.num_prefix)
        cka_fig = viz.heatmap_plot(M, title="layer-wise CKA (single image, patch tokens)",
                                   xtick_labels=labels, ytick_labels=labels,
                                   cmap="viridis", figsize=(6.2, 5.2), vmin=0, vmax=1)

        drift = representation_drift(H, num_prefix=out.num_prefix)
        prof = residual_norm_profile(H, num_prefix=out.num_prefix)
        curves = viz.line_plot(
            {"cls norm": prof["cls_norms"], "patch norm (mean)": prof["patch_norm_mean"]},
            title="residual stream norms", ylabel="L2 norm",
        )
        drift_fig = None
        if drift.size:
            drift_fig = viz.line_plot({"cosine(layer l, l+1)": drift},
                                      title="representation drift between consecutive layers",
                                      ylabel="cosine similarity", ylim=(0, 1))

        msg = (f"Residual stream: **{H.shape[0]}** layers x {H.shape[1]} tokens x "
               f"{H.shape[2]} dims.  \n"
               f"cls norm grows {prof['cls_norms'][0]:.2f} -> {prof['cls_norms'][-1]:.2f}.  \n"
               f"Patch norms climb to a mean of {prof['patch_norm_mean'][-1]:.1f} "
               f"(max {prof['patch_norm_max'][-1]:.1f}), and the gap between mean and max "
               f"is the signature of a few tokens absorbing global context.  \n"
               f"To make this a real adaptation decision rather than a guess, run the "
               f"layer-wise probe below — it tells you which blocks actually encode the "
               f"label.")
        return msg, cka_fig, curves, drift_fig
    except Exception as e:
        return _err(e), None, None, None


def render_cka_multi(dataset_flag: str, n_images: int, progress=gr.Progress()):
    try:
        if LAB.encoder is None:
            return "**No backbone loaded.** Load one on **1 · Inference** first.", None
        progress(0.05, desc="loading samples")
        flag = mdata.flag_from_option(dataset_flag)
        samples = mdata.fetch_samples(flag, split="test", n=int(n_images), size=224)
        hidden = []
        for i, s in enumerate(samples):
            o = LAB.encoder(s.image, capture_attention=False)
            hidden.append(o.hidden_states)
            progress(0.1 + 0.85 * (i + 1) / len(samples), desc=f"{i + 1}/{len(samples)}")
        M = cka_matrix_multi(hidden, num_prefix=LAB.out.num_prefix if LAB.out else 1)
        labels = ["embed"] + [f"b{i}" for i in range(1, M.shape[0])]
        fig = viz.heatmap_plot(M, title=f"layer-wise CKA over {len(samples)} {flag} images",
                               xtick_labels=labels, ytick_labels=labels, cmap="viridis",
                               figsize=(6.2, 5.2), vmin=0, vmax=1)
        return f"CKA over {len(samples)} images.", fig
    except Exception as e:
        return _err(e), None


def run_layer_probe(dataset_flag: str, n_per_class: int, pool: str,
                    progress=gr.Progress()):
    try:
        if LAB.encoder is None:
            return ("**No backbone loaded.** Load one on **1 · Inference** first.",
                    None, None)
        progress(0.02, desc="loading probe set")
        flag = mdata.flag_from_option(dataset_flag)
        samples = mdata.build_probe_set(flag, n_per_class=int(n_per_class), size=224)
        if len(samples) < 10:
            return f"probe set too small ({len(samples)})", None, None
        X = LAB.encoder.layer_features(
            [s.image for s in samples], pool=pool,
            progress=lambda f, desc="": progress(0.05 + 0.85 * f, desc=desc),
        )
        y = np.array([s.label_scalar for s in samples])

        curve = layer_probe_curve_pooled(X, y)
        drift = representation_drift(LAB.out.hidden_states, num_prefix=LAB.out.num_prefix) \
            if LAB.out is not None else None
        rec = adapt_layer_recommendation(curve, drift)

        fig = viz.line_plot(
            {"linear probe accuracy": curve["accuracy"]},
            title=f"layer-wise linear probe — {flag} ({len(samples)} images, {pool}-pooled)",
            ylabel="accuracy", xtick_labels=curve["layer_labels"],
            ylim=(0, 1), hline=1.0 / max(len(np.unique(y)), 2),
        )

        # layer feature norms for the same set
        norms = np.linalg.norm(X, axis=-1).mean(axis=0)
        nfig = viz.line_plot({"mean feature norm": norms},
                             title="mean pooled feature norm per layer", ylabel="L2 norm")

        txt = (f"### Layer-wise probe\n"
               f"- probe set: {len(samples)} images, {len(np.unique(y))} classes, "
               f"{pool}-pooled features\n"
               f"- best layer: **{curve['layer_labels'][curve['best_layer']]}** "
               f"at **{curve['best_accuracy']:.3f}** accuracy "
               f"(chance = {1.0 / max(len(np.unique(y)), 2):.3f})\n"
               f"- per-layer accuracies: "
               f"{', '.join(f'{a:.3f}' for a in curve['accuracy'])}\n\n"
               f"### Adaptation advice\n{rec['advice']}\n\n"
               f"Suggested adapter blocks: **{rec['suggested_start_block']}–"
               f"{rec['suggested_end_block']}**"
               + (f", most-changed block {rec.get('most_changed_block')}"
                  if rec.get("most_changed_block") else ""))
        return txt, fig, nfig
    except Exception as e:
        return _err(e), None, None


# --------------------------------------------------------------------- tab 5

def registry_table():
    rows = []
    for s in registry.MODELS.values():
        rows.append([
            s.key, s.label, s.family, s.domain, s.modality,
            f"{s.patch_size}px", s.embed_dim, s.num_blocks, s.image_size,
            "yes" if s.gated else "no",
            "ckpt" if s.ckpt else ("hf" if s.hf_id else "timm"),
        ])
    return rows


REGISTRY_HEADERS = ["key", "label", "family", "domain", "modality", "patch", "dim",
                    "blocks", "img", "gated", "source"]


# ----------------------------------------------------------------------- build

GUIDE_INFERENCE = """
**This tab is the entry point — nothing on the other tabs works until you do step ③ here.**

1. **Pick a backbone** and press *Load model*. Medical backbones are listed first.
   *Choose a medical one to match your image:* RAD-DINO for chest X-ray, MedDINOv3 for CT.
2. **Pick an image** — fetch a MedMNIST+ sample, or drop in your own.
   *Check the download size under the dropdown first.* MedMNIST+ upsamples every image to
   224px, so a set like `pathmnist` is a **12.6 GB** download the first time; it is cached
   afterwards. The list is sorted smallest first — `breastmnist` is 31 MB,
   `pneumoniamnist` is 214 MB. If a fetch seems to hang with a spinner, it is almost
   certainly downloading.
3. **Run inference & analyse** — this computes everything the other tabs display.

*Then explore:* tab 2 for attention, tab 3 for patch embeddings, tab 4 for the residual stream.

**What the four output panels mean** (①②③ are all drawn at the same size so they line up
patch for patch)

- **① what the model saw** — your image after the backbone's own resize and intensity
  normalisation. Your upload was 224px; this could be 518px. Always check this first, because
  everything downstream describes *this* image, not the file on your disk.
- **② patch PCA → RGB** — each square is one patch (14px or 16px), coloured by the first three
  principal components of its feature vector. It is a genuinely tiny image — one pixel per
  patch, so a 37×37 grid — which is why it is drawn enlarged with hard edges. Patches sharing
  a colour are represented similarly by the model. Coherent coloured anatomy = the backbone
  sees structure; confetti = it does not.
- **③ PC1 foreground mask** — the first principal component as a brightness map, thresholded.
  The DINO papers' own trick: these backbones usually encode "subject vs background" in PC1,
  so a threshold on it behaves like a free segmentation. Nothing was ever trained to segment.
- **④ kNN retrieval** — see the box below.

**Is there a ground truth?** Yes, whenever you fetch a MedMNIST+ sample: the analysis prints
its label and name at the top, and the nearest-neighbour line tells you whether retrieval
agrees with it. For an uploaded image there is no label, so you are reading the maps
qualitatively — which is exactly what ①–③ are for.

**What kNN retrieval is for.** Embed a labelled reference set once (the optional gallery
below), then for any image find the reference images whose embeddings are closest by cosine
similarity. It is a direct test of whether the backbone's representation is already
clinically organised: if the nearest neighbours are the same class, a linear head will work
and adaptation is unnecessary. If they are random, the features do not separate this
modality — which is the case for domain adaptation. It is the qualitative companion to the
layer-wise linear probe on tab 4: retrieval shows you *examples*, the probe gives you a
number.
"""

GUIDE_ATTENTION = """
**Every panel here is the cls token's attention spread over the patch grid.** The cls token
is the summary token; where it points is a decent proxy for what the model considers relevant.

- **single head** — one attention head at one layer. Individual heads are often specialised.
- **heads pooled** — mean/max/min across that layer's heads. This is what most papers show.
- **attention rollout** — attention accumulated through *all* layers, which is usually far
  more interpretable than any single layer. The discard ratio zeroes out the weakest
  attentions before propagating, which sharpens it.
- **entropy curve / heatmap** — normalised entropy of the cls→patch distribution. **1.0 means
  the head spreads attention perfectly uniformly (uninformative); 0 means it fired on one
  patch.** Look for layers where entropy collapses — those are doing selection.

*Reading tip:* early layers are usually high-entropy and late layers low. If a model's
rollout map ignores the anatomy entirely, check tab 3 for high-norm outlier tokens before
concluding the model is broken.
"""

GUIDE_TOKENS = """
**The diagnostic layer.** This is where you find out whether a feature map is trustworthy.

- **PCA → RGB** — same as tab 1, at full detail.
- **foreground mask** — the PC1 threshold. Lower the quantile to keep more area.
- **patch similarity** — cosine similarity from one chosen patch to all others. Red approves
  the query patch's location. This is how you test whether "similar content" maps to similar
  features.
- **query patch location** — the red square shows where you queried, so you can sanity-check
  the similarity map.
- **token UMAP** — patch tokens projected to 2-D and coloured by k-means cluster. Distinct
  colours in distinct anatomical regions = good features.
- **token norms per layer** — L2 norm of every token through the residual stream. Watch for
  a few patch tokens exploding in norm.
- **norm outliers** — the same UMAP coloured by last-layer norm z-score.

**Two things that bite everyone:**
1. **A handful of patch tokens absorb the global context** and get enormous norms — they act
   as registers. They are not anatomy, and they distort attention maps. Count them here.
2. **DINOv3 features carry absolute-position bias.** A similarity map can look like a cross
   centred on your query patch purely because of where the patch *is*, not what it contains.
   Tick **positional debiasing** to project out the coordinate subspace and compare.
"""

GUIDE_RESIDUAL = """
**This is the tab that tells you whether to bother with domain adaptation, and where.**

- **layer CKA** — how similar the representation is between every pair of layers. Bright
  off-diagonal blocks = layers that change nothing. A sharp transition = the model
  reorganising.
- **norm profile** — token norms across layers.
- **representation drift** — cosine similarity between consecutive layers. Low = that block
  rewrote the representation.

**The headline measurement is the layer-wise linear probe.** It trains a plain logistic
regression on the frozen features at *every* layer and reports accuracy:

- **If a late layer is already near 100%**, the frozen features contain the label — adapt
  nothing, just fit a classifier head.
- **If accuracy peaks mid-network and then falls**, the late layers have specialised away
  from your domain. That is the classic signature of a backbone from the wrong modality, and
  it is exactly when adaptation pays.
- **The tool prints a suggested adapter block range** from that curve, so phase-2 LoRA work
  has a concrete target instead of a guess.

Probing is slow because it extracts features for a whole dataset — use *images per class* to
control the cost.
"""

GUIDE_MODELS = """
**Where each backbone comes from, and what to watch out for.**

- `gated = yes` means the Hugging Face repo needs a one-time licence acceptance while logged
  in. Where an ungated timm mirror exists, the loader quietly falls back to it, so the lab
  works either way.
- **MedDINOv3** is not a `transformers` model. It ships a raw DINO reference-format
  `model.pth` (fused qkv, layer scale, RoPE, 4 storage tokens). The loader builds the matching
  timm architecture and remaps the keys — 162/162 tensors, zero shape mismatches — and
  refuses anything under 90% coverage rather than silently running a half-initialised model.
- **CXformer** ships custom modelling code and is loaded with `trust_remote_code=True`.
  It is CC-BY-NC and research-only.
- **Preprocessing**: `model` uses each backbone's own image processor (correct); `uniform`
  forces one shared resize and mean/std. Use `uniform` when you are comparing backbones and
  want the only difference to be the weights.
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="medfm Lab") as demo:
        gr.Markdown(
            "# medfm Lab\n"
            "Local inference, attention and residual-stream analysis for medical vision "
            "foundation models — everything runs on this Mac, no cloud calls."
        )

        with gr.Tab("1 · Inference"):
            with gr.Accordion("❔ What can I do on this tab? (click to open)", open=False):
                gr.Markdown(GUIDE_INFERENCE)

            next_hint = gr.Markdown(_hint("start"))

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=250):
                    gr.Markdown("#### ① Choose a backbone")
                    model_dd = gr.Dropdown(
                        choices=MODEL_CHOICES, value=DEFAULT_MODEL, label="backbone",
                        info="Medical models are listed first. Match the backbone to your "
                             "modality for meaningful results.")
                    preproc = gr.Radio(
                        choices=["model", "uniform"], value="model", label="preprocessing",
                        info="model = the backbone's own image processor (correct). "
                             "uniform = one shared resize + mean/std, for fair cross-model "
                             "comparison.")
                    load_btn = gr.Button("Load model", variant="primary")
                    model_status = gr.Markdown("_no model loaded yet_")

                with gr.Column(scale=1, min_width=250):
                    gr.Markdown("#### ② Choose an image")
                    ds_dd = gr.Dropdown(
                        choices=DATASET_CHOICES, value=DEFAULT_DATASET,
                        label="MedMNIST+ set",
                        info="12 public biomedical sets at 224px across CT, OCT, X-ray, "
                             "pathology, ultrasound, fundus and blood smear.")
                    index_sl = gr.Slider(0, 100, value=0, step=1, label="test index",
                                         info="Which image from the test split to load.")
                    ds_notice = gr.Markdown(_dataset_notice(DEFAULT_DATASET))
                    with gr.Accordion("📖 What is this dataset? (classes and what each means)",
                                      open=False):
                        ds_guide = gr.Markdown(_dataset_guide_body(DEFAULT_DATASET))
                    sample_btn = gr.Button("Fetch MedMNIST+ sample")
                    up = gr.Image(type="numpy", label="…or upload your own image",
                                  height=180)
                    img_status = gr.Markdown("_no image yet_")

                with gr.Column(scale=1, min_width=250):
                    gr.Markdown("#### ③ Analyse")
                    capt = gr.Checkbox(
                        value=True, label="capture attention",
                        info="Required for tab 2. Costs a little speed and memory; turn it "
                             "off if you only want embeddings.")
                    topk = gr.Slider(1, 16, value=8, step=1, label="kNN neighbours",
                                     info="How many nearest reference images to retrieve, "
                                          "once a gallery exists.")
                    analyze_btn = gr.Button("Run inference & analyse", variant="primary")

                    with gr.Accordion("Optional · reference gallery for kNN retrieval",
                                      open=False):
                        gr.Markdown(
                            "Embeds a labelled set of images so the analysis can show you "
                            "the nearest neighbours of your image. Useful for spotting "
                            "whether a backbone clusters by class."
                        )
                        gal_ds = gr.Dropdown(choices=DATASET_CHOICES,
                                             value=DEFAULT_DATASET, label="gallery set")
                        gal_n = gr.Slider(16, 400, value=96, step=8, label="gallery size")
                        gal_pool = gr.Radio(choices=["cls", "mean"], value="cls",
                                            label="pooling")
                        gal_btn = gr.Button("Build gallery")
                        gal_status = gr.Markdown("_gallery empty_")

            gr.Markdown("---")
            gr.Markdown("### Results")
            summary = gr.Markdown("_run step ③ to see results_")
            metrics = gr.HTML("")
            with gr.Row():
                seen_out = gr.Image(label="① what the model saw", type="numpy", height=330)
                pca_out = gr.Image(label="② patch PCA → RGB  (one pixel per patch)",
                                   type="numpy", height=330)
                fg_out = gr.Image(label="③ PC1 foreground mask", type="numpy", height=330)
            ret = gr.Image(label="④ kNN retrieval against the reference gallery",
                           type="numpy")

            load_btn.click(load_model, [model_dd, preproc], [model_status, next_hint])
            ds_dd.change(_dataset_notice, [ds_dd], [ds_notice])
            ds_dd.change(_dataset_guide_body, [ds_dd], [ds_guide])
            sample_btn.click(fetch_sample, [ds_dd, index_sl], [up, img_status, next_hint])
            analyze_btn.click(analyze, [up, capt, topk],
                              [summary, metrics, seen_out, pca_out, fg_out, ret, next_hint,
                               img_status])
            gal_btn.click(build_gallery, [gal_ds, gal_n, gal_pool], [gal_status, next_hint])

        with gr.Tab("2 · Attention"):
            with gr.Accordion("❔ What can I do on this tab? (click to open)", open=False):
                gr.Markdown(GUIDE_ATTENTION)
            with gr.Row():
                layer_sl = gr.Slider(0, 11, value=11, step=1, label="layer",
                                     info="Which transformer block to inspect. Late layers "
                                          "are usually the most selective.")
                head_sl = gr.Slider(0, 11, value=0, step=1, label="head",
                                    info="Individual heads often specialise; sweep this to "
                                         "find the interpretable ones.")
                fusion = gr.Radio(choices=["mean", "max", "min"], value="mean",
                                  label="head fusion",
                                  info="How to combine the heads of one layer.")
                discard = gr.Slider(0.0, 0.9, value=0.0, step=0.05,
                                    label="rollout discard ratio",
                                    info="Zeroes the weakest attentions before propagating "
                                         "through layers. Higher = sharper map.")
                cmap_dd = gr.Dropdown(
                    choices=["turbo", "magma", "viridis", "jet", "coolwarm"],
                    value="turbo", label="colormap")
            attn_btn = gr.Button("Render attention", variant="primary")
            attn_msg = gr.Markdown("")
            with gr.Row():
                a_single = gr.Image(label="single head", type="numpy")
                a_pool = gr.Image(label="heads pooled", type="numpy")
                a_roll = gr.Image(label="attention rollout (all layers)", type="numpy")
            with gr.Row():
                ent_curve = gr.Image(label="entropy / concentration per layer", type="numpy")
                ent_heat = gr.Image(label="entropy heatmap (layer × head)", type="numpy")

            attn_btn.click(render_attention,
                           [layer_sl, head_sl, fusion, discard, cmap_dd],
                           [attn_msg, a_single, a_pool, a_roll, ent_curve, ent_heat])

        with gr.Tab("3 · Patch tokens"):
            with gr.Accordion("❔ What can I do on this tab? (click to open)", open=False):
                gr.Markdown(GUIDE_TOKENS)
            with gr.Row():
                nclu = gr.Slider(2, 12, value=6, step=1, label="UMAP k-means clusters",
                                 info="How many groups the token UMAP is coloured into.")
                maskq = gr.Slider(0.05, 0.6, value=0.25, step=0.05,
                                  label="foreground quantile",
                                  info="Fraction of the PC1 range treated as background. "
                                       "Raise to keep more of the image.")
                prow = gr.Slider(0, 36, value=18, step=1, label="query patch row",
                                 info="Patch grid row to use as the similarity query.")
                pcol = gr.Slider(0, 36, value=18, step=1, label="query patch col",
                                 info="Patch grid column to use as the similarity query.")
                deb = gr.Checkbox(
                    value=False, label="positional debiasing",
                    info="Projects out the absolute-coordinate subspace. DINOv3 features "
                         "carry strong position bias that fakes convincing similarity maps.")
            emb_btn = gr.Button("Render patch analysis", variant="primary")
            emb_msg = gr.Markdown("")
            with gr.Row():
                e_rgb = gr.Image(label="PCA → RGB", type="numpy")
                e_fg = gr.Image(label="foreground mask", type="numpy")
            with gr.Row():
                e_sim = gr.Image(label="patch similarity", type="numpy")
                e_mark = gr.Image(label="query patch location", type="numpy")
            e_umap = gr.Image(label="token UMAP (k-means coloured)", type="numpy")
            with gr.Row():
                e_norm = gr.Image(label="token norms per layer", type="numpy")
                e_out = gr.Image(label="norm outliers", type="numpy")
            emb_btn.click(render_embeddings, [nclu, prow, pcol, deb, maskq],
                          [emb_msg, e_rgb, e_fg, e_sim, e_mark, e_umap, e_norm, e_out])

        with gr.Tab("4 · Residual stream"):
            with gr.Accordion("❔ What can I do on this tab? (click to open)", open=False):
                gr.Markdown(GUIDE_RESIDUAL)
            res_btn = gr.Button("Render residual analysis (fast)", variant="primary")
            res_msg = gr.Markdown("")
            with gr.Row():
                r_cka = gr.Image(label="layer CKA (single image)", type="numpy")
                r_norm = gr.Image(label="norm profile", type="numpy")
            r_drift = gr.Image(label="representation drift", type="numpy")
            res_btn.click(render_residual, None, [res_msg, r_cka, r_norm, r_drift])

            gr.Markdown("---\n### Layer-wise linear probe — the decision tool")
            gr.Markdown(
                "Trains a plain logistic regression on frozen features at every layer. "
                "Tells you whether adaptation is needed at all, and which blocks to target."
            )
            with gr.Row():
                pr_ds = gr.Dropdown(choices=DATASET_CHOICES, value=DEFAULT_DATASET,
                                    label="probe set",
                                    info="Match this to the backbone's modality for a "
                                         "meaningful curve.")
                pr_n = gr.Slider(5, 100, value=30, step=5, label="images per class",
                                 info="Cost scales linearly. 30 is a good default.")
                pr_pool = gr.Radio(choices=["mean", "max", "cls"], value="mean",
                                   label="token pooling",
                                   info="How patch tokens are collapsed into one vector "
                                        "per image before probing.")
            pr_btn = gr.Button("Run probe (slow — extracts features for the whole set)",
                               variant="primary")
            pr_msg = gr.Markdown("")
            with gr.Row():
                pr_fig = gr.Image(label="probe accuracy per layer", type="numpy")
                pr_norm = gr.Image(label="feature norm per layer", type="numpy")

            gr.Markdown("---\n### CKA over a dataset")
            with gr.Row():
                ck_ds = gr.Dropdown(choices=DATASET_CHOICES, value=DEFAULT_DATASET,
                                    label="set")
                ck_n = gr.Slider(2, 16, value=6, step=1, label="images",
                                 info="More images = a more stable layer-similarity "
                                      "picture.")
                ck_btn = gr.Button("Compute multi-image CKA")
            ck_msg = gr.Markdown("")
            ck_fig = gr.Image(label="layer CKA (multi-image)", type="numpy")

            pr_btn.click(run_layer_probe, [pr_ds, pr_n, pr_pool], [pr_msg, pr_fig, pr_norm])
            ck_btn.click(render_cka_multi, [ck_ds, ck_n], [ck_msg, ck_fig])

        with gr.Tab("5 · Models"):
            with gr.Accordion("❔ What can I do on this tab? (click to open)", open=False):
                gr.Markdown(GUIDE_MODELS)
            gr.Markdown("### What each backbone is for")
            gr.Markdown("\n\n---\n\n".join(
                f"**`{k}`** — {registry.MODELS[k].label}\n\n"
                f"{guides.model_guide_markdown(k)}"
                for k in registry.MODELS
            ))
            gr.Markdown("### Catalogue")
            gr.Dataframe(value=registry_table(), headers=REGISTRY_HEADERS, wrap=True,
                         interactive=False)
            gr.Markdown(
                f"device `{'mps' if torch.backends.mps.is_available() else 'cpu'}` · "
                f"torch `{torch.__version__}` · "
                f"transformers `{__import__('transformers').__version__}` · "
                f"timm `{__import__('timm').__version__}` · "
                f"errors are logged to `logs/errors.log`"
            )

        with gr.Tab("6 · Glossary"):
            gr.Markdown(guides.glossary_markdown())

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=1)
    app.launch(server_name="127.0.0.1",
               server_port=int(os.environ.get("MEDFM_PORT", 7860)),
               theme=gr.themes.Soft(), inbrowser=True)
