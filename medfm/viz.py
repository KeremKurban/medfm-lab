"""Plotting helpers. Everything returns something Gradio can display (numpy arrays)."""

from __future__ import annotations

import io
from typing import Optional

import numpy as np

# A perceptually ordered colormap that reads well on greyscale medical images.
DEFAULT_CMAP = "turbo"


def _cmap(name: str):
    import matplotlib

    return matplotlib.colormaps[name] if hasattr(matplotlib, "colormaps") else \
        matplotlib.cm.get_cmap(name)


def normalize01(x: np.ndarray, robust: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if robust:
        lo, hi = np.percentile(x, [1, 99])
    else:
        lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def overlay_heatmap(image: np.ndarray, heat: np.ndarray, alpha: float = 0.5,
                    cmap: str = DEFAULT_CMAP, resize_to_image: bool = True) -> np.ndarray:
    """Alpha-blend a (h, w) heatmap onto a greyscale image. Returns uint8 RGB."""
    import matplotlib.pyplot as plt

    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    img = normalize01(img, robust=False)

    h = normalize01(np.asarray(heat, dtype=np.float32))
    if resize_to_image and h.shape != img.shape:
        from PIL import Image

        h = np.asarray(
            Image.fromarray((h * 255).astype(np.uint8)).resize(
                (img.shape[1], img.shape[0]), Image.BICUBIC
            ),
            dtype=np.float32,
        ) / 255.0

    rgba = _cmap(cmap)(h)
    rgb = rgba[..., :3]

    base = np.stack([img] * 3, axis=-1)
    out = (1 - alpha) * base + alpha * rgb
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def heat_image(heat: np.ndarray, cmap: str = DEFAULT_CMAP, robust: bool = True) -> np.ndarray:
    """Render a scalar map through a colormap, passing RGB images through unchanged."""
    h = np.asarray(heat, dtype=np.float32)
    if h.ndim == 3 and h.shape[-1] == 3:
        # Already an RGB image (e.g. a PCA->RGB map); matplotlib would treat the last axis
        # as RGB values and return a (h, w, 3, 4) array if we handed it to a colormap.
        return (np.clip(h, 0, 1) * 255).astype(np.uint8)
    h = normalize01(h, robust=robust)
    return (np.clip(_cmap(cmap)(h)[..., :3], 0, 1) * 255).astype(np.uint8)


def mask_overlay(image: np.ndarray, mask: np.ndarray, color=(0.1, 0.9, 0.3),
                 alpha: float = 0.45, dim: float = 0.45) -> np.ndarray:
    """Dim the background and tint the masked region."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    img = normalize01(img, robust=False)
    from PIL import Image

    if mask.shape != img.shape:
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (img.shape[1], img.shape[0]), Image.NEAREST
            )
        ) > 127
    base = np.stack([img] * 3, axis=-1) * dim
    tint = np.array(color, dtype=np.float32)[None, None, :]
    out = np.where(mask[..., None], (1 - alpha) * base + alpha * tint, base)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def upscale(arr: np.ndarray, size: int = 448, nearest: bool = True) -> np.ndarray:
    """Blow a small map up to a displayable size.

    Patch-grid maps are one pixel per patch: a 37x37 grid at 518px input is genuinely a
    37-by-37-pixel image, which renders as a postage stamp. Nearest-neighbour keeps each
    patch a crisp square, the way the DINO papers show them.
    """
    from PIL import Image

    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8) if a.max() <= 1.001 else a.astype(np.uint8)
    if a.shape[0] == size and a.shape[1] == size:
        return a
    resample = Image.NEAREST if nearest else Image.BICUBIC
    return np.asarray(Image.fromarray(a).resize((size, size), resample))


def caption(text: str, figsize=(7.4, 1.0)) -> np.ndarray:
    """A figure whose only job is to carry a caption at a readable size."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.text(0.0, 0.5, text, fontsize=8.5, va="center", ha="left", wrap=True,
            color="#2b2b2b")
    return figure_to_array(fig)


def side_by_side(panels: list, titles: list, cell: int = 300) -> np.ndarray:
    """Row of equally sized panels with titles, for before/after comparisons."""
    import matplotlib.pyplot as plt

    n = max(len(panels), 1)
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 2.5))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        if i < len(panels) and panels[i] is not None:
            a = np.asarray(panels[i])
            if a.ndim == 3 and a.shape[-1] == 1:
                a = a[..., 0]
            ax.imshow(a, cmap=None if (a.ndim == 3 and a.shape[-1] == 3) else "gray")
        ax.set_xticks([])
        ax.set_yticks([])
        if i < len(titles):
            ax.set_title(titles[i], fontsize=8)
    return figure_to_array(fig)


def figure_to_array(fig) -> np.ndarray:
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    from PIL import Image

    arr = np.asarray(Image.open(buf).convert("RGB"))
    import matplotlib.pyplot as plt

    plt.close(fig)
    return arr


def line_plot(series: dict[str, np.ndarray], title: str = "", xlabel: str = "layer",
              ylabel: str = "", xtick_labels: Optional[list] = None,
              ylim: Optional[tuple] = None, hline: Optional[float] = None,
              figsize=(7.2, 3.0)) -> np.ndarray:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for name, y in series.items():
        y = np.asarray(y)
        ax.plot(np.arange(len(y)), y, marker="o", ms=3.5, lw=1.6, label=name)
    if hline is not None:
        ax.axhline(hline, ls="--", lw=1, color="grey", alpha=0.8)
    if xtick_labels is not None and len(xtick_labels) == len(next(iter(series.values()))):
        ax.set_xticks(np.arange(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    if len(series) > 1:
        ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    return figure_to_array(fig)


def heatmap_plot(M: np.ndarray, title: str = "", xtick_labels=None, ytick_labels=None,
                 cmap: str = "viridis", figsize=(6.0, 5.0), vmin=None, vmax=None,
                 annotate: bool = False) -> np.ndarray:
    import matplotlib.pyplot as plt

    M = np.asarray(M, dtype=np.float32)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=10)
    if xtick_labels is not None:
        ax.set_xticks(np.arange(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6)
    if ytick_labels is not None:
        ax.set_yticks(np.arange(len(ytick_labels)))
        ax.set_yticklabels(ytick_labels, fontsize=6)
    if annotate and M.size <= 400:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return figure_to_array(fig)


def scatter_plot(coords: np.ndarray, color=None, size: float = 14, title: str = "",
                 xlabel: str = "dim 1", ylabel: str = "dim 2", cmap: str = "tab10",
                 colorbar_label: str = "", figsize=(5.4, 4.4)) -> np.ndarray:
    import matplotlib.pyplot as plt

    coords = np.asarray(coords, dtype=np.float32)
    fig, ax = plt.subplots(figsize=figsize)
    if color is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=size, alpha=0.85, lw=0)
    else:
        color = np.asarray(color)
        discrete = color.dtype.kind in "iu" and len(np.unique(color)) <= 20
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=color, s=size, alpha=0.9, lw=0,
                        cmap=("tab10" if discrete else "plasma"))
        if not discrete:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.2, lw=0.5)
    return figure_to_array(fig)


def gallery_strip(images: list, labels: list, scores: Optional[list] = None,
                  title: str = "", cell: int = 128, max_items: int = 8) -> np.ndarray:
    """Contact sheet of retrieval hits: [image | label | similarity]."""
    import matplotlib.pyplot as plt

    n = min(len(images), max_items)
    if n == 0:
        return np.zeros((cell, cell, 3), dtype=np.uint8)

    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 1.9))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        img = np.asarray(images[i])
        if img.ndim == 3 and img.shape[-1] == 3:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        cap = f"y={labels[i]}"
        if scores is not None:
            cap += f"\n{scores[i]:.2f}"
        ax.set_title(cap, fontsize=7)
    if title:
        fig.suptitle(title, fontsize=9)
    return figure_to_array(fig)


def bar_plot(values: dict[str, float], title: str = "", figsize=(5.4, 2.8),
             ylabel: str = "", horizontal: bool = True, xerr: Optional[np.ndarray] = None) -> np.ndarray:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    names = list(values)
    vals = np.array([values[k] for k in names], dtype=np.float32)
    if horizontal:
        ax.barh(names, vals, xerr=xerr, color="#4c78a8")
        ax.invert_yaxis()
    else:
        ax.bar(names, vals, yerr=xerr, color="#4c78a8")
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9) if not horizontal else ax.set_xlabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5, axis="x" if horizontal else "y")
    return figure_to_array(fig)


def histogram_plot(x: np.ndarray, title: str = "", xlabel: str = "", bins: int = 40,
                   figsize=(6.0, 2.6), vline: Optional[float] = None) -> np.ndarray:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(np.asarray(x).ravel(), bins=bins, color="#4c78a8", alpha=0.85)
    if vline is not None:
        ax.axvline(vline, color="crimson", ls="--", lw=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    return figure_to_array(fig)
