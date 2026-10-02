"""MedMNIST+ at 224px — the fast, multi-modality benchmark for this lab.

MedMNIST v2 ships 12 2-D datasets pre-processed at 28x28; MedMNIST+ adds 64, 128 and
224 px versions of the same splits, which is what makes them usable for foundation-model
evaluation (a 14 or 16 px patch model sees a 28x28 image as a 2x2 patch grid, which is
useless). Everything here defaults to size=224.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_ROOT = os.path.expanduser("~/medfm-lab/data/medmnist")

# Curated 2-D subset: modality coverage across CT, OCT, X-ray, pathology, ultrasound,
# fundus, dermatoscopy and blood smear.
MEDMNIST_2D = {
    "pathmnist":      ("Pathology (colon, H&E)", "pathology", "multi-class"),
    "tissuemnist":    ("Microscopy (tissue)", "microscopy", "multi-class"),
    "bloodmnist":     ("Microscopy (blood cells)", "microscopy", "multi-class"),
    "octmnist":       ("OCT (retina)", "oct", "multi-class"),
    "retinamnist":    ("Fundus (retina)", "fundus", "ordinal"),
    "dermamnist":     ("Dermatoscopy", "dermatoscopy", "multi-class"),
    "breastmnist":    ("Ultrasound (breast)", "ultrasound", "binary"),
    "pneumoniamnist": ("Chest X-ray (pneumonia)", "x-ray", "binary"),
    "chestmnist":     ("Chest X-ray (14 findings)", "x-ray", "multi-label"),
    "organamnist":    ("CT (abdominal, axial)", "ct", "multi-class"),
    "organcmnist":    ("CT (abdominal, coronal)", "ct", "multi-class"),
    "organsmnist":    ("CT (abdominal, sagittal)", "ct", "multi-class"),
}

TASK_TO_LABELS = {
    "binary": 2,
    "multi-class": None,     # filled from medmnist INFO
    "multi-label": None,
    "ordinal": None,
}

# Exact Content-Length of the MedMNIST+ 224px archives on Zenodo (measured via HEAD requests).
# MedMNIST+ resamples 28px sources up to 224px and the files balloon: pathmnist goes from
# 206 MB to 12.6 GB. Without these numbers the UI cheerfully starts a 12 GB download behind a
# spinner that is indistinguishable from a hang, so they drive both the size labels and the
# truncation check below.
SIZE_224_GB = {
    "breastmnist": 0.031,
    "retinamnist": 0.128,
    "pneumoniamnist": 0.214,
    "organcmnist": 0.760,
    "organsmnist": 0.803,
    "dermamnist": 1.091,
    "bloodmnist": 1.541,
    "organamnist": 1.804,
    "tissuemnist": 3.434,
    "chestmnist": 3.889,
    "octmnist": 3.959,
    "pathmnist": 12.630,
}


def size_label(flag: str, size: int = 224) -> str:
    gb = SIZE_224_GB.get(flag) if size == 224 else None
    if gb is None:
        return "size unknown"
    if gb < 1:
        return f"{gb * 1000:.0f} MB download"
    return f"{gb:.1f} GB download"


def option_label(flag: str) -> str:
    """Dropdown text: flag, description, and the real one-time download size."""
    desc = MEDMNIST_2D.get(flag, ("", "", ""))[0]
    return f"{flag} — {desc} · {size_label(flag)}"


def ensure_intact(flag: str, size: int = 224, root: str = DEFAULT_ROOT) -> str | None:
    """Remove a truncated cache file so it gets re-downloaded.

    medmnist caches purely by filename and torchvision writes straight to the final path, so
    an interrupted download leaves a partial .npz that is happily accepted on the next run
    and then fails deep inside np.load with a message that gives no hint about the cause.
    """
    if size != 224 or flag not in SIZE_224_GB:
        return None
    path = os.path.join(root, f"{flag}_{size}.npz")
    if not os.path.exists(path):
        return None
    expected = SIZE_224_GB[flag] * 1e9
    actual = os.path.getsize(path)
    if actual < expected * 0.98:
        os.remove(path)
        return (f"removed a truncated {os.path.basename(path)} "
                f"({actual / 1e9:.2f} GB of {expected / 1e9:.2f} GB) — it will re-download")
    return None


@dataclass
class Sample:
    image: "np.ndarray"      # (S, S, 3) uint8 RGB
    label: np.ndarray        # (n_classes,) for multi-label, else shape (1,)
    index: int
    dataset: str

    @property
    def label_scalar(self) -> int:
        return int(np.asarray(self.label).ravel()[0])


def medmnist_info():
    from medmnist import INFO

    return INFO


def load_split(flag: str = "pathmnist", split: str = "test", size: int = 224,
               root: str = DEFAULT_ROOT, download: bool = True, as_rgb: bool = True):
    """Return the raw medmnist Dataset object for a flag/split/size."""
    import medmnist
    from medmnist import INFO

    if flag not in INFO:
        raise KeyError(f"unknown medmnist flag {flag!r}")
    DataClass = getattr(medmnist, INFO[flag]["python_class"])
    os.makedirs(root, exist_ok=True)
    ensure_intact(flag, size=size, root=root)
    return DataClass(split=split, size=size, download=download, as_rgb=as_rgb, root=root)


def dataset_meta(flag: str) -> dict:
    from medmnist import INFO

    info = INFO[flag]
    return {
        "flag": flag,
        "description": MEDMNIST_2D.get(flag, ("", "", ""))[0],
        "modality": MEDMNIST_2D.get(flag, ("", "", ""))[1],
        "n_channels": info["n_channels"],
        "task": info["task"],
        "n_classes": len(info["label"]),
        "label_map": info["label"],
        "n_samples": info["n_samples"],
        "size_224_gb": SIZE_224_GB.get(flag),
        "cached": os.path.exists(os.path.join(DEFAULT_ROOT, f"{flag}_224.npz")),
    }


def fetch_samples(flag: str = "pathmnist", split: str = "test", n: int = 64,
                  size: int = 224, offset: int = 0, root: str = DEFAULT_ROOT) -> list[Sample]:
    """Pull n images out of a split as Sample objects."""
    ds = load_split(flag=flag, split=split, size=size, root=root)
    total = len(ds)
    if n <= 0 or n > total:
        n = total
    idxs = range(offset, min(offset + n, total))
    out = []
    for i in idxs:
        img, lab = ds[i]
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            a = arr.astype(np.float32)
            if a.max() <= 1.0 + 1e-6:
                a = a * 255.0
            arr = np.clip(a, 0, 255).astype(np.uint8)
        out.append(Sample(image=arr, label=np.asarray(lab), index=int(i),
                          dataset=flag))
    return out


def label_name(flag: str, value: int) -> str:
    try:
        from medmnist import INFO

        return INFO[flag]["label"].get(str(value), str(value))
    except Exception:
        return str(value)


def build_gallery(flag: str = "pathmnist", n: int = 96, size: int = 224,
                  split: str = "test", root: str = DEFAULT_ROOT) -> list[Sample]:
    """A small labelled reference set for kNN retrieval and probe fitting."""
    return fetch_samples(flag=flag, split=split, n=n, size=size, root=root)


def build_probe_set(flag: str = "pathmnist", n_per_class: int = 40, size: int = 224,
                    split: str = "train", root: str = DEFAULT_ROOT,
                    binary_from_multilabel: int = 1) -> list[Sample]:
    """Stratified probe set for layer-wise linear probing.

    For multi-label sets (chestmnist) we collapse to a binary task on one finding so the
    probe stays a clean single-label problem.
    """
    ds = load_split(flag=flag, split=split, size=size, root=root)
    meta = dataset_meta(flag)
    buckets: dict[int, list[Sample]] = {}
    limit = len(ds)
    for i in range(limit):
        img, lab = ds[i]
        y = int(np.asarray(lab).ravel()[0])
        if meta["task"] == "multi-label":
            y = int(np.asarray(lab).ravel()[binary_from_multilabel])
        if len(buckets.get(y, [])) >= n_per_class:
            if len(buckets) >= meta["n_classes"] and all(
                len(v) >= n_per_class for v in buckets.values()
            ):
                break
            continue
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            a = arr.astype(np.float32)
            arr = np.clip((a * 255.0 if a.max() <= 1.0 + 1e-6 else a), 0, 255).astype(np.uint8)
        buckets.setdefault(y, []).append(
            Sample(image=arr, label=np.asarray([y]), index=int(i), dataset=flag)
        )
        if len(buckets) >= meta["n_classes"] and all(
            len(v) >= n_per_class for v in buckets.values()
        ):
            break

    out: list[Sample] = []
    for y in sorted(buckets):
        out.extend(buckets[y][:n_per_class])
    return out


def gallery_options() -> list[str]:
    return [f"{k} — {v[0]}" for k, v in MEDMNIST_2D.items()]


def flag_from_option(option: str) -> str:
    """Recover the medmnist flag from a dropdown display string.

    The UI cannot show bare flags (unreadable) and tuple choices are unreliable in Gradio 6,
    so the label IS the value and has to be parsed back down to the flag.
    """
    if option in INFO_KEYS:
        return option
    return option.split(" — ")[0].split(" - ")[0].strip()


INFO_KEYS: tuple = ()


def _init_info_keys():
    global INFO_KEYS
    try:
        from medmnist import INFO

        INFO_KEYS = tuple(INFO.keys())
    except Exception:
        INFO_KEYS = tuple(MEDMNIST_2D.keys())


_init_info_keys()
