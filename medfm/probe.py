"""Frozen-feature probes.

If a linear model on frozen features already solves the task, there is nothing to adapt
and domain adaptation is wasted compute. If it does not, the layer-wise probe curve tells
you which blocks are worth touching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ProbeResult:
    accuracy: float
    std: float = 0.0
    n_train: int = 0
    n_test: int = 0
    classes: list = field(default_factory=list)
    confusion: Optional[np.ndarray] = None
    per_class_accuracy: Optional[np.ndarray] = None
    model: object = None


def _make_logreg(C: float = 1.0, class_weight=None):
    """LogisticRegression with the kwargs that this scikit-learn version still accepts.

    `multi_class` and `n_jobs` were removed from LogisticRegression in recent releases, so
    passing them unconditionally breaks on any modern install.
    """
    from sklearn.linear_model import LogisticRegression

    try:
        return LogisticRegression(max_iter=3000, C=C, class_weight=class_weight, n_jobs=1)
    except TypeError:
        return LogisticRegression(max_iter=3000, C=C, class_weight=class_weight)


def fit_linear_probe(X: np.ndarray, y: np.ndarray, test_frac: float = 0.3,
                     C: float = 1.0, seed: int = 0, return_model: bool = False,
                     class_weight: str = "balanced") -> ProbeResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).ravel()
    classes = np.unique(y)

    if len(classes) < 2 or len(y) < 4:
        return ProbeResult(accuracy=float("nan"), n_train=len(y), classes=classes.tolist())

    strat = y if min(np.bincount(y.astype(int))) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_frac, random_state=seed,
                                          stratify=strat)
    model = make_pipeline(
        StandardScaler(),
        _make_logreg(C=C, class_weight=class_weight),
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = float((pred == yte).mean())

    cm = confusion_matrix(yte, pred, labels=classes)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    return ProbeResult(
        accuracy=acc,
        n_train=len(ytr),
        n_test=len(yte),
        classes=classes.tolist(),
        confusion=cm,
        per_class_accuracy=per_class,
        model=model if return_model else None,
    )


def knn_probe(X: np.ndarray, y: np.ndarray, k: int = 5, test_frac: float = 0.3,
              seed: int = 0, metric: str = "cosine") -> ProbeResult:
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).ravel()
    classes = np.unique(y)
    if len(classes) < 2 or len(y) < 4:
        return ProbeResult(accuracy=float("nan"), classes=classes.tolist())

    strat = y if min(np.bincount(y.astype(int))) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_frac, random_state=seed,
                                          stratify=strat)
    model = make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=min(k, len(Xtr)),
                                                             metric=metric))
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = float((pred == yte).mean())
    cm = confusion_matrix(yte, pred, labels=classes)
    return ProbeResult(
        accuracy=acc, n_train=len(Xtr), n_test=len(Xte), classes=classes.tolist(),
        confusion=cm, per_class_accuracy=cm.diagonal() / np.maximum(cm.sum(axis=1), 1),
        model=model,
    )


class FeatureIndex:
    """A labelled gallery of embeddings with a cosine kNN retrieval API."""

    def __init__(self):
        self.vectors: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.images: list = []
        self.indices: Optional[np.ndarray] = None   # source dataset indices, to exclude self
        self.dataset: str = ""
        self.model_key: str = ""

    def build(self, vectors: np.ndarray, labels: np.ndarray, images: Optional[list] = None,
              dataset: str = "", model_key: str = "",
              indices: Optional[np.ndarray] = None):
        V = np.asarray(vectors, dtype=np.float32)
        self.vectors = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        self.labels = np.asarray(labels).ravel()
        self.images = list(images) if images is not None else []
        self.indices = None if indices is None else np.asarray(indices).ravel()
        self.dataset = dataset
        self.model_key = model_key
        return self

    @property
    def size(self) -> int:
        return 0 if self.vectors is None else int(self.vectors.shape[0])

    def query(self, vector: np.ndarray, k: int = 8,
              exclude_index: Optional[int] = None) -> list[dict]:
        """Nearest neighbours by cosine similarity.

        `exclude_index` drops a specific source sample — pass the index of the query image
        when it may also be in the gallery. Retrieval that includes the query itself returns
        similarity 1.0 and a guaranteed label match, which looks like a perfect result and
        measures nothing.
        """
        if self.vectors is None or self.size == 0:
            return []
        q = np.asarray(vector, dtype=np.float32).ravel()
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = self.vectors @ q

        mask = np.ones(self.size, dtype=bool)
        if exclude_index is not None and self.indices is not None:
            mask = self.indices != int(exclude_index)
        if not mask.any():
            return []

        cand = np.where(mask)[0]
        order = cand[np.argsort(-sims[cand])]
        top = order[: int(min(k, len(order)))]
        return [
            {
                "index": int(self.indices[i]) if self.indices is not None else int(i),
                "similarity": float(sims[i]),
                "label": int(self.labels[i]),
                "image": self.images[i] if i < len(self.images) else None,
            }
            for i in top
        ]

    def label_histogram(self, vector: np.ndarray, k: int = 8,
                        exclude_index: Optional[int] = None) -> dict:
        hits = self.query(vector, k=k, exclude_index=exclude_index)
        hist: dict[int, float] = {}
        total = sum(max(h["similarity"], 0.0) for h in hits) or 1.0
        for h in hits:
            hist[h["label"]] = hist.get(h["label"], 0.0) + max(h["similarity"], 0.0) / total
        return dict(sorted(hist.items(), key=lambda kv: -kv[1]))
