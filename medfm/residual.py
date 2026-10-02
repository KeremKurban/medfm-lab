"""Residual-stream analysis.

The residual stream is the sequence of hidden states `x_0 .. x_L`, where `x_0` is the
patch embedding and `x_l` is the output of block `l`. Everything the model knows at
layer `l` is linearly readable from `x_l`, which makes layer-wise linear probing the
single most useful measurement for deciding *where* to intervene when adapting a
backbone to a new domain.

Provided here: norms, CKA between layers, linear-probe curves, representation drift,
and a quick "which layer should I adapt" recommendation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def cka(X: np.ndarray, Y: np.ndarray, kernel: str = "linear") -> float:
    """Centred Kernel Alignment between two representation matrices (N, D)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]

    def _gram(A, kind):
        if kind == "linear":
            return A @ A.T
        if kind == "rbf":
            sq = np.sum(A ** 2, axis=1)
            d2 = sq[:, None] + sq[None, :] - 2 * A @ A.T
            med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
            return np.exp(-d2 / max(med, 1e-12))
        raise ValueError("kernel must be linear or rbf")

    K, L = _gram(X, kernel), _gram(Y, kernel)
    H = np.eye(n) - 1.0 / n
    K = H @ K @ H
    L = H @ L @ H
    num = np.sum(K * L)
    den = np.sqrt(np.sum(K * K) * np.sum(L * L)) + 1e-12
    return float(num / den)


def cka_matrix(hidden_states: np.ndarray, num_prefix: int = 1,
               max_tokens: int = 2048, kernel: str = "linear",
               token_sample_seed: int = 0) -> np.ndarray:
    """(L+1, L+1) CKA between every pair of layers, over patch tokens.

    hidden_states: (L+1, T, D) — uses patch tokens only (skips cls/registers).
    """
    H = np.asarray(hidden_states, dtype=np.float32)
    Lp1 = H.shape[0]
    X = H[:, num_prefix:, :]                                  # (L+1, P, D)
    P = X.shape[1]
    if P > max_tokens:
        rng = np.random.default_rng(token_sample_seed)
        sel = rng.choice(P, size=max_tokens, replace=False)
        X = X[:, sel, :]

    flat = [X[i] for i in range(Lp1)]
    M = np.eye(Lp1, dtype=np.float64)
    for i in range(Lp1):
        for j in range(i + 1, Lp1):
            v = cka(flat[i], flat[j], kernel=kernel)
            M[i, j] = M[j, i] = v
    return M


def cka_matrix_multi(hidden_list: list, num_prefix: int = 1, max_tokens: int = 4096,
                     kernel: str = "linear", seed: int = 0) -> np.ndarray:
    """CKA across layers pooled over several images (concatenates patch tokens).

    Gives a far more stable layer-similarity picture than a single image, at the cost of
    a few hundred MB of memory at most.
    """
    stack = [np.asarray(h, dtype=np.float32) for h in hidden_list]
    Lp1 = stack[0].shape[0]
    per_layer = []
    for l in range(Lp1):
        toks = np.concatenate([h[l, num_prefix:, :] for h in stack], axis=0)
        per_layer.append(toks)
    P = per_layer[0].shape[0]
    if P > max_tokens:
        rng = np.random.default_rng(seed)
        sel = rng.choice(P, size=max_tokens, replace=False)
        per_layer = [t[sel] for t in per_layer]

    M = np.eye(Lp1, dtype=np.float64)
    for i in range(Lp1):
        for j in range(i + 1, Lp1):
            v = cka(per_layer[i], per_layer[j], kernel=kernel)
            M[i, j] = M[j, i] = v
    return M


def layer_probe_curve(hidden_states: np.ndarray, labels: np.ndarray,
                      num_prefix: int = 1, pool: str = "mean",
                      clf: str = "logreg", n_splits: int = 5,
                      seed: int = 0, C: float = 1.0) -> dict:
    """Linear-probe accuracy at every layer — where is the label linearly decodable?

    hidden_states: (L+1, T, D) for ONE image, or (N, L+1, T, D) for many. With a single
    image you get a degenerate curve (one sample per layer); pass a stack.
    """
    H = np.asarray(hidden_states, dtype=np.float32)
    if H.ndim == 3:
        H = H[None, ...]
    N, Lp1, T, D = H.shape
    y = np.asarray(labels)

    accs, pops = [], []
    for l in range(Lp1):
        X = H[:, l, num_prefix:, :]                            # (N, P, D)
        if pool == "mean":
            Z = X.mean(axis=1)
        elif pool == "max":
            Z = X.max(axis=1)
        elif pool == "cls":
            Z = H[:, l, 0, :]
        else:
            raise ValueError("pool must be mean/max/cls")
        scores = _cv_score(Z, y, clf=clf, n_splits=n_splits, seed=seed, C=C)
        accs.append(scores["mean"])
        pops.append(scores["std"])

    return _pack_probe_curve(np.asarray(accs), np.asarray(pops))


def layer_probe_curve_pooled(X: np.ndarray, labels: np.ndarray, clf: str = "logreg",
                             n_splits: int = 5, seed: int = 0, C: float = 1.0) -> dict:
    """Same as layer_probe_curve but for pre-pooled features of shape (N, L+1, D).

    This is the memory-safe route: pool inside the encoder loop, probe afterwards.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"expected (N, L+1, D), got {X.shape}")
    y = np.asarray(labels)
    accs, pops = [], []
    for l in range(X.shape[1]):
        s = _cv_score(X[:, l, :], y, clf=clf, n_splits=n_splits, seed=seed, C=C)
        accs.append(s["mean"])
        pops.append(s["std"])
    return _pack_probe_curve(np.asarray(accs), np.asarray(pops))


def _pack_probe_curve(accs: np.ndarray, pops: np.ndarray) -> dict:
    best = int(np.argmax(accs)) if accs.size else 0
    return {
        "accuracy": accs,
        "std": pops,
        "best_layer": best,
        "best_accuracy": float(accs[best]) if accs.size else 0.0,
        "layer_labels": ["embed"] + [f"block {i + 1}" for i in range(len(accs) - 1)],
    }


def _cv_score(Z, y, clf="logreg", n_splits=5, seed=0, C=1.0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y)
    if clf == "logreg":
        kwargs = dict(max_iter=2000, C=C)
        try:
            model_bare = LogisticRegression(n_jobs=1, **kwargs)
        except TypeError:
            # n_jobs was removed from LogisticRegression in recent scikit-learn
            model_bare = LogisticRegression(**kwargs)
        model = make_pipeline(StandardScaler(), model_bare)
    else:
        from sklearn.neighbors import KNeighborsClassifier

        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

    min_count = min(np.bincount(y.astype(int))) if y.dtype.kind in "iu" else 1
    k = int(min(n_splits, max(2, min_count)))
    if k < 2 or len(y) < 2 * k:
        Xtr, Xte, ytr, yte = train_test_split(Z, y, test_size=0.3, random_state=seed,
                                              stratify=y if min_count >= 2 else None)
        model.fit(Xtr, ytr)
        return {"mean": float(accuracy_score(yte, model.predict(Xte))), "std": 0.0}

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    s = cross_val_score(model, Z, y, cv=cv, scoring="accuracy")
    return {"mean": float(s.mean()), "std": float(s.std())}


def representation_drift(hidden_states: np.ndarray, num_prefix: int = 1) -> np.ndarray:
    """Cosine similarity between layer l and l+1 patch representations (DINO's CKA-lite).

    A low value means that block rewrote the representation substantially.
    """
    H = np.asarray(hidden_states, dtype=np.float32)[:, num_prefix:, :]
    Lp1 = H.shape[0]
    sims = np.zeros(Lp1 - 1, dtype=np.float32)
    for l in range(Lp1 - 1):
        A = H[l]
        B = H[l + 1]
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        sims[l] = float((An * Bn).sum(axis=1).mean())
    return sims


def residual_norm_profile(hidden_states: np.ndarray, num_prefix: int = 1) -> dict:
    """Per-layer norm statistics of cls and patch tokens — growth, spikes, collapse."""
    H = np.asarray(hidden_states, dtype=np.float32)
    cls_norms = np.linalg.norm(H[:, 0, :], axis=-1)
    patches = H[:, num_prefix:, :]
    patch_norms = np.linalg.norm(patches, axis=-1)              # (L+1, P)
    return {
        "cls_norms": cls_norms,
        "patch_norm_mean": patch_norms.mean(axis=1),
        "patch_norm_max": patch_norms.max(axis=1),
        "patch_norm_std": patch_norms.std(axis=1),
        "layer_labels": ["embed"] + [f"block {i + 1}" for i in range(H.shape[0] - 1)],
    }


def adapt_layer_recommendation(probe_curve: dict, drift: Optional[np.ndarray] = None) -> dict:
    """Turn probe + drift measurements into a concrete recommendation.

    Heuristic: adapters are cheapest and most effective in the middle-to-late blocks,
    where semantic content has formed but the representation is still changing.
    """
    acc = np.asarray(probe_curve["accuracy"])
    best = int(np.argmax(acc))
    late = acc[max(best - 3, 0):]
    plateau = int(max(best - 3, 0) + int(np.argmax(late)))
    rec = {
        "best_probe_layer": best,
        "plateau_layer": plateau,
        "suggested_start_block": max(1, plateau - 3),
        "suggested_end_block": best,
    }
    if drift is not None and len(drift):
        rec["most_changed_block"] = int(np.argmin(drift)) + 1
    rec["advice"] = (
        f"highest linear decodability at layer {best}; place adapters roughly in blocks "
        f"{rec['suggested_start_block']}-{rec['suggested_end_block']} and keep the early "
        f"blocks frozen to protect generic low-level features"
    )
    return rec
