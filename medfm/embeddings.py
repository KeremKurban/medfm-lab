"""Patch-token embedding analysis: PCA maps, UMAP, similarity, token statistics.

Most of this is the DINOv2/v3 paper's own diagnostic toolkit, generalised so it can be
pointed at any backbone and any layer of the residual stream.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# --------------------------------------------------------------------- PCA maps

def pca_rgb(tokens: np.ndarray, grid: tuple[int, int], n_components: int = 3,
            mask_background: bool = True, mask_quantile: float = 0.25,
            whiten: bool = False, seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Map the first three PCs of patch features to RGB, as in the DINO papers.

    tokens: (P, D). Returns (rgb (h, w, 3) in [0,1], mask (h, w) bool, info dict).
    """
    from sklearn.decomposition import PCA

    X = np.asarray(tokens, dtype=np.float32)
    P, D = X.shape
    k = min(n_components, D, max(P - 1, 1))
    pca = PCA(n_components=k, whiten=whiten, random_state=seed)
    comps = pca.fit_transform(X)                       # (P, k)

    if comps.shape[1] < 3:
        comps = np.pad(comps, ((0, 0), (0, 3 - comps.shape[1])))

    # Per-channel percentile stretch, then align signs so the map is reproducible.
    rgb = np.zeros((P, 3), dtype=np.float32)
    signs = np.ones(3, dtype=np.float32)
    for c in range(3):
        v = comps[:, c]
        lo, hi = np.percentile(v, [1, 99])
        if hi - lo < 1e-9:
            rgb[:, c] = 0.5
            continue
        scaled = np.clip((v - lo) / (hi - lo), 0, 1)
        # Sign is arbitrary in PCA; anchor it so the brightest PC1 region stays bright.
        if c == 0 and scaled.mean() > 0.5:
            scaled = 1.0 - scaled
            signs[c] = -1.0
        rgb[:, c] = scaled

    h, w = grid
    rgb_img = rgb[: h * w].reshape(h, w, 3)

    mask = np.ones((h, w), dtype=bool)
    info = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "signs": signs.tolist(),
    }
    if mask_background and P >= h * w:
        pc1 = comps[: h * w, 0].reshape(h, w)
        thr = np.quantile(pc1, mask_quantile)
        mask = pc1 > thr
        info["foreground_threshold"] = float(thr)
    return rgb_img, mask, info


def pca_project(tokens: np.ndarray, n_components: int = 3, seed: int = 0) -> np.ndarray:
    from sklearn.decomposition import PCA

    X = np.asarray(tokens, dtype=np.float32)
    k = min(n_components, X.shape[1], max(X.shape[0] - 1, 1))
    return PCA(n_components=k, random_state=seed).fit_transform(X)


# ------------------------------------------------------------------------ UMAP

def umap_2d(tokens: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1,
            metric: str = "cosine", seed: int = 0) -> np.ndarray:
    """2-D UMAP of patch tokens. Falls back to PCA if umap-learn is unavailable."""
    X = np.asarray(tokens, dtype=np.float32)
    try:
        import umap

        n = max(2, min(n_neighbors, X.shape[0] - 1))
        return umap.UMAP(n_components=2, n_neighbors=n, min_dist=min_dist,
                         metric=metric, random_state=seed).fit_transform(X)
    except Exception:
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(X)


def cluster_tokens(tokens: np.ndarray, n_clusters: int = 6, seed: int = 0) -> np.ndarray:
    """KMeans token cluster ids — the standard way to read structure off a UMAP."""
    from sklearn.cluster import KMeans

    X = np.asarray(tokens, dtype=np.float32)
    k = max(2, min(n_clusters, X.shape[0]))
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


# ------------------------------------------------------------------- similarity

def similarity_map(tokens: np.ndarray, query_idx: int, grid: tuple[int, int],
                   normalize: bool = True) -> np.ndarray:
    """Cosine similarity from one patch to every patch, as a (h, w) map."""
    X = np.asarray(tokens, dtype=np.float32)
    q = X[query_idx]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    qn = q / (np.linalg.norm(q) + 1e-8)
    sim = Xn @ qn
    h, w = grid
    m = sim[: h * w].reshape(h, w)
    if normalize:
        m = (m - m.min()) / max(m.max() - m.min(), 1e-8)
    return m


def similarity_matrix(tokens_a: np.ndarray, tokens_b: np.ndarray) -> np.ndarray:
    A = np.asarray(tokens_a, dtype=np.float32)
    B = np.asarray(tokens_b, dtype=np.float32)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A @ B.T


def positional_debias(tokens: np.ndarray, grid: tuple[int, int],
                      n_components: int = 4) -> np.ndarray:
    """Remove the coordinate-aligned positional component of a feature map.

    DINOv3 patch features carry strong absolute-position structure; projecting out the
    low-rank subspace spanned by (row, col) coordinates makes similarity maps reflect
    content rather than location.
    """
    X = np.asarray(tokens, dtype=np.float32)
    h, w = grid
    P = h * w
    if X.shape[0] < P:
        return X
    rows = np.repeat(np.arange(h), w).astype(np.float32)
    cols = np.tile(np.arange(w), h).astype(np.float32)
    coords = np.stack([rows, cols, rows * cols, rows ** 2, cols ** 2], axis=1)[:P]
    Z = np.concatenate([coords, np.ones((P, 1), dtype=np.float32)], axis=1)
    # ridge least squares, solve for every feature dimension at once
    lam = 1e-3 * np.trace(Z.T @ Z) / max(Z.shape[1], 1)
    A = Z.T @ Z + lam * np.eye(Z.shape[1], dtype=np.float32)
    beta = np.linalg.solve(A, Z.T @ X[:P])
    out = X.copy()
    out[:P] = X[:P] - Z @ beta
    return out


# ---------------------------------------------------------------- token statistics

def token_norms(hidden_states: np.ndarray) -> np.ndarray:
    """(L+1, T) L2 norm of every token at every layer of the residual stream."""
    H = np.asarray(hidden_states, dtype=np.float32)
    return np.linalg.norm(H, axis=-1)


def outlier_tokens(hidden_states: np.ndarray, layer: int = -1,
                   z_thresh: float = 3.0, num_prefix: int = 0) -> dict:
    """High-norm outlier tokens (registers / artifacts) at one layer.

    ViTs routinely dump global context into a few very high-norm patch tokens, which is
    exactly the failure mode that makes attention maps look wrong on medical images.

    `num_prefix` restricts the statistics and the returned arrays to patch tokens, which is
    what you want when aligning with a patch-token scatter plot. Leave it at 0 to include
    cls and register tokens (i.e. to see whether the registers themselves are outliers).
    """
    H = np.asarray(hidden_states, dtype=np.float32)
    tokens = H[layer][num_prefix:]
    norms = np.linalg.norm(tokens, axis=-1)
    mu, sd = float(norms.mean()), float(norms.std() + 1e-8)
    z = (norms - mu) / sd
    idx = np.where(z > z_thresh)[0]
    return {
        "norms": norms,
        "z": z,
        "outlier_idx": idx.tolist(),
        "n_outliers": int(len(idx)),
        "mean_norm": mu,
        "std_norm": sd,
        "max_z": float(z.max()) if z.size else 0.0,
    }


def effective_rank(tokens: np.ndarray) -> float:
    """Participation ratio of the token covariance spectrum — feature-space richness."""
    X = np.asarray(tokens, dtype=np.float32)
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    return float((s.sum() ** 2) / (np.sum(s ** 2) + 1e-12))


def anisotropy(tokens: np.ndarray) -> float:
    """Mean pairwise cosine similarity — high values mean a collapsed representation."""
    X = np.asarray(tokens, dtype=np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    G = Xn @ Xn.T
    n = G.shape[0]
    if n < 2:
        return 0.0
    off = (G.sum() - np.trace(G)) / (n * (n - 1))
    return float(off)
