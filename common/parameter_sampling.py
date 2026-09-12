"""Aballay-style parameter grid construction for arctan and lshape.

Replicates the parameter distributions in Aballay et al. 2025
(JCP 545:114447) Sec. 4.3.1 and 4.3.2:

  arctan (arctangent 2D):
      alpha in [1, 20] under the "reversed base-2 logarithmic" rule
              alpha_j = alpha_min + alpha_max - 2^{beta_j},
          with beta_j equispaced in [log2(alpha_min), log2(alpha_max)].
      s1, s2 uniform in [0.1, 0.9].

  lshape (L-shape):
      sigma1, sigma2 = 10^{beta_j} with beta_j equispaced in [-1, 1].

A full-factorial Cartesian grid is built across all parameter axes (no
random sampling). The 70/30 train/test split is deterministic given the
seed; corners of the parameter cube are forced into the training set
(Aballay extrapolation guarantee).
"""
from __future__ import annotations

from typing import Dict

import numpy as np


# --------------------------------------------------------------------------
# Marginal distributions
# --------------------------------------------------------------------------


def sample_alpha_log_inv_base2(
    n_points: int = 20,
    alpha_min: float = 1.0,
    alpha_max: float = 20.0,
) -> np.ndarray:
    """Aballay's reversed base-2 logarithmic alpha distribution.

    alpha_j = alpha_min + alpha_max - 2^{beta_j}, with beta_j equispaced
    in [log2(alpha_min), log2(alpha_max)]. Biases toward high-alpha
    (singular-gradient) regime.

    First and last elements equal ``alpha_min`` and ``alpha_max``.
    """
    n = int(n_points)
    if n < 2:
        raise ValueError(f"n_points must be >= 2, got {n}")
    if alpha_min <= 0 or alpha_max <= alpha_min:
        raise ValueError(f"need 0 < alpha_min < alpha_max, got {alpha_min}, {alpha_max}")

    beta = np.linspace(
        np.log2(alpha_min), np.log2(alpha_max), n, dtype=np.float64
    )
    alpha = alpha_min + alpha_max - np.power(2.0, beta)
    # alpha is strictly DESCENDING under that formula (beta increasing).
    # Aballay's plot uses the same order. We return ascending so the
    # grid layout is intuitive, but it's just an ordering convention.
    return np.sort(alpha)


def sample_uniform_log_base10(
    n_points: int, exp_min: float, exp_max: float
) -> np.ndarray:
    """sigma_j = 10^{beta_j} with beta_j equispaced in [exp_min, exp_max]."""
    n = int(n_points)
    if n < 2:
        raise ValueError(f"n_points must be >= 2, got {n}")
    beta = np.linspace(float(exp_min), float(exp_max), n, dtype=np.float64)
    return np.power(10.0, beta)


def sample_uniform_linear(
    n_points: int, vmin: float, vmax: float
) -> np.ndarray:
    """Equispaced grid on [vmin, vmax]."""
    n = int(n_points)
    if n < 2:
        raise ValueError(f"n_points must be >= 2, got {n}")
    return np.linspace(float(vmin), float(vmax), n, dtype=np.float64)


# --------------------------------------------------------------------------
# Cartesian grid and train/test split with forced corners
# --------------------------------------------------------------------------


def cartesian_grid(*axes: np.ndarray) -> np.ndarray:
    """Build a full-factorial Cartesian grid from 1D axes.

    Returns shape (prod(n_i), d). Rows ordered lexicographically with the
    first axis being the slowest-varying.
    """
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.reshape(-1) for m in mesh], axis=-1)
    return pts.astype(np.float64)


def _corner_indices(axes: tuple[np.ndarray, ...]) -> np.ndarray:
    """Indices into the flat grid (lex order, ij meshgrid) of the 2^d corners."""
    # The corners of each axis are positions 0 and n_i - 1.
    shape = tuple(len(a) for a in axes)
    d = len(shape)
    corners = []
    # Iterate 2^d combinations.
    for mask in range(2 ** d):
        idx = []
        for k in range(d):
            bit = (mask >> k) & 1
            idx.append(shape[k] - 1 if bit else 0)
        # Flat index under "ij" lex order: i0 * (n1*n2*..) + i1 * (n2*..) + ...
        strides = []
        running = 1
        for k in range(d - 1, -1, -1):
            strides.append(running)
            running *= shape[k]
        strides = list(reversed(strides))
        flat = sum(idx[k] * strides[k] for k in range(d))
        corners.append(flat)
    return np.asarray(sorted(set(corners)), dtype=np.int64)


def split_with_forced_corners(
    grid: np.ndarray,
    axes: tuple[np.ndarray, ...],
    train_frac: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Deterministic 70/30 (or arbitrary) split that forces parameter-cube corners
    into the training set.

    Returns ``{'train': (n_train, d), 'test': (n_test, d), 'train_idx': ..., 'test_idx': ...}``.
    """
    n_total = grid.shape[0]
    n_train = int(round(float(train_frac) * float(n_total)))
    n_test = n_total - n_train
    if n_test <= 0 or n_train <= 0:
        raise ValueError(f"non-degenerate split required; got n_train={n_train}, n_test={n_test}")

    rng = np.random.default_rng(int(seed))

    corners = _corner_indices(axes)
    n_corners = int(corners.size)
    if n_corners > n_train:
        raise ValueError(
            f"#corners ({n_corners}) exceeds n_train ({n_train}); "
            f"cannot force-include all corners."
        )

    all_idx = np.arange(n_total, dtype=np.int64)
    is_corner = np.zeros(n_total, dtype=bool)
    is_corner[corners] = True
    noncorner_idx = all_idx[~is_corner]
    perm = rng.permutation(noncorner_idx)

    # Train = corners + first (n_train - n_corners) shuffled non-corners.
    train_idx = np.concatenate([corners, perm[: n_train - n_corners]])
    test_idx = perm[n_train - n_corners :]
    # Sort train and test indices so the order is reproducible regardless
    # of rng implementation drift.
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    assert train_idx.size == n_train
    assert test_idx.size == n_test
    assert np.intersect1d(train_idx, test_idx).size == 0

    return {
        "train": grid[train_idx],
        "test": grid[test_idx],
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def split_with_forced_corners_3way(
    grid: np.ndarray,
    axes: tuple[np.ndarray, ...],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Deterministic 3-way (train/val/test) split that forces the parameter-cube
    corners into the TRAIN set (unified validation protocol).

    Identical corner/shuffle logic to ``split_with_forced_corners`` but carves a
    held-out VALIDATION block (for η²-validation early stopping) between train and
    test. ``test`` is the remainder; train ∩ val ∩ test = ∅ by construction.

    Returns ``{'train','val','test','train_idx','val_idx','test_idx'}``.
    """
    n_total = grid.shape[0]
    n_train = int(round(float(train_frac) * float(n_total)))
    n_val = int(round(float(val_frac) * float(n_total)))
    n_test = n_total - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"non-degenerate 3-way split required; got "
            f"n_train={n_train}, n_val={n_val}, n_test={n_test}"
        )

    rng = np.random.default_rng(int(seed))
    corners = _corner_indices(axes)
    n_corners = int(corners.size)
    if n_corners > n_train:
        raise ValueError(
            f"#corners ({n_corners}) exceeds n_train ({n_train}); "
            f"cannot force-include all corners."
        )

    all_idx = np.arange(n_total, dtype=np.int64)
    is_corner = np.zeros(n_total, dtype=bool)
    is_corner[corners] = True
    noncorner_idx = all_idx[~is_corner]
    perm = rng.permutation(noncorner_idx)

    # Train = corners + first (n_train - n_corners) shuffled non-corners; then a
    # contiguous VAL block; the rest is TEST. Sort each for rng-drift stability.
    n_fill = n_train - n_corners
    train_idx = np.sort(np.concatenate([corners, perm[:n_fill]]))
    val_idx = np.sort(perm[n_fill : n_fill + n_val])
    test_idx = np.sort(perm[n_fill + n_val :])
    assert train_idx.size == n_train and val_idx.size == n_val and test_idx.size == n_test
    assert np.intersect1d(train_idx, val_idx).size == 0
    assert np.intersect1d(train_idx, test_idx).size == 0
    assert np.intersect1d(val_idx, test_idx).size == 0

    return {
        "train": grid[train_idx], "val": grid[val_idx], "test": grid[test_idx],
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
    }


# --------------------------------------------------------------------------
# Convenience builders
# --------------------------------------------------------------------------


def _split(grid, axes, train_frac, val_frac, seed):
    """Dispatch to the 2-way (val_frac is None) or 3-way splitter."""
    if val_frac is None:
        return split_with_forced_corners(grid, axes, train_frac, seed)
    return split_with_forced_corners_3way(grid, axes, train_frac, val_frac, seed)


def build_arctan_grid(
    *,
    n_alpha: int = 20,
    n_s1: int = 10,
    n_s2: int = 10,
    alpha_min: float = 1.0,
    alpha_max: float = 20.0,
    s_min: float = 0.1,
    s_max: float = 0.9,
    train_frac: float = 0.7,
    seed: int = 0,
    val_frac: float | None = None,
) -> Dict[str, np.ndarray]:
    """Full arctan grid (alpha, s1, s2). Default: 70/30 train/test. If ``val_frac``
    is given (val protocol), a deterministic 70/15/15 train/val/test split."""
    alpha_ax = sample_alpha_log_inv_base2(n_alpha, alpha_min, alpha_max)
    s1_ax = sample_uniform_linear(n_s1, s_min, s_max)
    s2_ax = sample_uniform_linear(n_s2, s_min, s_max)
    axes = (alpha_ax, s1_ax, s2_ax)
    grid = cartesian_grid(*axes)
    splits = _split(grid, axes, train_frac, val_frac, seed)
    splits["axes"] = {"alpha": alpha_ax, "s1": s1_ax, "s2": s2_ax}
    splits["grid"] = grid
    return splits


def build_lshape_grid(
    *,
    n_sigma1: int = 20,
    n_sigma2: int = 20,
    exp_min: float = -1.0,
    exp_max: float = 1.0,
    train_frac: float = 0.7,
    seed: int = 0,
    val_frac: float | None = None,
) -> Dict[str, np.ndarray]:
    """Full lshape grid (sigma1, sigma2). Default: 70/30 train/test. If ``val_frac``
    is given (val protocol), a deterministic 70/15/15 train/val/test split."""
    sigma1_ax = sample_uniform_log_base10(n_sigma1, exp_min, exp_max)
    sigma2_ax = sample_uniform_log_base10(n_sigma2, exp_min, exp_max)
    axes = (sigma1_ax, sigma2_ax)
    grid = cartesian_grid(*axes)
    splits = _split(grid, axes, train_frac, val_frac, seed)
    splits["axes"] = {"sigma1": sigma1_ax, "sigma2": sigma2_ax}
    splits["grid"] = grid
    return splits


def build_advdiff_grid(
    *,
    n_logeps: int = 20,
    n_b: int = 20,
    logeps_min: float = -2.0,
    logeps_max: float = -1.5,
    b_min: float = 0.5,
    b_max: float = 2.0,
    train_frac: float = 0.7,
    seed: int = 0,
    val_frac: float | None = None,
) -> Dict[str, np.ndarray]:
    """Full advdiff grid (logeps, b). Default: 70/30 train/test. If ``val_frac`` is
    given (val protocol), a deterministic 70/15/15 train/val/test split.

    ``logeps`` samples uniformly in log-base-10 of eps (so physical eps
    spans [10^logeps_min, 10^logeps_max] uniformly in log; the network
    receives logeps directly as nu[0], and the solver uses
    eps = 10^logeps). ``b`` samples uniformly in linear scale (no log).
    Corners of the (logeps, b) rectangle are forced into the training set
    (same convention as arctan/lshape).
    """
    logeps_ax = sample_uniform_linear(n_logeps, logeps_min, logeps_max)
    b_ax = sample_uniform_linear(n_b, b_min, b_max)
    axes = (logeps_ax, b_ax)
    grid = cartesian_grid(*axes)
    splits = _split(grid, axes, train_frac, val_frac, seed)
    splits["axes"] = {"logeps": logeps_ax, "b": b_ax}
    splits["grid"] = grid
    return splits


__all__ = [
    "sample_alpha_log_inv_base2",
    "sample_uniform_log_base10",
    "sample_uniform_linear",
    "cartesian_grid",
    "split_with_forced_corners",
    "split_with_forced_corners_3way",
    "build_arctan_grid",
    "build_lshape_grid",
    "build_advdiff_grid",
]
