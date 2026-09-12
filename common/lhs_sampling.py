"""Reusable Latin Hypercube Sampling helpers, decoupled from project config.

Provides a clean ``lhs(lo, hi, n, seed, d=1)`` that returns a deterministic
LHS draw. Kept for occasional use; the canonical Aballay-style parameter
grid is built in :mod:`src.shared.parameter_sampling` (full-factorial grid,
not LHS).
"""
from __future__ import annotations


import numpy as np

try:
    from scipy.stats import qmc
    _HAS_QMC = True
except Exception:  # pragma: no cover
    _HAS_QMC = False


def lhs(lo: float, hi: float, n: int, seed: int) -> np.ndarray:
    """1D LHS on [lo, hi]. Returns float64 array of shape (n,)."""
    n = int(n)
    seed = int(seed)
    if _HAS_QMC:
        sampler = qmc.LatinHypercube(d=1, seed=seed)
        pts = sampler.random(n=n).flatten()
    else:  # fallback that still stratifies
        rng = np.random.default_rng(seed)
        # Stratified-but-uniform fallback (no QMC scrambling).
        u = (np.arange(n) + rng.random(n)) / float(n)
        pts = rng.permutation(u)
    return (float(lo) + (float(hi) - float(lo)) * pts).astype(np.float64)


def lhs_nd(lo: np.ndarray, hi: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Multivariate LHS on [lo, hi] component-wise. Shape: (n, d)."""
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    if lo.shape != hi.shape:
        raise ValueError("lo and hi must have the same shape")
    d = int(lo.size)
    if _HAS_QMC:
        sampler = qmc.LatinHypercube(d=d, seed=int(seed))
        pts = sampler.random(n=int(n))
    else:  # pragma: no cover
        rng = np.random.default_rng(int(seed))
        cols = []
        for _ in range(d):
            u = (np.arange(int(n)) + rng.random(int(n))) / float(int(n))
            cols.append(rng.permutation(u))
        pts = np.stack(cols, axis=1)
    return (lo[None, :] + (hi - lo)[None, :] * pts).astype(np.float64)


__all__ = ["lhs", "lhs_nd"]
