"""Eq-26 (manuscript) loss plumbing shared by the three 2D experiments.

Provides:

  * ``eta_unif_denoms``      — η²(θ_unif^{(N)}; ν) for a batch of parameters
                               (chunked vmap; one uniform solve per (ν, N)).
  * ``make_denoms_fn``       — per-level denominator callback for the
                               continuation loops (cached per N; train + val).
  * ``make_h1_metric_factory_arctan / _p4`` — host-side H¹-seminorm relative
                               error of the predicted-mesh solve vs. the
                               ANALYTIC exact solution (full val set).
  * ``make_h1_metric_factory_lshape``       — same vs. a stored IGA self-
                               reference cache (immersed IGA self-reference, p=5) on a FIXED val
                               subset (the direct metric is grid-heavy).

The h1 metrics consume the (u_h, knots) stacks returned by the combined
``make_val_forward_*`` factories in ``training``/``training_advdiff`` — i.e. the
SAME solve the val loss already pays for; no duplicate solves.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np

from common.h1_seminorm_2d import h1_seminorm_sq_2d
from src.nonparametric.eta_estimator_2d import (
    eta_uniform_squared_arctan,
    eta_uniform_squared_lshape,
)
from src.nonparametric.advdiff.eta_estimator_advdiff import (
    eta_uniform_squared_advdiff,
)

_ETA_UNIF = {
    "arctan": eta_uniform_squared_arctan,
    "lshape": eta_uniform_squared_lshape,
    "advdiff": eta_uniform_squared_advdiff,
}


def eta_unif_denoms(
    params_arr: np.ndarray,
    N: int,
    p: int,
    *,
    kind: str,
    q_est: int,
    chunk: int = 32,
) -> np.ndarray:
    """η²(θ_unif^{(N)}; ν_i) for every row of ``params_arr`` (chunked vmap).

    One uniform-mesh Galerkin solve per ν at this level; ``chunk`` bounds the
    batched-solve memory (the K factor of a dense N×N solve × chunk).
    """
    fn = _ETA_UNIF[str(kind)]
    f = jax.jit(jax.vmap(lambda s: fn(s, int(N), int(p), q_est=int(q_est))))
    outs = []
    arr = np.asarray(params_arr, dtype=np.float64)
    for i in range(0, arr.shape[0], int(chunk)):
        outs.append(np.asarray(f(jnp.asarray(arr[i : i + int(chunk)],
                                             dtype=DEFAULT_DTYPE))))
    return np.concatenate(outs) if outs else np.zeros((0,))


def make_denoms_fn(
    train_params: np.ndarray,
    val_params: Optional[np.ndarray],
    *,
    kind: str,
    p: int,
    q_est: int,
    chunk: int = 32,
) -> Callable[[int], Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Continuation callback: ``denoms_fn(N) -> (train_denoms, val_denoms)``
    with the eq-26 same-level uniform normalisation, cached per N."""
    cache: Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]] = {}

    def denoms_fn(N: int):
        N = int(N)
        if N not in cache:
            tr = eta_unif_denoms(train_params, N, p, kind=kind, q_est=q_est, chunk=chunk)
            va = (eta_unif_denoms(val_params, N, p, kind=kind, q_est=q_est, chunk=chunk)
                  if val_params is not None else None)
            cache[N] = (tr, va)
        return cache[N]

    return denoms_fn


# --------------------------------------------------------------------------
# H¹-error metric factories (consume the val_forward (u_h, knots) stacks).
# --------------------------------------------------------------------------


def make_h1_metric_factory_arctan(p: int, *, metric_denoms: np.ndarray,
                              quad_points_per_dim: int = 20):
    """arctan (arctangent): h1_rel(σ) = sqrt(∫|∇u_h−∇u*|² / ‖∇u*‖²_analytic).

    ``metric_denoms`` = analytic |u^σ*|²_{H¹} per VAL σ (the deprecated
    training denominators, repurposed as the metric denominator)."""
    from src.nonparametric.arctan.pde import grad_u_sigma

    dens = np.asarray(metric_denoms, dtype=np.float64)

    def factory(N: int):
        def h1_fn(u_hs, kxs, kys, sigmas):
            out = np.full(sigmas.shape[0], np.nan)
            for i in range(sigmas.shape[0]):
                a, s1, s2 = (float(sigmas[i][0]), float(sigmas[i][1]),
                             float(sigmas[i][2]))
                gfn = lambda x, y: grad_u_sigma(x, y, a, s1, s2)
                num = float(h1_seminorm_sq_2d(
                    u_hs[i], kxs[i], kys[i], int(p), gfn,
                    quad_points_per_dim=int(quad_points_per_dim),
                ))
                if dens[i] > 0.0 and np.isfinite(num):
                    out[i] = math.sqrt(max(num, 0.0) / dens[i])
            return out
        return h1_fn

    return factory


def make_h1_metric_factory_advdiff(p: int, *, metric_denoms: np.ndarray,
                              quad_points_per_dim: int = 20):
    """advdiff (advection–diffusion) analogue of ``make_h1_metric_factory_arctan``."""
    from src.nonparametric.advdiff.pde import grad_u_exact_advdiff

    dens = np.asarray(metric_denoms, dtype=np.float64)

    def factory(N: int):
        def h1_fn(u_hs, kxs, kys, nus):
            out = np.full(nus.shape[0], np.nan)
            for i in range(nus.shape[0]):
                eps = float(10.0 ** float(nus[i][0]))
                b = float(nus[i][1])
                gfn = lambda x, y: grad_u_exact_advdiff(x, y, eps, b)
                num = float(h1_seminorm_sq_2d(
                    u_hs[i], kxs[i], kys[i], int(p), gfn,
                    quad_points_per_dim=int(quad_points_per_dim),
                ))
                if dens[i] > 0.0 and np.isfinite(num):
                    out[i] = math.sqrt(max(num, 0.0) / dens[i])
            return out
        return h1_fn

    return factory


def make_h1_metric_factory_lshape(p: int, *, ref_cache_path: Optional[str],
                              n_val: int, subset_size: int = 8,
                              subset_seed: int = 0):
    """L-shape: direct σ-weighted H¹ error vs. the immersed IGA self-reference
    (degree p=5, which cancels the C⁰ corner cut and yields the manuscript's
    direct error ≲ 2e-5), on a FIXED val subset (the direct metric evaluates
    both gradients on an 800²-point grid — too heavy for the full val set per
    epoch). Subset indices are drawn once with ``subset_seed``.

    ``ref_cache_path`` defaults to ``<repo>/references/reference_lshape.pkl``;
    place that file there (or pass an explicit path) — no IGA_REF env var needed.
    Returns ``(factory, subset_idx)``; ``factory`` is None when the cache is
    missing/unreadable (the history column is then NaN)."""
    from src.parametric.evaluation_v2 import _direct_h1_error_lshape, _lookup_ref_entry

    if ref_cache_path is None:
        ref_cache_path = str(Path(__file__).resolve().parents[3]
                             / "references" / "reference_lshape.pkl")
    path = Path(ref_cache_path)
    if not path.exists():
        print(f"[eq26] WARNING: L-shape reference not found ({path}); "
              f"h1_rel_val_median will be NaN. Place references/reference_lshape.pkl.",
              flush=True)
        return None, np.array([], dtype=int)
    with open(path, "rb") as fh:
        ref_cache = pickle.load(fh)

    rng = np.random.default_rng(int(subset_seed))
    k = min(int(subset_size), int(n_val))
    subset_idx = np.sort(rng.choice(int(n_val), size=k, replace=False))

    def factory(N: int):
        def h1_fn(u_hs, kxs, kys, sigmas):
            out = np.full(sigmas.shape[0], np.nan)
            for i in subset_idx:
                s1, s2 = float(sigmas[i][0]), float(sigmas[i][1])
                entry = _lookup_ref_entry(ref_cache, s1, s2)
                if entry is None:
                    continue
                rel, _, _, _ = _direct_h1_error_lshape(
                    np.asarray(u_hs[i]), np.asarray(kxs[i]), np.asarray(kys[i]),
                    int(p), s1, s2, entry,
                )
                out[i] = rel
            return out
        return h1_fn

    return factory, subset_idx


__all__ = [
    "eta_unif_denoms",
    "make_denoms_fn",
    "make_h1_metric_factory_arctan",
    "make_h1_metric_factory_lshape",
    "make_h1_metric_factory_advdiff",
]
