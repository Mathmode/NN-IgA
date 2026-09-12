"""lshape Ritz-energy reference (replacement for the Aballay NGSolve / Netgen ref).

Strategy
--------
For each (sigma1, sigma2) on the evaluation set, we compute a high-fidelity
Ritz energy by running r-adapt on a fine tensor-product mesh (default
N = 256, p = 1) for a bounded number of L-BFGS-B iterations. The resulting
value is taken as ``J(u^sigma)`` and cached to disk under

    results/reference_p3/cache.npz

The reference is consistent with our own discretisation (same Greville
masking, same solver), which is important for the relative metric to
make sense as a model-vs-reference comparison.

Cluster-only by design: a 256x256 dense solve uses ~520 MB and is slow on
the laptop. The smoke pipeline uses a mini-cache with N = 32 (or 64).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common._precision import ensure_double_precision

ensure_double_precision()

import jax.numpy as jnp
import numpy as np

from src.nonparametric.r_adapt import (
    p3_effective_n_elem,
    radapt_p3_one_sample,
    theta_to_knots_lshape,
)
from src.nonparametric.solver_2d import galerkin_solve_lshape


@dataclass
class ReferenceEntry:
    sigma1: float
    sigma2: float
    j_ref: float
    n_iter: int
    converged: bool
    n_elem: int
    p: int


def compute_ritz_reference(
    sigma1: float,
    sigma2: float,
    *,
    n_elem: int = 256,
    p: int = 1,
    max_iter: int = 2000,
    use_radapt: bool = True,
) -> ReferenceEntry:
    """Compute reference J(u^sigma) for one (sigma1, sigma2).

    If ``use_radapt=False``, returns the value on the UNIFORM mesh
    (cheap, used in smoke tests).
    """
    half = int(n_elem) // 2
    if 2 * half != int(n_elem):
        raise ValueError(f"n_elem must be even; got {n_elem}")

    if not use_radapt:
        knots_u = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), p)
        n_eff = p3_effective_n_elem(int(n_elem), int(p))
        res = galerkin_solve_lshape(
            knots_u, knots_u, p, n_eff, n_eff,
            jnp.asarray(sigma1), jnp.asarray(sigma2), q_K=p + 1, q_F=2,
        )
        return ReferenceEntry(
            sigma1=float(sigma1), sigma2=float(sigma2),
            j_ref=float(res.ritz_energy),
            n_iter=0, converged=True,
            n_elem=int(n_elem), p=int(p),
        )

    result, _ = radapt_p3_one_sample(
        sigma1, sigma2, n_elem=int(n_elem), p=int(p), max_iter=int(max_iter)
    )
    return ReferenceEntry(
        sigma1=float(sigma1), sigma2=float(sigma2),
        j_ref=float(result.energy_final),
        n_iter=int(result.n_iter), converged=bool(result.converged),
        n_elem=int(n_elem), p=int(p),
    )


def build_reference_cache(
    sigma_tuples: np.ndarray,
    *,
    cache_path: Path,
    n_elem: int = 256,
    p: int = 1,
    max_iter: int = 2000,
    use_radapt: bool = True,
    force_recompute: bool = False,
    verbose: bool = True,
) -> dict:
    """Compute J_ref for every (sigma1, sigma2) in ``sigma_tuples`` and write
    a ``.npz`` cache.

    ``sigma_tuples`` is a (n, 2) array of (sigma1, sigma2) entries.
    """
    cache_path = Path(cache_path)
    sigma_tuples = np.asarray(sigma_tuples, dtype=np.float64).reshape(-1, 2)
    if cache_path.exists() and not force_recompute:
        existing = np.load(cache_path)
        # Re-use only if the (sigma1, sigma2) lists match exactly.
        if (existing["sigma_tuples"].shape == sigma_tuples.shape and
                np.allclose(existing["sigma_tuples"], sigma_tuples)):
            if verbose:
                print(f"reference cache OK: {cache_path}")
            return {
                "sigma_tuples": existing["sigma_tuples"],
                "j_ref": existing["j_ref"],
                "n_elem": int(existing["n_elem"][0]),
                "p": int(existing["p"][0]),
            }

    j_ref = np.zeros(sigma_tuples.shape[0], dtype=np.float64)
    converged_arr = np.zeros(sigma_tuples.shape[0], dtype=np.int32)
    n_iter_arr = np.zeros(sigma_tuples.shape[0], dtype=np.int32)
    for i, (s1, s2) in enumerate(sigma_tuples):
        entry = compute_ritz_reference(
            float(s1), float(s2),
            n_elem=int(n_elem), p=int(p), max_iter=int(max_iter),
            use_radapt=bool(use_radapt),
        )
        j_ref[i] = entry.j_ref
        converged_arr[i] = int(entry.converged)
        n_iter_arr[i] = int(entry.n_iter)
        if verbose and (i % 10 == 0 or i + 1 == sigma_tuples.shape[0]):
            print(
                f"  [{i + 1:>3}/{sigma_tuples.shape[0]}] "
                f"sigma=({s1:.4g}, {s2:.4g}) J_ref={entry.j_ref:.4e} "
                f"iters={entry.n_iter} ok={entry.converged}"
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        sigma_tuples=sigma_tuples,
        j_ref=j_ref,
        converged=converged_arr,
        n_iter=n_iter_arr,
        n_elem=np.asarray([int(n_elem)], dtype=np.int32),
        p=np.asarray([int(p)], dtype=np.int32),
    )
    if verbose:
        print(f"reference cache written to {cache_path}")
    return {
        "sigma_tuples": sigma_tuples,
        "j_ref": j_ref,
        "n_elem": int(n_elem),
        "p": int(p),
    }


__all__ = [
    "ReferenceEntry",
    "compute_ritz_reference",
    "build_reference_cache",
]
