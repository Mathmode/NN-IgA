"""Visualization data for the trained positional density network (Exp. 1).

``evaluate_viz_set`` writes the breakpoints/greville/knots/solution/d2u CSVs
used by the figures. The test-split summary rows are produced by
``evaluation_v2.evaluate_test_split_v2``.

`method` is one of {"uniform", "positional", "positional_corrected"}.

Memory-bounded design
---------------------
``evaluate_viz_set`` accepts a ``clear_caches_between_levels`` flag (default
True) that calls ``jax.clear_caches()`` + ``gc.collect()`` after each
refinement level N. This prevents the JAX compilation cache from growing
unbounded as the shapes change with N. Each CSV is appended to after every
level so a killed job leaves partial-but-valid files.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np

from src.config import (
    EVAL_LEVELS,
    H_MIN_SCHEDULE,
    T_SCHEDULE,
    VIZ,
)
from common.csv_utils import append_csv_dicts
from src.nonparametric.discretization import (
    element_solution_coeffs_t,
    differentiate_poly_t_wrt_x,
    solve_state,
    stiffness_quadrature_order,
)
from src.nonparametric.knots import (
    breakpoints_from_knots,
    theta_to_knots,
)
from src.nonparametric.pde import (
    d2u_exact_power,
    u_exact_power,
)
from src.parametric.corrector_v2 import correct_lbfgsb_v2
from src.parametric.evaluation_v2 import (
    _eta_from_theta,
    _theta_positional,
    _theta_uniform,
)
from src.parametric.positional_density_network import PDNParams


# -----------------------------------------------------------------------------
# Field-name constants (the canonical CSV schemas)
# -----------------------------------------------------------------------------


VIZ_FIELDNAMES: Dict[str, Tuple[str, ...]] = {
    "breakpoints": ("p", "N", "beta", "method", "breakpoint_index", "x_value"),
    "greville": ("p", "N", "beta", "method", "greville_index", "x_greville",
                 "u_h_at_greville", "u_exact_at_greville"),
    "knots": ("p", "N", "beta", "method", "knot_index", "knot_value", "multiplicity"),
    "solution": ("p", "N", "beta", "method", "x_eval", "u_exact", "u_h"),
    "second_derivative": ("p", "N", "beta", "method", "x_eval", "u_exact_dd", "u_h_dd"),
}


# -----------------------------------------------------------------------------
# Visualization data (only viz_seed)
# -----------------------------------------------------------------------------


def _viz_grid() -> np.ndarray:
    """Concatenated grid: 500 logspace [x_log_min, x_log_max] + 500 linspace [0,1]."""
    n_total = int(VIZ["x_eval_grid_size"])     # type: ignore[index]
    n_log = n_total // 2
    n_lin = n_total - n_log
    log_min = float(VIZ["x_log_min"])         # type: ignore[index]
    log_max = float(VIZ["x_log_max"])         # type: ignore[index]
    g_log = np.geomspace(log_min, log_max, n_log)
    g_lin = np.linspace(0.0, 1.0, n_lin)
    grid = np.unique(np.concatenate([g_log, g_lin]))
    grid = np.clip(grid, 0.0, 1.0)
    return grid


def _evaluate_uh_and_d2uh(theta: jnp.ndarray, *, p: int, h_min: float, beta: float, x_eval: np.ndarray):
    """Return (u_h, d2u_h) at x_eval points (numpy arrays)."""
    q_order = stiffness_quadrature_order(int(p))
    u_full, cache = solve_state(theta, int(p), int(q_order), float(h_min), float(beta))
    u_coeffs_t = element_solution_coeffs_t(u_full, cache, int(p))
    d2_coeffs_t = differentiate_poly_t_wrt_x(u_coeffs_t, cache.sizes, order=2)

    a_np = np.asarray(jax.device_get(cache.a), dtype=np.float64)
    b_np = np.asarray(jax.device_get(cache.b), dtype=np.float64)
    sizes_np = np.asarray(jax.device_get(cache.sizes), dtype=np.float64)
    u_co_np = np.asarray(jax.device_get(u_coeffs_t), dtype=np.float64)   # (n_elem, deg+1)
    d2_co_np = np.asarray(jax.device_get(d2_coeffs_t), dtype=np.float64)  # (n_elem, deg-1)

    n_elem = sizes_np.shape[0]
    deg = u_co_np.shape[1] - 1
    x_arr = np.asarray(x_eval, dtype=np.float64)

    # For each x, find its element index e (search in [a_e, b_e]).
    # Use right-side searchsorted on b_np so x in (b_{e-1}, b_e] maps to e.
    edges = np.concatenate([[a_np[0]], b_np])
    eid = np.searchsorted(edges, x_arr, side="left") - 1
    eid = np.clip(eid, 0, n_elem - 1)

    t_arr = np.zeros_like(x_arr)
    nz = sizes_np[eid] > 0.0
    t_arr[nz] = (x_arr[nz] - a_np[eid][nz]) / sizes_np[eid][nz]
    t_arr = np.clip(t_arr, 0.0, 1.0)

    u_h = np.zeros_like(x_arr)
    d2u_h = np.zeros_like(x_arr)
    for k in range(deg + 1):
        u_h += u_co_np[eid, k] * (t_arr ** k)
    for k in range(d2_co_np.shape[1]):
        d2u_h += d2_co_np[eid, k] * (t_arr ** k)
    return u_h, d2u_h


@dataclass
class VizData:
    breakpoints: List[Dict[str, object]]
    greville: List[Dict[str, object]]
    knots: List[Dict[str, object]]
    solution: List[Dict[str, object]]
    second_derivative: List[Dict[str, object]]


def _knots_with_multiplicity(knots_np: np.ndarray, p: int) -> List[Tuple[float, int]]:
    out: List[Tuple[float, int]] = []
    if knots_np.size == 0:
        return out
    cur = float(knots_np[0])
    cnt = 1
    for v in knots_np[1:]:
        v = float(v)
        if abs(v - cur) <= 1e-14:
            cnt += 1
        else:
            out.append((cur, cnt))
            cur = v
            cnt = 1
    out.append((cur, cnt))
    return out


def _greville_abscissae(knots_np: np.ndarray, p: int) -> np.ndarray:
    """Standard Greville abscissae of length n_basis = len(knots) - p - 1."""
    p = int(p)
    n_basis = int(knots_np.shape[0]) - p - 1
    out = np.zeros((n_basis,), dtype=np.float64)
    for i in range(n_basis):
        out[i] = float(np.mean(knots_np[i + 1 : i + 1 + p]))
    return out


def _basis_at_x(knots_np: np.ndarray, p: int, x: np.ndarray) -> np.ndarray:
    """Evaluate every basis function at every x — returns (n_basis, len(x)) matrix.

    Used only for the Greville u_h evaluation; small enough to run on CPU.
    """
    from common.bspline_basis import bspline_basis_local

    p = int(p)
    knots_j = jnp.asarray(knots_np, dtype=DEFAULT_DTYPE)
    x_j = jnp.asarray(x, dtype=DEFAULT_DTYPE)
    N_loc, _dN, _d2N, spans = bspline_basis_local(x_j, knots_j, p)
    N_loc_np = np.asarray(jax.device_get(N_loc), dtype=np.float64)     # (len(x), p+1)
    spans_np = np.asarray(jax.device_get(spans), dtype=np.int64)
    n_basis = int(knots_np.shape[0]) - p - 1
    out = np.zeros((n_basis, x.shape[0]), dtype=np.float64)
    for i, sp in enumerate(spans_np):
        for j in range(p + 1):
            row = sp - p + j
            if 0 <= row < n_basis:
                out[row, i] += N_loc_np[i, j]
    return out


def _theta_for_method(method: str, params: PDNParams, *, p: int, N: int, beta: float, h_min: float) -> jnp.ndarray:
    if method == "uniform":
        return _theta_uniform(int(N))
    if method == "positional":
        return _theta_positional(params, float(beta), p=int(p), N=int(N))
    if method == "positional_corrected":
        theta_pos = _theta_positional(params, float(beta), p=int(p), N=int(N))
        T = float(T_SCHEDULE[int(p)][int(N)])
        eta_init = _eta_from_theta(theta_pos, p=int(p), h_min=float(h_min), beta=float(beta))
        res = correct_lbfgsb_v2(
            theta_pos, p=int(p), N=int(N), h_min=float(h_min), beta=float(beta),
            T=T, eta_init_for_gate=float(eta_init), record_history=False,
        )
        return res.theta
    raise ValueError(f"Unknown method {method!r}")


def evaluate_viz_set(
    params: PDNParams,
    *,
    p: int,
    output_dir: Path,
    eval_levels: Iterable[int] = EVAL_LEVELS,
    viz_betas: Iterable[float] | None = None,
    clear_caches_between_levels: bool = True,
    verbose: bool = True,
    run_corrector: bool = True,
) -> Dict[str, Path]:
    """Stream the 5 visualization CSVs to ``output_dir`` (one per file).

    Files written, all with name pattern ``{name}_p{p}.csv``:
      - breakpoints  (mesh knot breakpoints)
      - greville     (Greville abscissae + u_h sample + u_exact sample)
      - knots        (full knot vector with multiplicities)
      - solution     (u_exact, u_h on a 1000-pt grid)
      - second_derivative (d2u_exact, d2u_h on the same grid)

    Memory-bounded: like ``evaluate_test_split_v2``, this clears the JAX
    cache after every refinement level. Each file is appended to per
    level so a killed job leaves valid partial files.

    Returns a dict ``{name: Path}`` of the written CSVs.
    """
    if viz_betas is None:
        viz_betas = tuple(VIZ["betas"])  # type: ignore[index]
    viz_betas = tuple(float(b) for b in viz_betas)
    eval_levels = tuple(int(N) for N in eval_levels)
    methods = (("uniform", "positional", "positional_corrected")
               if run_corrector else ("uniform", "positional"))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: Dict[str, Path] = {
        name: output_dir / f"{name}_p{int(p)}.csv" for name in VIZ_FIELDNAMES
    }
    # Wipe stale files so the first append writes the header.
    for path in csv_paths.values():
        if path.exists():
            path.unlink()

    x_eval = _viz_grid()

    for N in eval_levels:
        if verbose:
            print(
                f"  [viz] N={N} start; processing {len(viz_betas)} betas x {len(methods)} methods",
                flush=True,
            )
        t0 = time.perf_counter()
        h_min = float(H_MIN_SCHEDULE[int(p)][int(N)])

        # Per-N batches, flushed to each CSV at the end of the level.
        batches: Dict[str, List[Dict[str, object]]] = {name: [] for name in VIZ_FIELDNAMES}

        for beta in viz_betas:
            for method in methods:
                theta = _theta_for_method(method, params, p=int(p), N=int(N), beta=beta, h_min=h_min)
                knots = np.asarray(
                    jax.device_get(theta_to_knots(theta, 0.0, 1.0, int(p), h_min=h_min)),
                    dtype=np.float64,
                )
                bp = np.asarray(
                    jax.device_get(breakpoints_from_knots(jnp.asarray(knots, dtype=DEFAULT_DTYPE), int(p))),
                    dtype=np.float64,
                )
                grev = _greville_abscissae(knots, int(p))

                # IGA solve for control coefficients used at Greville sampling.
                q_order = stiffness_quadrature_order(int(p))
                u_full, _cache = solve_state(theta, int(p), int(q_order), float(h_min), beta)
                u_full_np = np.asarray(jax.device_get(u_full), dtype=np.float64)

                B_grev = _basis_at_x(knots, int(p), grev)
                u_h_grev = (B_grev.T @ u_full_np).reshape(-1)
                u_exact_grev = np.asarray(
                    jax.device_get(u_exact_power(jnp.asarray(grev, dtype=DEFAULT_DTYPE), beta=beta)),
                    dtype=np.float64,
                )

                u_h_grid, d2u_grid = _evaluate_uh_and_d2uh(
                    theta, p=int(p), h_min=float(h_min), beta=beta, x_eval=x_eval
                )
                u_exact_grid = np.asarray(
                    jax.device_get(u_exact_power(jnp.asarray(x_eval, dtype=DEFAULT_DTYPE), beta=beta)),
                    dtype=np.float64,
                )
                d2u_exact_grid = np.asarray(
                    jax.device_get(d2u_exact_power(jnp.asarray(x_eval, dtype=DEFAULT_DTYPE), beta=beta)),
                    dtype=np.float64,
                )

                for i, x in enumerate(bp):
                    batches["breakpoints"].append(
                        {"p": int(p), "N": int(N), "beta": beta, "method": method,
                         "breakpoint_index": int(i), "x_value": float(x)}
                    )
                for i, x in enumerate(grev):
                    batches["greville"].append(
                        {"p": int(p), "N": int(N), "beta": beta, "method": method,
                         "greville_index": int(i),
                         "x_greville": float(x),
                         "u_h_at_greville": float(u_h_grev[i]),
                         "u_exact_at_greville": float(u_exact_grev[i])}
                    )
                for i, (k_val, mult) in enumerate(_knots_with_multiplicity(knots, int(p))):
                    batches["knots"].append(
                        {"p": int(p), "N": int(N), "beta": beta, "method": method,
                         "knot_index": int(i), "knot_value": float(k_val), "multiplicity": int(mult)}
                    )
                for i, x in enumerate(x_eval):
                    batches["solution"].append(
                        {"p": int(p), "N": int(N), "beta": beta, "method": method,
                         "x_eval": float(x),
                         "u_exact": float(u_exact_grid[i]),
                         "u_h": float(u_h_grid[i])}
                    )
                    batches["second_derivative"].append(
                        {"p": int(p), "N": int(N), "beta": beta, "method": method,
                         "x_eval": float(x),
                         "u_exact_dd": float(d2u_exact_grid[i]),
                         "u_h_dd": float(d2u_grid[i])}
                    )

        # Flush this N's batches to the corresponding CSVs.
        n_total = 0
        for name, rows in batches.items():
            if rows:
                append_csv_dicts(csv_paths[name], rows, list(VIZ_FIELDNAMES[name]))
                n_total += len(rows)

        elapsed = time.perf_counter() - t0
        if verbose:
            print(
                f"  [viz] N={N} done in {elapsed:.1f}s ({n_total} rows total)",
                flush=True,
            )

        if clear_caches_between_levels:
            jax.clear_caches()
            gc.collect()
            if verbose:
                print(f"  [viz] cleared JAX caches after N={N}", flush=True)

    return csv_paths


__all__ = [
    "VIZ_FIELDNAMES",
    "evaluate_viz_set",
    "VizData",
]
