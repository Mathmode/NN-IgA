"""Evaluation routines for trained positional density network.

Produces:
- ``evaluate_test_split``  -> rows for `summary.csv`
- ``evaluate_viz_set``     -> writes breakpoints/greville/knots/solution/d2u CSVs

For each (p, N, beta, method) we report:
- T (saturation cap), h_min,
- iters / corrector_iters / corrector_status,
- eta, eta_rel, eta_norm,
- H1_abs, H1_rel, I_eff,
- mesh_h_min, mesh_max_h, mesh_ratio,
- hmin_active, saturated_frac, logit_max_abs,
- oscillation, grad_norm_final,
- runtime_sec.

`method` is one of {"uniform", "positional", "positional_corrected"}.

Memory-bounded design
---------------------
Both ``evaluate_test_split`` and ``evaluate_viz_set`` accept a
``clear_caches_between_levels`` flag (default True) that calls
``jax.clear_caches()`` + ``gc.collect()`` after each refinement level N.
This prevents the JAX compilation cache from growing unbounded as the
shapes change with N.

Both also support incremental CSV writes — the file is appended to after
every level so a killed job leaves a partial-but-valid CSV that can be
inspected (or, with ``eval_singular.py``, re-run from the saved checkpoint).
"""
from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    theta_to_sizes,
)
from src.nonparametric.metrics import diagnostic_metrics_from_state
from src.nonparametric.pde import (
    d2u_exact_power,
    u_exact_power,
)
from src.nonparametric.quadrature_analytic import (
    reduced_loss,
    residual_loss_terms_from_state,
)
from src.nonparametric.r_adapt import normalized_residual_estimator
from src.parametric.corrector import correct_lbfgsb
from src.parametric.positional_density_network import (
    PDNParams,
    cell_xi,
    forward_scalar,
    gauge_fix,
    saturate,
)


# -----------------------------------------------------------------------------
# Field-name constants (the canonical CSV schemas)
# -----------------------------------------------------------------------------


SUMMARY_FIELDNAMES: Tuple[str, ...] = (
    "p", "N", "beta", "seed", "method",
    "T", "h_min",
    "iters", "corrector_iters", "corrector_status",
    "eta", "eta_rel", "eta_norm",
    "H1_abs", "H1_rel", "I_eff",
    "mesh_h_min", "mesh_max_h", "mesh_ratio",
    "hmin_active", "saturated_frac", "logit_max_abs",
    "oscillation", "grad_norm_final",
    "runtime_sec",
)

VIZ_FIELDNAMES: Dict[str, Tuple[str, ...]] = {
    "breakpoints": ("p", "N", "beta", "method", "breakpoint_index", "x_value"),
    "greville": ("p", "N", "beta", "method", "greville_index", "x_greville",
                 "u_h_at_greville", "u_exact_at_greville"),
    "knots": ("p", "N", "beta", "method", "knot_index", "knot_value", "multiplicity"),
    "solution": ("p", "N", "beta", "method", "x_eval", "u_exact", "u_h"),
    "second_derivative": ("p", "N", "beta", "method", "x_eval", "u_exact_dd", "u_h_dd"),
}


# -----------------------------------------------------------------------------
# Single-sample evaluator: returns one summary-row dict
# -----------------------------------------------------------------------------


def _hmin_active(sizes_np: np.ndarray, h_min: float, *, slack: float = 1.05) -> int:
    """1 if any cell is at or below slack * h_min."""
    return int(bool(np.any(sizes_np <= slack * float(h_min))))


def _saturated_fraction(theta_np: np.ndarray, T: float, *, frac: float = 0.99) -> float:
    return float(np.mean(np.abs(theta_np) >= frac * float(T)))


def _full_h1_metrics(theta, *, p: int, h_min: float, beta: float) -> Tuple[float, float, float, float, float]:
    """Compute eta_residual, eta_rel, H1_abs, H1_rel, oscillation for one theta."""
    q_order = stiffness_quadrature_order(int(p))
    u_full, cache = solve_state(theta, int(p), int(q_order), float(h_min), float(beta))
    loss_terms = residual_loss_terms_from_state(u_full, cache, int(p), float(beta))
    diag = diagnostic_metrics_from_state(u_full, cache, int(p), float(beta))
    eta_residual = float(jnp.sqrt(jnp.maximum(2.0 * loss_terms.total_loss, 0.0)))
    eta_rel = float(normalized_residual_estimator(jnp.asarray(eta_residual), beta=float(beta)))
    h1_abs = float(diag.err_full_h1_abs)
    h1_rel = float(diag.err_full_h1_rel)
    osc_abs = float(diag.osc_abs)
    return eta_residual, eta_rel, h1_abs, h1_rel, osc_abs


def _grad_norm_final(theta, *, p: int, h_min: float, beta: float) -> float:
    q_order = stiffness_quadrature_order(int(p))
    g = jax.grad(lambda th: reduced_loss(th, int(p), int(q_order), float(h_min), beta=float(beta)))(
        jnp.asarray(theta, dtype=DEFAULT_DTYPE)
    )
    return float(jnp.linalg.norm(g))


def _theta_uniform(N: int) -> jnp.ndarray:
    """Uniform mesh: all logits zero -> sizes = 1/N."""
    return jnp.zeros((int(N),), dtype=DEFAULT_DTYPE)


def _theta_positional(params: PDNParams, beta: float, *, p: int, N: int) -> jnp.ndarray:
    xi = cell_xi(int(N))
    z = forward_scalar(params, beta, xi, int(N))
    z = gauge_fix(z)
    T = float(T_SCHEDULE[int(p)][int(N)])
    return saturate(z, T)


def _summary_row(
    *,
    p: int,
    N: int,
    beta: float,
    seed: int,
    method: str,
    T: float,
    h_min: float,
    iters: int,
    corrector_iters: int,
    corrector_status: str,
    theta: jnp.ndarray,
) -> Dict[str, object]:
    eta, eta_rel, h1_abs, h1_rel, osc = _full_h1_metrics(theta, p=p, h_min=h_min, beta=beta)
    grad_norm = _grad_norm_final(theta, p=p, h_min=h_min, beta=beta)

    sizes = np.asarray(jax.device_get(theta_to_sizes(theta, float(h_min))), dtype=np.float64)
    theta_np = np.asarray(jax.device_get(theta), dtype=np.float64).reshape(-1)
    h_min_obs = float(np.min(sizes))
    h_max_obs = float(np.max(sizes))
    ratio = h_max_obs / max(h_min_obs, 1e-300)

    i_eff = float("nan") if h1_abs <= 0.0 else float(eta / h1_abs)
    eta_norm_denom = math.sqrt(max(eta * eta + osc * osc, 0.0))
    eta_norm = float("nan") if eta_norm_denom <= 0.0 else float(eta / eta_norm_denom)

    return {
        "p": int(p),
        "N": int(N),
        "beta": float(beta),
        "seed": int(seed),
        "method": str(method),
        "T": float(T),
        "h_min": float(h_min),
        "iters": int(iters),
        "corrector_iters": int(corrector_iters),
        "corrector_status": str(corrector_status),
        "eta": float(eta),
        "eta_rel": float(eta_rel),
        "eta_norm": float(eta_norm),
        "H1_abs": float(h1_abs),
        "H1_rel": float(h1_rel),
        "I_eff": float(i_eff),
        "mesh_h_min": float(h_min_obs),
        "mesh_max_h": float(h_max_obs),
        "mesh_ratio": float(ratio),
        "hmin_active": int(_hmin_active(sizes, float(h_min))),
        "saturated_frac": float(_saturated_fraction(theta_np, float(T))),
        "logit_max_abs": float(np.max(np.abs(theta_np))),
        "oscillation": float(osc),
        "grad_norm_final": float(grad_norm),
    }


def evaluate_one(
    params: PDNParams,
    *,
    p: int,
    N: int,
    beta: float,
    seed: int,
    run_corrector: bool = True,
) -> List[Dict[str, object]]:
    """uniform + positional rows, plus positional_corrected only when
    ``run_corrector`` is True (corrector-free release sets it False — the
    L-BFGS-B corrector is then never called)."""
    h_min = float(H_MIN_SCHEDULE[int(p)][int(N)])
    T = float(T_SCHEDULE[int(p)][int(N)])
    rows: List[Dict[str, object]] = []

    # 1) Uniform baseline
    t0 = time.perf_counter()
    theta_u = _theta_uniform(int(N))
    row_u = _summary_row(
        p=p, N=N, beta=beta, seed=seed, method="uniform",
        T=T, h_min=h_min, iters=0, corrector_iters=0, corrector_status="",
        theta=theta_u,
    )
    row_u["runtime_sec"] = float(time.perf_counter() - t0)
    rows.append(row_u)

    # 2) Positional warm-start (no correction)
    t0 = time.perf_counter()
    theta_pos = _theta_positional(params, float(beta), p=int(p), N=int(N))
    row_pos = _summary_row(
        p=p, N=N, beta=beta, seed=seed, method="positional",
        T=T, h_min=h_min, iters=0, corrector_iters=0, corrector_status="",
        theta=theta_pos,
    )
    row_pos["runtime_sec"] = float(time.perf_counter() - t0)
    rows.append(row_pos)

    # 3) Positional + corrector. Skipped entirely when run_corrector is False
    # (corrector-free release): correct_lbfgsb is the slow L-BFGS-B bottleneck.
    if run_corrector:
        t0 = time.perf_counter()
        eta_init = float(row_pos["eta"])
        res = correct_lbfgsb(
            theta_pos, p=int(p), N=int(N), h_min=h_min, beta=float(beta),
            T=T, eta_init=eta_init, eta_target=eta_init,
        )
        row_corr = _summary_row(
            p=p, N=N, beta=beta, seed=seed, method="positional_corrected",
            T=T, h_min=h_min, iters=0, corrector_iters=int(res.iters),
            corrector_status=res.status, theta=res.theta,
        )
        row_corr["runtime_sec"] = float(time.perf_counter() - t0)
        rows.append(row_corr)

    return rows


def evaluate_test_split(
    params: PDNParams,
    *,
    p: int,
    seed: int,
    test_betas: np.ndarray,
    eval_levels: Iterable[int] = EVAL_LEVELS,
    incremental_csv_path: Optional[Path] = None,
    clear_caches_between_levels: bool = True,
    verbose: bool = True,
    run_corrector: bool = True,
) -> List[Dict[str, object]]:
    """Loop over (N, beta) and produce summary rows.

    Parameters
    ----------
    run_corrector : bool, default True
        When False (corrector-free release, consistent with the 2D CORR=0
        default), the L-BFGS-B corrector is never called and only the
        uniform + positional rows are produced.
    incremental_csv_path : Path, optional
        If given, append the rows of every level to this CSV file as soon as
        the level finishes. The header is written on the first call (the file
        is wiped first if it already exists). A killed job leaves a
        partial-but-valid CSV.
    clear_caches_between_levels : bool, default True
        After each level, call ``jax.clear_caches()`` and ``gc.collect()``.
        Required to keep memory bounded — without it the JAX compilation
        cache grows monotonically as N changes between levels.
    verbose : bool, default True
        Print per-level progress with timestamps. All prints use ``flush=True``
        so they appear in SLURM logs in real time.
    """
    test_betas = np.asarray(test_betas, dtype=np.float64).reshape(-1)
    eval_levels = tuple(int(N) for N in eval_levels)
    methods = (("uniform", "positional", "positional_corrected")
               if run_corrector else ("uniform", "positional"))

    # Wipe any stale CSV so the first append writes the header.
    if incremental_csv_path is not None:
        incremental_csv_path = Path(incremental_csv_path)
        if incremental_csv_path.exists():
            incremental_csv_path.unlink()

    all_rows: List[Dict[str, object]] = []
    n_betas = int(test_betas.shape[0])
    n_methods = len(methods)

    for N in eval_levels:
        if verbose:
            print(
                f"  [eval] N={N} start; processing {n_betas} betas x {n_methods} methods",
                flush=True,
            )
        t0 = time.perf_counter()
        rows_for_this_N: List[Dict[str, object]] = []
        for b in test_betas:
            rows = evaluate_one(params, p=int(p), N=int(N), beta=float(b), seed=int(seed),
                                run_corrector=run_corrector)
            rows_for_this_N.extend(rows)

        if incremental_csv_path is not None:
            append_csv_dicts(incremental_csv_path, rows_for_this_N, list(SUMMARY_FIELDNAMES))

        all_rows.extend(rows_for_this_N)
        elapsed = time.perf_counter() - t0
        if verbose:
            print(
                f"  [eval] N={N} done in {elapsed:.1f}s ({len(rows_for_this_N)} rows)",
                flush=True,
            )

        if clear_caches_between_levels:
            jax.clear_caches()
            gc.collect()
            if verbose:
                print(f"  [eval] cleared JAX caches after N={N}", flush=True)

    return all_rows


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
        eta_init, *_ = _full_h1_metrics(theta_pos, p=int(p), h_min=float(h_min), beta=float(beta))
        res = correct_lbfgsb(
            theta_pos, p=int(p), N=int(N), h_min=float(h_min), beta=float(beta),
            T=T, eta_init=float(eta_init), eta_target=float(eta_init),
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

    Memory-bounded: like ``evaluate_test_split``, this clears the JAX
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
    "SUMMARY_FIELDNAMES",
    "VIZ_FIELDNAMES",
    "evaluate_one",
    "evaluate_test_split",
    "evaluate_viz_set",
    "VizData",
]
