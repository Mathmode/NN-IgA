"""Evaluation v2 — enriched CSV schema (unification target).

Extends `evaluation.py` with 26 new columns (multistart_* removed per
decision 7; eff_index added per decision 7-F; H1_semi_abs / H1_semi_rel
added so the effectivity index uses the dimensionally correct
H¹-seminorm denominator).

Schema strategy
---------------
- All 25 original columns from v1 are preserved with identical names and
  values for the (uniform, positional) rows. The positional_corrected
  rows use the new single-warm-start corrector and a global-τ gate.
- New columns: certification flags (residual_gate_pass, shape_gate_pass,
  certified, tau, kappa, accepted_raw, accepted_corrected,
  corrector_success, lbfgsb_status), estimator decomposition
  (eta_volume, eta_neumann, eta_neumann_fraction), mesh explicit
  (min_h, max_h, logits_min, logits_max), saturation breakdown
  (logit_saturation_lower, logit_saturation_upper), optimizer internals
  (lbfgsb_nit, lbfgsb_nfev, lbfgsb_ngev), runtime, history_loss,
  H¹-seminorm error (H1_semi_abs, H1_semi_rel), **and decision 7-F:
  eff_index = eta / H1_semi_abs per row**.

Effectivity-index definition
---------------------------
The residual estimator ``eta`` controls the energy norm; for these
Poisson-type problems the energy norm equals the H¹ seminorm. Therefore
the effectivity index uses the H¹ **seminorm** of the error (not the
full H¹ norm) in the denominator:

    eff_index = eta / |u - u_h|_{H¹-seminorm} = eta / H1_semi_abs

The legacy v1 ``I_eff`` column is preserved (so old readers don't
break) but is aliased to the same corrected value as ``eff_index``.

Each (p, N, beta, seed) produces exactly 3 rows: uniform, positional,
positional_corrected.

Per decision 7-B, the residual gate threshold ``τ`` is a global constant
calibrated ONCE on a held-out validation subset (see
``calibrate_tau_global`` below) and passed into
``evaluate_test_split_v2`` via the ``tau_global`` argument.
"""
from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np

from src.config import EVAL_LEVELS, H_MIN_SCHEDULE, T_SCHEDULE
from common.csv_utils import append_csv_dicts
from src.nonparametric.discretization import (
    solve_state,
    stiffness_quadrature_order,
)
from src.nonparametric.knots import theta_to_sizes
from src.nonparametric.metrics import diagnostic_metrics_from_state
from src.nonparametric.quadrature_analytic import (
    reduced_loss,
    residual_loss_terms_from_state,
)
from src.nonparametric.r_adapt import normalized_residual_estimator
from src.parametric.corrector_v2 import (
    CorrectorV2Result,
    correct_lbfgsb_v2,
)
from src.parametric.evaluation import (
    SUMMARY_FIELDNAMES as SUMMARY_FIELDNAMES_V1,
)
from src.parametric.positional_density_network import (
    PDNParams,
    cell_xi,
    forward_scalar,
    gauge_fix,
    saturate,
)


# -----------------------------------------------------------------------------
# Schema: 25 v1 + 26 v2 = 51 columns
#   (multistart_K and multistart_init removed per the unification target;
#    eff_index added per decision 7-F; H1_semi_abs / H1_semi_rel added
#    so the effectivity index can use the correct H¹-seminorm
#    denominator — see _full_h1_metrics for the rationale.)
# -----------------------------------------------------------------------------


V2_NEW_FIELDS: Tuple[str, ...] = (
    # certification (9) — decision 7-B/7-D
    "accepted_raw", "accepted_corrected",
    "residual_gate_pass", "shape_gate_pass",
    "certified", "tau", "kappa",
    "corrector_success", "lbfgsb_status",
    # estimator decomposition (3)
    "eta_volume", "eta_neumann", "eta_neumann_fraction",
    # mesh explicit (4) — `mesh_ratio` already in v1
    "min_h", "max_h", "logits_min", "logits_max",
    # saturation breakdown (2)
    "logit_saturation_lower", "logit_saturation_upper",
    # optimizer internals (3)
    "lbfgsb_nit", "lbfgsb_nfev", "lbfgsb_ngev",
    # cost and trajectory (2)
    "runtime", "history_loss",
    # H¹-seminorm error (2) — the dimensionally correct denominator for
    # the effectivity index. ``H1_semi_abs = |u - u_h|_{H¹-seminorm}``.
    "H1_semi_abs", "H1_semi_rel",
    # decision 7-F: precomputed effectivity index
    # ``eff_index = eta / H1_semi_abs`` (energy-norm-consistent).
    "eff_index",
)

SUMMARY_V2_FIELDNAMES: Tuple[str, ...] = SUMMARY_FIELDNAMES_V1 + V2_NEW_FIELDS


# -----------------------------------------------------------------------------
# Helpers reproduced from v1 (kept private to v2; do NOT alter to preserve
# bit-exact equivalence in the K=1 path)
# -----------------------------------------------------------------------------


def _hmin_active(sizes_np: np.ndarray, h_min: float, *, slack: float = 1.05) -> int:
    return int(bool(np.any(sizes_np <= slack * float(h_min))))


def _saturated_fraction_v1(theta_np: np.ndarray, T: float, *, frac: float = 0.99) -> float:
    return float(np.mean(np.abs(theta_np) >= frac * float(T)))


def _full_h1_metrics(theta, *, p: int, h_min: float, beta: float):
    """Compute eta_residual, eta_rel, H1_abs, H1_rel, H1_semi_abs, H1_semi_rel,
    oscillation, eta_volume, eta_neumann.

    Notes
    -----
    * ``H1_abs`` / ``H1_rel`` are the **full H¹ norm** of the error
      (= sqrt(L² + energy)); same as the v1 schema.
    * ``H1_semi_abs`` / ``H1_semi_rel`` are the **H¹ seminorm** of the
      error (= the energy term alone). They come from the existing
      ``err_energy_abs`` / ``err_energy_rel`` fields of
      ``PowerDiagnostics`` — no recomputation. The seminorm is the
      correct denominator for the effectivity index because the
      residual estimator ``eta`` controls the **energy norm**, which
      for these Poisson-type problems equals the H¹ seminorm.
    """
    q_order = stiffness_quadrature_order(int(p))
    u_full, cache = solve_state(theta, int(p), int(q_order), float(h_min), float(beta))
    loss_terms = residual_loss_terms_from_state(u_full, cache, int(p), float(beta))
    diag = diagnostic_metrics_from_state(u_full, cache, int(p), float(beta))
    eta_residual = float(jnp.sqrt(jnp.maximum(2.0 * loss_terms.total_loss, 0.0)))
    eta_rel = float(normalized_residual_estimator(jnp.asarray(eta_residual), beta=float(beta)))
    h1_abs = float(diag.err_full_h1_abs)
    h1_rel = float(diag.err_full_h1_rel)
    h1_semi_abs = float(diag.err_energy_abs)   # |u - u_h|_{H¹-seminorm}
    h1_semi_rel = float(diag.err_energy_rel)
    osc_abs = float(diag.osc_abs)
    eta_volume = float(jnp.sqrt(jnp.maximum(2.0 * loss_terms.volume_loss, 0.0)))
    eta_neumann = float(jnp.sqrt(jnp.maximum(2.0 * loss_terms.neumann_loss, 0.0)))
    return (eta_residual, eta_rel, h1_abs, h1_rel,
            h1_semi_abs, h1_semi_rel,
            osc_abs, eta_volume, eta_neumann)


def _grad_norm_final(theta, *, p: int, h_min: float, beta: float) -> float:
    q_order = stiffness_quadrature_order(int(p))
    g = jax.grad(lambda th: reduced_loss(th, int(p), int(q_order), float(h_min), beta=float(beta)))(
        jnp.asarray(theta, dtype=DEFAULT_DTYPE)
    )
    return float(jnp.linalg.norm(g))


def _theta_uniform(N: int) -> jnp.ndarray:
    return jnp.zeros((int(N),), dtype=DEFAULT_DTYPE)


def _theta_positional(params: PDNParams, beta: float, *, p: int, N: int) -> jnp.ndarray:
    xi = cell_xi(int(N))
    z = forward_scalar(params, beta, xi, int(N))
    z = gauge_fix(z)
    T = float(T_SCHEDULE[int(p)][int(N)])
    return saturate(z, T)


# -----------------------------------------------------------------------------
# Build a single row dict (51 columns)
# -----------------------------------------------------------------------------


def _row(
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
    runtime_sec: float,
    corrector_result: Optional[CorrectorV2Result] = None,
    raw_theta_for_raw_gate: Optional[jnp.ndarray] = None,
) -> Dict[str, object]:
    """Build a v2 row.

    ``corrector_result`` is given only for method='positional_corrected'.
    For 'uniform' and 'positional', the corrector-specific columns get
    sensible no-op values (NaN / empty / 0).

    ``raw_theta_for_raw_gate``: for ``positional_corrected``, this is the
    uncorrected theta from the PDN. Used to compute the ``accepted_raw``
    flag (would the warm-start alone have passed the gate).
    """
    (eta, eta_rel, h1_abs, h1_rel,
     h1_semi_abs, h1_semi_rel,
     osc, eta_vol, eta_neum) = _full_h1_metrics(
        theta, p=p, h_min=h_min, beta=beta
    )
    grad_norm = _grad_norm_final(theta, p=p, h_min=h_min, beta=beta)

    sizes = np.asarray(jax.device_get(theta_to_sizes(theta, float(h_min))), dtype=np.float64)
    theta_np = np.asarray(jax.device_get(theta), dtype=np.float64).reshape(-1)
    h_min_obs = float(np.min(sizes))
    h_max_obs = float(np.max(sizes))
    ratio = h_max_obs / max(h_min_obs, 1e-300)

    # Effectivity index: I_eff = eta / |u - u_h|_{H¹-seminorm}. The
    # residual estimator ``eta`` controls the energy norm, which for
    # the Poisson problems considered here equals the H¹ seminorm — so
    # the seminorm is the correct (dimensionally consistent)
    # denominator. The legacy v1 ``I_eff`` column had the WRONG
    # denominator (full H¹ norm); it is preserved here only for v1
    # schema back-compat and aliased to the corrected value.
    eff_index_value = (
        float(eta / h1_semi_abs)
        if (h1_semi_abs > 0.0 and math.isfinite(eta)) else float("nan")
    )
    i_eff = eff_index_value           # legacy alias, same formula now
    eta_norm_denom = math.sqrt(max(eta * eta + osc * osc, 0.0))
    eta_norm = float("nan") if eta_norm_denom <= 0.0 else float(eta / eta_norm_denom)

    # === New v2 fields ===
    # IMPORTANT: use the decomposition from the SAME loss_terms call that
    # defines `eta` (computed inside _full_h1_metrics). The corrector's
    # _eta_decomposition is a separate JIT trace; small numerical drift
    # between the two paths can make eta_volume² + eta_neumann² differ
    # from eta² by O(ulp). To guarantee the test
    # ``test_v2_eta_decomposition_sums`` (eta_vol² + eta_neum² == eta²)
    # passes exactly, always use the v1-path values returned by
    # _full_h1_metrics (those are bit-consistent with `eta`).
    eta_vol_v2 = float(eta_vol)
    eta_neum_v2 = float(eta_neum)
    eta_neum_frac = (
        float(eta_neum_v2 * eta_neum_v2 / max(eta_vol_v2 * eta_vol_v2 + eta_neum_v2 * eta_neum_v2, 1e-300))
        if (eta_vol_v2 + eta_neum_v2) > 0.0 else 0.0
    )

    sat_lower = float(np.mean(theta_np < -0.95 * T))
    sat_upper = float(np.mean(theta_np > 0.95 * T))

    from src.config import CORRECTOR as _CFG_CORRECTOR
    _AR_MAX = float(_CFG_CORRECTOR["shape_gate_AR_max"])

    if corrector_result is not None:
        residual_gate_pass = bool(corrector_result.residual_gate_pass)
        shape_gate_pass = bool(corrector_result.shape_gate_pass)
        certified = bool(corrector_result.certified)
        tau = float(corrector_result.tau)
        kappa = float(corrector_result.kappa)
        corrector_success = bool(corrector_result.corrector_success)
        lbfgsb_status = str(corrector_result.lbfgsb_status)
        lbfgsb_nit = int(corrector_result.lbfgsb_nit)
        lbfgsb_nfev = int(corrector_result.lbfgsb_nfev)
        lbfgsb_ngev = int(corrector_result.lbfgsb_ngev)
        history_loss_list = list(corrector_result.history_loss)
        # accepted_raw: would the uncorrected warm-start have passed the gate?
        if raw_theta_for_raw_gate is not None:
            raw_eta = _eta_from_theta(raw_theta_for_raw_gate, p=p, h_min=h_min, beta=beta)
            if not math.isnan(tau):
                raw_residual_pass = bool(raw_eta <= tau)
            else:
                raw_residual_pass = True
            raw_sizes = np.asarray(
                jax.device_get(theta_to_sizes(raw_theta_for_raw_gate, float(h_min))),
                dtype=np.float64,
            )
            raw_ratio = float(np.max(raw_sizes)) / max(float(np.min(raw_sizes)), 1e-300)
            raw_shape_pass = bool(raw_ratio <= _AR_MAX)
            accepted_raw = bool(raw_residual_pass and raw_shape_pass)
        else:
            accepted_raw = False
        accepted_corrected = bool(corrector_result.accepted)
    else:
        # uniform / positional: no corrector. Gates not applied.
        residual_gate_pass = True
        shape_gate_pass = True
        certified = True
        tau = float("nan")
        kappa = float("nan")
        corrector_success = False
        lbfgsb_status = ""
        lbfgsb_nit = 0
        lbfgsb_nfev = 0
        lbfgsb_ngev = 0
        history_loss_list = []
        accepted_raw = True
        accepted_corrected = False

    history_loss_str = ",".join(f"{x:.6e}" for x in history_loss_list)

    row: Dict[str, object] = {
        # 25 v1 columns (identical to v1 schema)
        "p": int(p), "N": int(N), "beta": float(beta), "seed": int(seed),
        "method": str(method),
        "T": float(T), "h_min": float(h_min),
        "iters": int(iters), "corrector_iters": int(corrector_iters),
        "corrector_status": str(corrector_status),
        "eta": float(eta), "eta_rel": float(eta_rel), "eta_norm": float(eta_norm),
        "H1_abs": float(h1_abs), "H1_rel": float(h1_rel), "I_eff": float(i_eff),
        "mesh_h_min": float(h_min_obs), "mesh_max_h": float(h_max_obs),
        "mesh_ratio": float(ratio),
        "hmin_active": int(_hmin_active(sizes, float(h_min))),
        "saturated_frac": float(_saturated_fraction_v1(theta_np, float(T))),
        "logit_max_abs": float(np.max(np.abs(theta_np))),
        "oscillation": float(osc), "grad_norm_final": float(grad_norm),
        "runtime_sec": float(runtime_sec),
        # New v2 columns
        "accepted_raw": int(bool(accepted_raw)),
        "accepted_corrected": int(bool(accepted_corrected)),
        "residual_gate_pass": int(bool(residual_gate_pass)),
        "shape_gate_pass": int(bool(shape_gate_pass)),
        "certified": int(bool(certified)),
        "tau": float(tau) if not math.isnan(tau) else "",
        "kappa": float(kappa) if not math.isnan(kappa) else "",
        "corrector_success": int(bool(corrector_success)),
        "lbfgsb_status": str(lbfgsb_status),
        "eta_volume": float(eta_vol_v2),
        "eta_neumann": float(eta_neum_v2),
        "eta_neumann_fraction": float(eta_neum_frac),
        "min_h": float(h_min_obs),
        "max_h": float(h_max_obs),
        "logits_min": float(np.min(theta_np)),
        "logits_max": float(np.max(theta_np)),
        "logit_saturation_lower": float(sat_lower),
        "logit_saturation_upper": float(sat_upper),
        "lbfgsb_nit": int(lbfgsb_nit),
        "lbfgsb_nfev": int(lbfgsb_nfev),
        "lbfgsb_ngev": int(lbfgsb_ngev),
        "runtime": float(runtime_sec),
        "history_loss": history_loss_str,
        # H¹-seminorm error of the solution (|u - u_h|_{H¹-semi}). The
        # residual estimator ``eta`` controls the energy norm — for these
        # Poisson problems, the H¹ seminorm — so this is the correct
        # denominator for the effectivity index.
        "H1_semi_abs": float(h1_semi_abs),
        "H1_semi_rel": float(h1_semi_rel),
        # Decision 7-F: precomputed effectivity index per row.
        # eff_index = eta / H1_semi_abs; NaN if H1_semi_abs is non-positive
        # or eta is non-finite. The legacy v1 ``I_eff`` column (above) is
        # now aliased to this same value.
        "eff_index": eff_index_value,
    }
    return row


def _eta_from_theta(theta: jnp.ndarray, *, p: int, h_min: float, beta: float) -> float:
    """eta_residual from theta (used for the accepted_raw gate)."""
    q_order = stiffness_quadrature_order(int(p))
    u_full, cache = solve_state(theta, int(p), int(q_order), float(h_min), float(beta))
    terms = residual_loss_terms_from_state(u_full, cache, int(p), float(beta))
    return float(jnp.sqrt(jnp.maximum(2.0 * terms.total_loss, 0.0)))


# -----------------------------------------------------------------------------
# evaluate_one_v2: 3 rows for K=1; 4 rows for K=[1,2]
# -----------------------------------------------------------------------------


def evaluate_one_v2(
    params: PDNParams,
    *,
    p: int,
    N: int,
    beta: float,
    seed: int,
    tau_global: Optional[float] = None,
    record_history: bool = True,
    corrector_budget: Optional[int] = None,
    run_corrector: bool = True,
) -> List[Dict[str, object]]:
    """Build the rows for one (p, N, beta, seed): (uniform, positional) always,
    plus positional_corrected only when ``run_corrector`` is True.

    Per decision 7-B, ``tau_global`` is the global residual-gate threshold
    calibrated once after training (see ``calibrate_tau_global``); pass it
    in here. ``None`` disables the residual gate.

    ``corrector_budget``: overrides L-BFGS-B maxiter for the corrected row;
    default uses ``config.CORRECTOR['budget']``.

    ``run_corrector``: when False (the corrector-free release, consistent with
    the 2D ``CORR=0`` default), the expensive L-BFGS-B corrector
    (``correct_lbfgsb_v2``) is NOT called at all and only the uniform +
    positional rows are emitted (zero added cost). Default True keeps the
    legacy 3-row behaviour.
    """
    h_min = float(H_MIN_SCHEDULE[int(p)][int(N)])
    T = float(T_SCHEDULE[int(p)][int(N)])
    rows: List[Dict[str, object]] = []

    # 1) Uniform.
    t0 = time.perf_counter()
    theta_u = _theta_uniform(int(N))
    rows.append(_row(
        p=p, N=N, beta=beta, seed=seed, method="uniform",
        T=T, h_min=h_min, iters=0, corrector_iters=0, corrector_status="",
        theta=theta_u, runtime_sec=float(time.perf_counter() - t0),
        corrector_result=None,
    ))

    # 2) Positional.
    t0 = time.perf_counter()
    theta_pos = _theta_positional(params, float(beta), p=int(p), N=int(N))
    rows.append(_row(
        p=p, N=N, beta=beta, seed=seed, method="positional",
        T=T, h_min=h_min, iters=0, corrector_iters=0, corrector_status="",
        theta=theta_pos, runtime_sec=float(time.perf_counter() - t0),
        corrector_result=None,
    ))

    # 3) Positional corrected — single warm-start (decision: drop multistart).
    # Skipped entirely when run_corrector is False (corrector-free release):
    # the L-BFGS-B call below is the slow bottleneck, so we do not run it.
    if run_corrector:
        raw_eta = _eta_from_theta(theta_pos, p=int(p), h_min=h_min, beta=float(beta))
        t0 = time.perf_counter()
        res = correct_lbfgsb_v2(
            theta_pos,
            p=int(p), N=int(N), h_min=h_min, beta=float(beta),
            T=T,
            budget=(int(corrector_budget)
                    if corrector_budget is not None else None),
            eta_init_for_gate=raw_eta,
            tau_global=(float(tau_global) if tau_global is not None else None),
            record_history=bool(record_history),
        )
        rows.append(_row(
            p=p, N=N, beta=beta, seed=seed, method="positional_corrected",
            T=T, h_min=h_min, iters=0, corrector_iters=int(res.iters),
            corrector_status=res.status,
            theta=res.theta, runtime_sec=float(time.perf_counter() - t0),
            corrector_result=res,
            raw_theta_for_raw_gate=theta_pos,
        ))

    return rows


# -----------------------------------------------------------------------------
# evaluate_test_split_v2
# -----------------------------------------------------------------------------


def calibrate_tau_global(
    params: PDNParams,
    *,
    p: int,
    calibration_betas: np.ndarray,
    N: int,
    kappa: Optional[float] = None,
) -> float:
    """Calibrate the global residual-gate threshold τ (decision 7-B).

    ``τ = κ · median_{β ∈ V} r_η(θ_φ(β); β)`` where
    ``r_η = η(θ_φ(β); β) / η(θ_unif(β); β)`` is the per-sample residual
    ratio over the validation subset ``V``. Use ``N = ANCHOR_N`` (the
    deepest training level) for the calibration; the resulting τ is
    applied at all evaluation N.

    Parameters
    ----------
    params : PDN parameters (trained).
    calibration_betas : 1D array of β values held out for τ calibration.
    N : refinement level at which η and η_unif are evaluated.
    kappa : multiplier on the median. Defaults to
        ``CORRECTOR["gate_kappa"]`` from ``src.config``.
    """
    from src.config import CORRECTOR
    kappa_val = float(CORRECTOR["gate_kappa"]) if kappa is None else float(kappa)
    h_min = float(H_MIN_SCHEDULE[int(p)][int(N)])
    betas = np.asarray(calibration_betas, dtype=np.float64).reshape(-1)
    ratios: List[float] = []
    for b in betas:
        theta_pos = _theta_positional(params, float(b), p=int(p), N=int(N))
        theta_unif = _theta_uniform(int(N))
        eta_pos = _eta_from_theta(theta_pos, p=int(p), h_min=h_min, beta=float(b))
        eta_unif = _eta_from_theta(theta_unif, p=int(p), h_min=h_min, beta=float(b))
        if eta_unif > 0.0:
            ratios.append(float(eta_pos / eta_unif))
    if not ratios:
        return float("nan")
    return float(kappa_val * float(np.median(np.asarray(ratios))))


def evaluate_test_split_v2(
    params: PDNParams,
    *,
    p: int,
    seed: int,
    test_betas: np.ndarray,
    eval_levels: Iterable[int] = EVAL_LEVELS,
    tau_global: Optional[float] = None,
    incremental_csv_path: Optional[Path] = None,
    clear_caches_between_levels: bool = True,
    verbose: bool = True,
    record_history: bool = True,
    corrector_budget: Optional[int] = None,
    run_corrector: bool = True,
) -> List[Dict[str, object]]:
    """Loop over (N, beta) and produce v2 rows.

    Each (N, beta, seed) produces 2 rows (uniform, positional) when
    ``run_corrector`` is False, or 3 rows (+ positional_corrected) when True.
    Multi-start was removed per the unification target; statistical spread
    comes from the SEED ensemble.

    Per decision 7-B, ``tau_global`` is the global residual-gate
    threshold calibrated once by ``calibrate_tau_global`` and reused
    across every (N, β). ``None`` disables the residual gate.

    ``corrector_budget``: if not None, overrides the L-BFGS-B maxiter
    for the corrected row. When None (default), uses
    ``config.CORRECTOR['budget']``.

    ``run_corrector``: when False (corrector-free release, consistent with
    the 2D ``CORR=0`` default), the L-BFGS-B corrector is never called and
    only the uniform + positional rows are written (zero added cost).
    """
    test_betas = np.asarray(test_betas, dtype=np.float64).reshape(-1)
    eval_levels = tuple(int(N) for N in eval_levels)

    if incremental_csv_path is not None:
        incremental_csv_path = Path(incremental_csv_path)
        if incremental_csv_path.exists():
            incremental_csv_path.unlink()

    all_rows: List[Dict[str, object]] = []
    n_betas = int(test_betas.shape[0])

    rows_per_sample = 3 if run_corrector else 2
    for N in eval_levels:
        if verbose:
            print(
                f"  [eval_v2] N={N} start; processing {n_betas} betas, "
                f"{rows_per_sample} rows per sample "
                f"(run_corrector={run_corrector}, tau_global={tau_global!r})",
                flush=True,
            )
        t0 = time.perf_counter()
        rows_for_this_N: List[Dict[str, object]] = []
        for b in test_betas:
            rows = evaluate_one_v2(
                params, p=int(p), N=int(N), beta=float(b), seed=int(seed),
                tau_global=tau_global,
                record_history=record_history,
                corrector_budget=corrector_budget,
                run_corrector=run_corrector,
            )
            rows_for_this_N.extend(rows)

        if incremental_csv_path is not None:
            append_csv_dicts(incremental_csv_path, rows_for_this_N, list(SUMMARY_V2_FIELDNAMES))

        all_rows.extend(rows_for_this_N)
        elapsed = time.perf_counter() - t0
        if verbose:
            print(
                f"  [eval_v2] N={N} done in {elapsed:.1f}s ({len(rows_for_this_N)} rows)",
                flush=True,
            )

        if clear_caches_between_levels:
            jax.clear_caches()
            gc.collect()
            if verbose:
                print(f"  [eval_v2] cleared JAX caches after N={N}", flush=True)

    return all_rows


__all__ = [
    "SUMMARY_V2_FIELDNAMES",
    "V2_NEW_FIELDS",
    "calibrate_tau_global",
    "evaluate_one_v2",
    "evaluate_test_split_v2",
]
