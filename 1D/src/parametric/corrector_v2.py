"""Online L-BFGS-B corrector v2 (single warm-start; multistart removed).

Per the unification target (decision: drop multi-start), this module now
runs ONE L-BFGS-B from the network warm-start, no perturbation, no
alternate-init. Statistical spread is reported from the existing seeds.

Per decision 7-B, the residual gate uses a **global** threshold
``τ = κ · median r_η`` calibrated on a held-out validation subset and
passed in via ``tau_global``; the per-sample ``κ · η_init`` form has
been retired.

Per decision 7-D, the shape gate is the per-axis box constraint
``max_h / min_h ≤ γ_sh`` (1D has a single axis).

Extras vs the original ``corrector.py``:

- **Loss trajectory recording** (history_loss): per-iteration loss values
  via scipy callback. Cap at 200 entries (decimate if more iters).
- **Estimator decomposition** (eta_volume, eta_neumann) returned in the
  result for downstream CSV columns.
- **LBFGS-B diagnostics**: nit, nfev, njev, status, success flag.
- **Logit saturation breakdown**: fraction below -0.95T, above +0.95T,
  and total saturation.
- **Acceptance flags**: residual_gate_pass, shape_gate_pass, certified.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from src.config import CORRECTOR, T_SCHEDULE
from src.nonparametric.discretization import (
    solve_state,
    stiffness_quadrature_order,
)
from src.nonparametric.knots import theta_to_sizes
from src.nonparametric.quadrature_analytic import (
    reduced_loss,
    residual_loss_terms_from_state,
)

Array = jnp.ndarray

HISTORY_LOSS_CAP = 200


@dataclass
class CorrectorV2Result:
    """Output of correct_lbfgsb_v2.

    Fields cover everything correct_lbfgsb returns + 25 new diagnostics
    required by evaluate_v2's enriched schema.
    """

    # Original (12)
    theta: Array
    loss_init: float
    loss_final: float
    iters: int
    fevals: int
    status: str
    saturated_frac: float
    aspect_ratio: float
    accepted: bool
    elapsed_sec: float
    grad_norm_init: float
    grad_norm_final: float

    # Estimator decomposition (3)
    eta_volume: float = 0.0
    eta_neumann: float = 0.0
    eta_neumann_fraction: float = 0.0

    # LBFGS-B counters (3)
    lbfgsb_nit: int = 0
    lbfgsb_nfev: int = 0
    lbfgsb_ngev: int = 0
    lbfgsb_status: str = ""
    corrector_success: bool = False

    # Logit saturation breakdown (2)
    logit_saturation_lower: float = 0.0
    logit_saturation_upper: float = 0.0

    # Logit summary (2)
    logits_min: float = 0.0
    logits_max: float = 0.0

    # Acceptance gates (4)
    residual_gate_pass: bool = True
    shape_gate_pass: bool = True
    certified: bool = True
    tau: float = float("nan")
    kappa: float = 1.5

    # Loss trajectory (1)
    history_loss: List[float] = field(default_factory=list)

    extras: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _make_loss_grad(p: int, h_min: float, beta: float):
    """JIT-compiled loss + grad for the residual loss at fixed (p, h_min, beta)."""
    q_order = stiffness_quadrature_order(int(p))

    def loss_fn(theta: Array) -> Array:
        return reduced_loss(theta, int(p), int(q_order), float(h_min), beta=float(beta))

    grad_fn = jax.grad(loss_fn)
    return loss_fn, grad_fn


def _eta_decomposition(theta: Array, p: int, h_min: float, beta: float) -> Tuple[float, float]:
    """Return (eta_volume, eta_neumann) such that eta² = eta_volume² + eta_neumann².

    Uses the same residual_loss_terms_from_state path as the metrics in
    evaluation.py, so the decomposition is exact (not approximated).
    """
    q_order = stiffness_quadrature_order(int(p))
    u_full, cache = solve_state(theta, int(p), int(q_order), float(h_min), float(beta))
    terms = residual_loss_terms_from_state(u_full, cache, int(p), float(beta))
    # terms.total_loss = 0.5 * eta_total^2, terms.volume_loss = 0.5 * eta_volume^2,
    # terms.neumann_loss = 0.5 * eta_neumann^2 (the residual loss factors as 0.5*eta²).
    eta_volume = float(jnp.sqrt(jnp.maximum(2.0 * terms.volume_loss, 0.0)))
    eta_neumann = float(jnp.sqrt(jnp.maximum(2.0 * terms.neumann_loss, 0.0)))
    return eta_volume, eta_neumann


def _saturated_fraction(theta: np.ndarray, T: float, *, frac: float = 0.99) -> float:
    """Fraction of components with |theta_i| >= frac * T (same as v1)."""
    return float(np.mean(np.abs(theta) >= frac * float(T)))


def _saturation_breakdown(theta: np.ndarray, T: float, *, frac: float = 0.95) -> Tuple[float, float]:
    """Return (frac_lower, frac_upper) of logits beyond ±frac·T (not absolute)."""
    T_thr = frac * float(T)
    lower = float(np.mean(theta < -T_thr))
    upper = float(np.mean(theta > T_thr))
    return lower, upper


def _aspect_ratio(theta: Array, h_min: float) -> float:
    sizes = np.asarray(jax.device_get(theta_to_sizes(theta, float(h_min))), dtype=np.float64)
    h_max = float(np.max(sizes))
    h_min_obs = float(np.min(sizes))
    return h_max / max(h_min_obs, 1e-300)


class _LossRecorder:
    """Scipy callback: records f(x_k) per iteration up to a cap.

    Costs one extra evaluation of f per L-BFGS-B iteration because scipy
    doesn't expose f in the callback. Acceptable given budget=80.
    """

    def __init__(self, loss_fn_np, cap: int = HISTORY_LOSS_CAP):
        self.loss_fn_np = loss_fn_np
        self.cap = int(cap)
        self.values: List[float] = []

    def __call__(self, xk):
        if len(self.values) < self.cap:
            try:
                f = float(self.loss_fn_np(xk))
            except Exception:
                f = float("nan")
            self.values.append(f)
        return False  # signal to scipy: do not stop

    def finalize(self) -> List[float]:
        """Return the recorded values, decimated if more than cap."""
        if len(self.values) <= self.cap:
            return list(self.values)
        # decimate uniformly
        idx = np.linspace(0, len(self.values) - 1, self.cap).astype(int)
        return [self.values[i] for i in idx]


# -----------------------------------------------------------------------------
# Single L-BFGS-B run (refactored)
# -----------------------------------------------------------------------------


def _single_lbfgsb(
    theta_init_np: np.ndarray,
    *,
    loss_fn,
    grad_fn,
    T: float,
    budget: int,
    ftol: float,
    gtol: float,
    record_history: bool,
) -> dict:
    """Run one L-BFGS-B from a given start. Returns a dict with diagnostics."""
    # Clamp to box (defensive against numerical creep).
    theta_init_np = np.clip(theta_init_np, -T, T).astype(np.float64)

    def np_loss_only(theta_np: np.ndarray) -> float:
        th = jnp.asarray(theta_np, dtype=DEFAULT_DTYPE)
        return float(loss_fn(th))

    def np_loss_grad(theta_np: np.ndarray) -> Tuple[float, np.ndarray]:
        th = jnp.asarray(theta_np, dtype=DEFAULT_DTYPE)
        l = float(loss_fn(th))
        g = np.asarray(jax.device_get(grad_fn(th)), dtype=np.float64).reshape(-1)
        return l, g

    loss_init, grad_init = np_loss_grad(theta_init_np)
    grad_norm_init = float(np.linalg.norm(grad_init))

    recorder = _LossRecorder(np_loss_only) if record_history else None
    callback = recorder if recorder is not None else None

    bounds = [(-T, T)] * theta_init_np.shape[0]

    t0 = time.perf_counter()
    res = minimize(
        np_loss_grad,
        theta_init_np,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": int(budget),
            "ftol": float(ftol),
            "gtol": float(gtol),
            "disp": False,
        },
    )
    elapsed = time.perf_counter() - t0

    theta_final_np = np.clip(np.asarray(res.x, dtype=np.float64), -T, T)
    theta_final = jnp.asarray(theta_final_np, dtype=DEFAULT_DTYPE)
    loss_final = float(res.fun)
    grad_final_arr = np.asarray(jax.device_get(grad_fn(theta_final)), dtype=np.float64).reshape(-1)
    grad_norm_final = float(np.linalg.norm(grad_final_arr))

    return {
        "theta_np": theta_final_np,
        "theta_jax": theta_final,
        "loss_init": float(loss_init),
        "loss_final": float(loss_final),
        "grad_norm_init": grad_norm_init,
        "grad_norm_final": grad_norm_final,
        "iters": int(res.nit),
        "fevals": int(res.nfev),
        "ngevs": int(getattr(res, "njev", 0)),
        "status_int": int(getattr(res, "status", 0)),
        "status_msg": str(res.message),
        "success": bool(res.success),
        "elapsed_sec": float(elapsed),
        "history_loss": recorder.finalize() if recorder is not None else [],
    }


# -----------------------------------------------------------------------------
# Public API: single warm-start L-BFGS-B (multistart removed by decision 7)
# -----------------------------------------------------------------------------


def correct_lbfgsb_v2(
    theta_init: Array,
    *,
    p: int,
    N: int,
    h_min: float,
    beta: float,
    T: Optional[float] = None,
    budget: Optional[int] = None,
    ftol: Optional[float] = None,
    gtol: Optional[float] = None,
    eta_init_for_gate: Optional[float] = None,
    eta_target_for_gate: Optional[float] = None,
    tau_global: Optional[float] = None,
    record_history: bool = True,
) -> CorrectorV2Result:
    """Box-constrained L-BFGS-B from a single warm-start θ_init.

    Per decision 7-B, the residual gate uses a **global** threshold
    ``tau_global`` (calibrated once after training as
    ``κ · median_{ν∈V} r_η``); pass it in here. If ``tau_global is None``
    the residual gate is reported as ``residual_gate_pass=True`` and the
    threshold column is NaN — useful for ungated calls.

    Per decision 7 (drop multi-start), the multi-start branch has been
    removed. Statistical spread should come from the existing seeds, not
    from re-running the corrector with perturbed inits.

    ``eta_target_for_gate`` is retained as a kept-for-callers parameter
    but no longer affects the gate logic (kept for back-compat with
    older eval scripts that still pass it).
    """
    T_val = float(T_SCHEDULE[int(p)][int(N)]) if T is None else float(T)
    budget_val = int(CORRECTOR["budget"]) if budget is None else int(budget)
    ftol_val = float(CORRECTOR["ftol"]) if ftol is None else float(ftol)
    gtol_val = float(CORRECTOR["gtol"]) if gtol is None else float(gtol)

    loss_fn, grad_fn = _make_loss_grad(int(p), float(h_min), float(beta))

    theta_init_np = np.asarray(jax.device_get(theta_init), dtype=np.float64).reshape(-1)
    theta_init_np = np.clip(theta_init_np, -T_val, T_val)

    run = _single_lbfgsb(
        theta_init_np,
        loss_fn=loss_fn, grad_fn=grad_fn, T=T_val,
        budget=budget_val, ftol=ftol_val, gtol=gtol_val,
        record_history=record_history,
    )

    primary = run
    total_elapsed = float(run["elapsed_sec"])

    # Compute extended diagnostics on the chosen theta.
    theta_final = run["theta_jax"]
    theta_final_np = run["theta_np"]
    loss_final = run["loss_final"]

    eta_volume, eta_neumann = _eta_decomposition(theta_final, int(p), float(h_min), float(beta))
    eta_total_sq = eta_volume * eta_volume + eta_neumann * eta_neumann
    eta_neumann_fraction = (
        float(eta_neumann * eta_neumann / eta_total_sq) if eta_total_sq > 0.0 else 0.0
    )

    saturated = _saturated_fraction(theta_final_np, T_val)
    sat_lower, sat_upper = _saturation_breakdown(theta_final_np, T_val, frac=0.95)

    ar = _aspect_ratio(theta_final, float(h_min))

    # Acceptance gates (decision 7-B: global τ; decision 7-D: shape AR).
    kappa = float(CORRECTOR["gate_kappa"])
    ar_max = float(CORRECTOR["shape_gate_AR_max"])
    eta_final = float(np.sqrt(max(2.0 * loss_final, 0.0)))
    if tau_global is not None:
        tau = float(tau_global)
        residual_pass = bool(eta_final <= tau)
    else:
        tau = float("nan")
        residual_pass = True
    shape_pass = bool(ar <= ar_max)
    accepted = bool(residual_pass and shape_pass)

    # corrector_success: non-ABNORMAL, non-NaN, scipy reported success
    msg = run["status_msg"]
    is_abnormal = "ABNORMAL" in msg.upper()
    corrector_success = bool(
        run["success"]
        and run["status_int"] == 0
        and not is_abnormal
        and not (loss_final != loss_final)  # NaN-safe check
    )

    return CorrectorV2Result(
        # Original
        theta=theta_final,
        loss_init=float(primary["loss_init"]),
        loss_final=float(loss_final),
        iters=int(run["iters"]),
        fevals=int(run["fevals"]),
        status=str(msg)[:80],
        saturated_frac=float(saturated),
        aspect_ratio=float(ar),
        accepted=accepted,
        elapsed_sec=float(total_elapsed),
        grad_norm_init=float(primary["grad_norm_init"]),
        grad_norm_final=float(run["grad_norm_final"]),
        # Estimator decomposition
        eta_volume=float(eta_volume),
        eta_neumann=float(eta_neumann),
        eta_neumann_fraction=float(eta_neumann_fraction),
        # LBFGS-B counters
        lbfgsb_nit=int(run["iters"]),
        lbfgsb_nfev=int(run["fevals"]),
        lbfgsb_ngev=int(run["ngevs"]),
        lbfgsb_status=str(msg)[:120],
        corrector_success=bool(corrector_success),
        # Saturation breakdown
        logit_saturation_lower=float(sat_lower),
        logit_saturation_upper=float(sat_upper),
        logits_min=float(np.min(theta_final_np)),
        logits_max=float(np.max(theta_final_np)),
        # Gates
        residual_gate_pass=bool(residual_pass),
        shape_gate_pass=bool(shape_pass),
        certified=bool(accepted),
        tau=float(tau),
        kappa=float(kappa),
        # Trajectory
        history_loss=list(run["history_loss"]),
        extras={
            "primary_success": bool(primary["success"]),
            "primary_loss_final": float(primary["loss_final"]),
            "n_starts_run": 1,
        },
    )


__all__ = [
    "CorrectorV2Result",
    "correct_lbfgsb_v2",
    "HISTORY_LOSS_CAP",
]
