"""Online L-BFGS-B corrector with box constraints (budget L80).

Given a warm-start theta_init in [-T, T]^N produced by the positional density
network (PDN) at level (p, N) and parameter beta, run a short L-BFGS-B refinement
of the residual loss. The box constraints keep theta inside [-T, T] so the
corrected mesh remains in the same admissible region the PDN is trained on.

The training loss is the analytic residual loss `reduced_loss(theta, p, q, h_min, beta)`
from `src.nonparametric.exp1.quadrature_analytic` — analytic monomial integrals,
no quadrature error in the loss itself, with adjoint gradient via custom_vjp.

Outputs: corrected theta + diagnostics (status, iterations, gate flags).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Tuple

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from src.config import CORRECTOR, T_SCHEDULE
from src.nonparametric.discretization import stiffness_quadrature_order
from src.nonparametric.knots import theta_to_sizes
from src.nonparametric.quadrature_analytic import reduced_loss

Array = jnp.ndarray


@dataclass
class CorrectorResult:
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
    extras: dict = field(default_factory=dict)


def _make_loss_grad(p: int, h_min: float, beta: float):
    q_order = stiffness_quadrature_order(int(p))

    def loss_fn(theta: Array) -> Array:
        return reduced_loss(theta, int(p), int(q_order), float(h_min), beta=float(beta))

    grad_fn = jax.grad(loss_fn)
    jit_value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    return loss_fn, grad_fn, jit_value_and_grad


def _saturated_fraction(theta: np.ndarray, T: float, *, frac: float = 0.99) -> float:
    """Fraction of components with |theta_i| >= frac * T."""
    return float(np.mean(np.abs(theta) >= frac * float(T)))


def _aspect_ratio(theta: Array, h_min: float) -> float:
    sizes = np.asarray(jax.device_get(theta_to_sizes(theta, float(h_min))), dtype=np.float64)
    h_max = float(np.max(sizes))
    h_min_obs = float(np.min(sizes))
    return h_max / max(h_min_obs, 1e-300)


def correct_lbfgsb(
    theta_init: Array,
    *,
    p: int,
    N: int,
    h_min: float,
    beta: float,
    T: float | None = None,
    budget: int | None = None,
    ftol: float | None = None,
    gtol: float | None = None,
    eta_init: float | None = None,
    eta_target: float | None = None,
) -> CorrectorResult:
    """L-BFGS-B with box constraints [-T, T] on theta.

    Parameters
    ----------
    theta_init : (N,)
        Initial logits from the positional density network (already in [-T, T]).
    p, N, h_min, beta : as in the non-parametric primitives.
    T : float, optional
        Box bound. Defaults to ``T_SCHEDULE[p][N]``.
    budget : int, optional
        Max L-BFGS-B iterations. Defaults to ``CORRECTOR['budget']`` (= 80).
    ftol, gtol : float, optional
        Stopping tolerances. Default from ``CORRECTOR``.
    eta_init, eta_target : float, optional
        Used only by the certificate gate. If both are provided, the corrector
        marks ``accepted = (eta_final <= max(eta_target, kappa * eta_init))``
        with kappa from CORRECTOR['gate_kappa']. Otherwise ``accepted = True``.
    """
    T = float(T_SCHEDULE[int(p)][int(N)]) if T is None else float(T)
    budget = int(CORRECTOR["budget"]) if budget is None else int(budget)
    ftol = float(CORRECTOR["ftol"]) if ftol is None else float(ftol)
    gtol = float(CORRECTOR["gtol"]) if gtol is None else float(gtol)

    loss_fn, grad_fn, _jvg = _make_loss_grad(int(p), float(h_min), float(beta))

    theta_np_init = np.asarray(jax.device_get(theta_init), dtype=np.float64).reshape(-1)
    # Clamp to box (defensive in case rounding pushed slightly outside).
    theta_np_init = np.clip(theta_np_init, -T, T)

    def np_loss_grad(theta_np: np.ndarray) -> Tuple[float, np.ndarray]:
        th = jnp.asarray(theta_np, dtype=DEFAULT_DTYPE)
        l = float(loss_fn(th))
        g = np.asarray(jax.device_get(grad_fn(th)), dtype=np.float64).reshape(-1)
        return l, g

    loss_init, grad_init = np_loss_grad(theta_np_init)
    grad_norm_init = float(np.linalg.norm(grad_init))

    bounds = [(-T, T)] * theta_np_init.shape[0]

    t0 = time.perf_counter()
    res = minimize(
        np_loss_grad,
        theta_np_init,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
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
    grad_final = np.asarray(jax.device_get(grad_fn(theta_final)), dtype=np.float64).reshape(-1)
    grad_norm_final = float(np.linalg.norm(grad_final))

    saturated = _saturated_fraction(theta_final_np, T)
    ar = _aspect_ratio(theta_final, float(h_min))

    # Certificate gate
    kappa = float(CORRECTOR["gate_kappa"])
    ar_max = float(CORRECTOR["shape_gate_AR_max"])
    accepted = True
    if eta_init is not None and eta_target is not None:
        eta_final = float(np.sqrt(max(2.0 * loss_final, 0.0)))
        certified = eta_final <= max(float(eta_target), kappa * float(eta_init))
        accepted = bool(certified) and (ar <= ar_max)

    return CorrectorResult(
        theta=theta_final,
        loss_init=float(loss_init),
        loss_final=float(loss_final),
        iters=int(res.nit),
        fevals=int(res.nfev),
        status=str(res.message)[:80],
        saturated_frac=float(saturated),
        aspect_ratio=float(ar),
        accepted=bool(accepted),
        elapsed_sec=float(elapsed),
        grad_norm_init=float(grad_norm_init),
        grad_norm_final=float(grad_norm_final),
        extras={"success": bool(res.success)},
    )


__all__ = ["CorrectorResult", "correct_lbfgsb"]
