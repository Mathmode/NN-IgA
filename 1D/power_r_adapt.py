from __future__ import annotations

"""r-adaptivity optimiser for Experiment 1 (power solution).

At each optimiser step:
  1) map unconstrained parameters ``theta`` -> open knot vector,
  2) assemble + solve the IGA Galerkin system,
  3) compute the residual estimator loss (analytic terms),
  4) update theta with Adam.

Training objective:
    L(theta) = 0.5 * eta(theta)^2
"""

from typing import Any, Dict, Optional

import numpy as np
import jax
import jax.numpy as jnp

from power_assembly import assemble_system_analytic
from power_knots import theta_to_knots
from power_metrics import compute_errors_power_analytic
from power_quadrature import estimator_loss_analytic_power
from power_solver import solve_system
from optimization import AdamTrainConfig, train_adam

Array = jnp.ndarray


def r_adapt_optimize_estimator(
    theta0: Array,
    *,
    degree: int,
    beta: float,
    qK: int,
    iters: int = 10_000,
    lr0: float = 1e-3,
    lr1: float = 1e-4,
    h_min: float = 1e-6,
    key: Optional[jax.random.PRNGKey] = None,
    log_every: int = 500,
) -> tuple[Array, Array, Array, list[Dict[str, Any]], Dict[str, float]]:
    """Optimise knot parameters by minimising the residual-estimator loss."""

    degree = int(degree)
    qK = int(qK)
    iters = int(iters)
    log_every = int(log_every)

    theta = jnp.asarray(theta0, dtype=jnp.float64).reshape(-1)
    beta_arr = jnp.asarray(float(beta), dtype=theta.dtype)

    def loss_and_aux(theta_in: Array):
        knots = theta_to_knots(theta_in, 0.0, 1.0, degree, h_min=h_min)
        K, F = assemble_system_analytic(knots, degree, beta_arr, qK=qK)
        U = solve_system(K, F)
        L = estimator_loss_analytic_power(knots, degree, U, beta=beta_arr)
        return L, (knots, U, K)

    cfg = AdamTrainConfig(
        iters=int(iters),
        lr1=float(lr0),
        lr2=float(lr1),
        lr_switch=0.5,
        log_every=int(log_every),
        noise_scale=1e-3,
    )

    def log_fn(it: int, theta_it: Array, loss: Array, aux, gnorm: Array, lr: float) -> Dict[str, Any]:
        knots, U, K = aux

        H1_rel, L2_rel = compute_errors_power_analytic(
            knots,
            degree,
            U,
            beta=float(beta_arr),
            relative=True,
        )

        # Condition number estimate (dense eigvals). Useful debug signal.
        K_np = np.asarray(K)
        try:
            lam = np.linalg.eigvalsh(K_np)
            lam_min = max(float(lam[0]), 1.0e-15)
            lam_max = float(lam[-1])
            condK = lam_max / lam_min
        except Exception:
            condK = float("nan")

        rec = {
            "iter": int(it),
            "loss": float(loss),
            "loss_val": float(loss),
            "H1_rel": float(H1_rel),
            "L2_rel": float(L2_rel),
            "h_min": float(np.min(np.diff(np.asarray(knots)[degree:-degree]))),
            "lr": float(lr),
            "gnorm": float(gnorm),
            "condK": float(condK),
        }

        # Standard console log (consistent across experiments)
        n_elem = int(theta_it.shape[0])
        print(
            f"[N={n_elem:3d} it {it:5d}] loss={rec['loss']:.3e} val={rec['loss_val']:.3e} | "
            f"H1_rel={rec['H1_rel']:.3e} | L2_rel={rec['L2_rel']:.3e} | h_min={rec['h_min']:.2e}"
        )
        return rec

    theta, aux_final, history, summary = train_adam(theta, loss_and_aux, cfg=cfg, key=key, log_fn=log_fn)
    knots_final, U_final, _K_final = aux_final
    return theta, knots_final, U_final, history, summary
