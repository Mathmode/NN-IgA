"""Online correction: take the network's mesh prediction and refine it
locally with a few L-BFGS-B iterations to push the Ritz energy down.

For ARCTAN, the corrector also accepts the analytic gradient callable and
operates on the H1 error metric directly. For LSHAPE, it minimises the Ritz
energy.

The corrector budget (number of L-BFGS-B iterations) is a key hyperparam;
the prompt's CORRECTOR config sets it to 80 by default.
"""
from __future__ import annotations


from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np

from src.nonparametric.r_adapt import (
    RAdaptResult,
    p3_effective_n_elem,
    radapt_local,
    theta_to_knots_arctan,
    theta_to_knots_lshape,
)
from src.nonparametric.solver_2d import galerkin_solve_arctan, galerkin_solve_lshape

Array = jnp.ndarray


# --------------------------------------------------------------------------
# Theta extraction from network prediction (inverse of softmax for warmstart).
# --------------------------------------------------------------------------


def theta_from_sizes_arctan(sizes: Array) -> Array:
    """Warmstart logits theta such that softmax(theta) ~ sizes.

    Uses theta = log(sizes) - mean(log(sizes)) (gauge-fixed). The softmax
    of this theta equals sizes up to renormalisation.
    """
    log_s = jnp.log(sizes)
    return log_s - jnp.mean(log_s)


def theta_from_sizes_lshape(sizes_left: Array, sizes_right: Array) -> tuple[Array, Array]:
    """Warmstart logits for LSHAPE (two halves)."""
    return theta_from_sizes_arctan(sizes_left), theta_from_sizes_arctan(sizes_right)


# --------------------------------------------------------------------------
# Corrector for ARCTAN
# --------------------------------------------------------------------------


def corrector_arctan(
    theta_x_init: Array,
    theta_y_init: Array,
    sigma: Array,                     # (3,)
    *,
    p: int,
    n_elem: int,
    q_K: int = 4,
    q_F: int = 40,
    max_iter: int = 80,
    T: float | None = None,
    ftol: float | None = None,
    gtol: float | None = None,
) -> tuple[RAdaptResult, Array, Array]:
    """Local L-BFGS-B on (theta_x, theta_y) to minimise the **residual
    a-posteriori estimator η²** for one σ-tuple in ARCTAN.

    Returns ``(result, knots_x_opt, knots_y_opt)``.

    Per decision 7-C / 7-D / unification: the optimisation is
    **box-constrained** in the bounded-logit cube ``q ∈ [-T, T]^m``;
    ``T`` defaults to ``T_ARCTAN`` from ``src.config``. ``ftol`` and ``gtol``
    default to ``CORRECTOR["ftol"]`` / ``CORRECTOR["gtol"]``.
    """
    from src.nonparametric.eta_estimator_2d import eta_squared_arctan
    from src.config import CORRECTOR, T_ARCTAN

    T_val = float(T_ARCTAN) if T is None else float(T)
    ftol_val = float(CORRECTOR["ftol"]) if ftol is None else float(ftol)
    gtol_val = float(CORRECTOR["gtol"]) if gtol is None else float(gtol)

    theta_x_init = np.asarray(theta_x_init, dtype=np.float64).reshape(-1)
    theta_y_init = np.asarray(theta_y_init, dtype=np.float64).reshape(-1)
    nx = int(theta_x_init.shape[0])
    ny = int(theta_y_init.shape[0])
    theta_flat0 = np.concatenate([theta_x_init, theta_y_init])

    def loss_fn(theta_flat):
        tx = theta_flat[:nx]
        ty = theta_flat[nx:]
        knots_x = theta_to_knots_arctan(tx, p)
        knots_y = theta_to_knots_arctan(ty, p)
        res = galerkin_solve_arctan(
            knots_x, knots_y, p, n_elem, n_elem,
            sigma[0], sigma[1], sigma[2],
            q_K=q_K, q_F=q_F,
        )
        return eta_squared_arctan(
            res.u_h, knots_x, knots_y, p, n_elem, n_elem,
            sigma[0], sigma[1], sigma[2], q_est=q_F,
        )

    loss_and_grad = jax.value_and_grad(loss_fn)

    def obj(theta_np):
        theta_jax = jnp.asarray(theta_np, dtype=DEFAULT_DTYPE)
        loss, grad = loss_and_grad(theta_jax)
        return float(loss), np.asarray(grad, dtype=np.float64)

    bounds = [(-T_val, T_val)] * (nx + ny)
    result = radapt_local(
        obj, theta_flat0,
        max_iter=int(max_iter), ftol=ftol_val, gtol=gtol_val,
        bounds=bounds,
    )
    theta_x_opt = jnp.asarray(result.theta_final[:nx], dtype=DEFAULT_DTYPE)
    theta_y_opt = jnp.asarray(result.theta_final[nx:], dtype=DEFAULT_DTYPE)
    knots_x_opt = theta_to_knots_arctan(theta_x_opt, p)
    knots_y_opt = theta_to_knots_arctan(theta_y_opt, p)
    return result, knots_x_opt, knots_y_opt


# --------------------------------------------------------------------------
# Corrector for LSHAPE
# --------------------------------------------------------------------------


def corrector_lshape(
    theta_x_left_init: Array,
    theta_x_right_init: Array,
    theta_y_left_init: Array,
    theta_y_right_init: Array,
    sigma: Array,                     # (2,)
    *,
    p: int,
    n_elem: int,
    q_K: int = 4,
    q_F: int = 2,
    max_iter: int = 80,
    T: float | None = None,
    ftol: float | None = None,
    gtol: float | None = None,
) -> tuple[RAdaptResult, Array, Array]:
    """Local L-BFGS-B on the 4 LSHAPE logit blocks to minimise the **residual
    a-posteriori estimator η²** for one σ-tuple.

    Per decision 7-C / 7-D / unification: optimisation is
    **box-constrained** in ``q ∈ [-T, T]^m`` with ``T`` defaulting to
    ``T_LSHAPE`` from ``src.config``. ``ftol`` and ``gtol`` default to
    ``CORRECTOR["ftol"]`` / ``CORRECTOR["gtol"]``.
    """
    from src.nonparametric.eta_estimator_2d import eta_squared_lshape
    from src.config import CORRECTOR, T_LSHAPE

    T_val = float(T_LSHAPE) if T is None else float(T)
    ftol_val = float(CORRECTOR["ftol"]) if ftol is None else float(ftol)
    gtol_val = float(CORRECTOR["gtol"]) if gtol is None else float(gtol)

    half = int(n_elem) // 2
    a = np.asarray(theta_x_left_init).reshape(-1)
    b = np.asarray(theta_x_right_init).reshape(-1)
    c = np.asarray(theta_y_left_init).reshape(-1)
    d = np.asarray(theta_y_right_init).reshape(-1)
    theta_flat0 = np.concatenate([a, b, c, d]).astype(np.float64)
    n_eff = p3_effective_n_elem(int(n_elem), int(p))

    def loss_fn(theta_flat):
        tx_l = theta_flat[:half]
        tx_r = theta_flat[half:2 * half]
        ty_l = theta_flat[2 * half:3 * half]
        ty_r = theta_flat[3 * half:]
        knots_x = theta_to_knots_lshape(tx_l, tx_r, p)
        knots_y = theta_to_knots_lshape(ty_l, ty_r, p)
        res = galerkin_solve_lshape(
            knots_x, knots_y, p, n_eff, n_eff,
            sigma[0], sigma[1],
            q_K=q_K, q_F=q_F,
        )
        return eta_squared_lshape(
            res.u_h, knots_x, knots_y, p, n_eff, n_eff,
            sigma[0], sigma[1], q_est=max(int(q_F), 4),
        )

    loss_and_grad = jax.value_and_grad(loss_fn)

    def obj(theta_np):
        theta_jax = jnp.asarray(theta_np, dtype=DEFAULT_DTYPE)
        loss, grad = loss_and_grad(theta_jax)
        return float(loss), np.asarray(grad, dtype=np.float64)

    bounds = [(-T_val, T_val)] * (4 * half)
    result = radapt_local(
        obj, theta_flat0,
        max_iter=int(max_iter), ftol=ftol_val, gtol=gtol_val,
        bounds=bounds,
    )
    tx_l = jnp.asarray(result.theta_final[:half], dtype=DEFAULT_DTYPE)
    tx_r = jnp.asarray(result.theta_final[half:2 * half], dtype=DEFAULT_DTYPE)
    ty_l = jnp.asarray(result.theta_final[2 * half:3 * half], dtype=DEFAULT_DTYPE)
    ty_r = jnp.asarray(result.theta_final[3 * half:], dtype=DEFAULT_DTYPE)
    knots_x_opt = theta_to_knots_lshape(tx_l, tx_r, p)
    knots_y_opt = theta_to_knots_lshape(ty_l, ty_r, p)
    return result, knots_x_opt, knots_y_opt


__all__ = [
    "theta_from_sizes_arctan",
    "theta_from_sizes_lshape",
    "corrector_arctan",
    "corrector_lshape",
]
