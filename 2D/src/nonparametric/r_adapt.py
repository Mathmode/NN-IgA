"""Local r-adapt optimizer for both arctan and lshape.

For each axis we have a vector of logits in R^n. We map them to a knot
vector via softmax + cumsum:

    sizes = softmax(theta)    in simplex (positive, sum = 1)
    knots = cumsum(sizes)     positions in (0, 1], with knots[0] = sizes[0]

For arctan (single patch on [0, 1]): n_logits = n_elem, sizes sum to 1.

For lshape (L-shape with fixed node at 0.5): we use TWO logit vectors per axis,
one for [0, 0.5] (n_elem / 2 elements) and one for [0.5, 1] (n_elem / 2).
Each block softmax sums to 0.5 (so the fixed node sits at 0.5 by construction).

Optimization runs via L-BFGS-B (scipy) on a flat parameter vector that
concatenates all logits. The objective is the Ritz energy of the FE solve.

For the smoke phase this is implemented as a synchronous (non-JIT-pure)
scipy optimization. For the cluster training a JAX-native LBFGS may be
considered (Phase 3+).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

Array = jnp.ndarray


# --------------------------------------------------------------------------
# Logits -> sizes -> knot positions
# --------------------------------------------------------------------------


def softmax_jax(theta: Array) -> Array:
    """Numerically stable softmax."""
    theta = jnp.asarray(theta)
    m = jnp.max(theta)
    e = jnp.exp(theta - m)
    return e / jnp.sum(e)


def theta_to_knots_arctan(theta: Array, p: int) -> Array:
    """Knot vector for arctan single-patch on [0, 1] with n_elem = len(theta) elements
    and degree p (open-uniform multiplicity at endpoints)."""
    sizes = softmax_jax(theta)
    interior = jnp.cumsum(sizes)[:-1]  # positions of n_elem - 1 interior knots
    zeros = jnp.zeros((p + 1,), dtype=sizes.dtype)
    ones = jnp.ones((p + 1,), dtype=sizes.dtype)
    return jnp.concatenate([zeros, interior, ones])


def theta_to_knots_lshape(theta_left: Array, theta_right: Array, p: int) -> Array:
    """Knot vector for lshape: two sub-meshes joined at 0.5, with multiplicity p+1
    at both endpoints (0 and 1) and **multiplicity p at the material
    interface 0.5** so the IGA basis is ``C^0`` there.

    Rationale (audit_p3_multipatch.md): the L-shape PDE has a piecewise
    diffusion coefficient σ jumping at x = 0.5 and y = 0.5. The physical
    solution has only ``σ ∂_n u`` continuous, not ``∂_n u`` itself.
    Multiplicity ``p`` reduces the basis to ``C^0`` at 0.5, matching the
    physics. The (p - 1) extra copies of 0.5 introduce (p - 1) zero-
    length "elements" at 0.5; their assembly contributions vanish.

    Gradient safety: this requires ``common.bspline_basis.safe_div`` to
    use the double-where pattern (commit ``safe_div_robust_multiplicity``);
    the prior single-where produced NaN gradients on the unselected branch
    when knot differences hit zero. Without that fix, training breaks.

    Effective element count for the solver: ``p3_effective_n_elem(N, p)``
        = N + (p - 1)
    where N is the logical mesh size (= number of physical elements per
    axis).
    """
    sizes_l = softmax_jax(theta_left) * 0.5    # rescale to fit [0, 0.5]
    sizes_r = softmax_jax(theta_right) * 0.5
    interior_l = jnp.cumsum(sizes_l)[:-1]                  # in (0, 0.5)
    interior_r = jnp.asarray(0.5, dtype=sizes_l.dtype) + jnp.cumsum(sizes_r)[:-1]
    # Multiplicity p at 0.5 → C^0 at the σ-discontinuity.
    fixed = jnp.full((int(p),), 0.5, dtype=sizes_l.dtype)
    interior = jnp.concatenate([interior_l, fixed, interior_r])
    zeros = jnp.zeros((p + 1,), dtype=sizes_l.dtype)
    ones = jnp.ones((p + 1,), dtype=sizes_l.dtype)
    return jnp.concatenate([zeros, interior, ones])


def p3_effective_n_elem(n_elem_logical: int, p: int) -> int:
    """Effective element count for a lshape mesh whose interior knot 0.5 has
    multiplicity ``p``.

    The solver's hard-coded ``n_x = n_elem + p`` formula expects this
    augmented count, since the (p - 1) extra copies of 0.5 introduce
    (p - 1) zero-length elements at the material interface.
    """
    return int(n_elem_logical) + int(p) - 1


def theta_uniform_arctan(n_elem: int) -> Array:
    """Logits whose softmax yields uniformly-spaced knots."""
    return jnp.zeros((int(n_elem),), dtype=DEFAULT_DTYPE)


def theta_uniform_p3_axis(n_elem: int) -> tuple[Array, Array]:
    """Half-block uniform logits for lshape: (theta_left, theta_right)."""
    half = int(n_elem) // 2
    if half * 2 != int(n_elem):
        raise ValueError(f"lshape needs even n_elem; got {n_elem}")
    z = jnp.zeros((half,), dtype=DEFAULT_DTYPE)
    return z, z


# --------------------------------------------------------------------------
# L-BFGS-B wrapper using scipy on a numpy-flattened parameter vector.
# --------------------------------------------------------------------------


@dataclass
class RAdaptResult:
    theta_final: np.ndarray
    energy_final: float
    n_iter: int
    converged: bool
    message: str


def radapt_local(
    objective_and_grad_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    theta0: np.ndarray,
    *,
    max_iter: int = 200,
    ftol: float = 1e-10,
    gtol: float = 1e-7,
    bounds: Optional[list[tuple[float, float]]] = None,
) -> RAdaptResult:
    """Run L-BFGS-B on a 1D parameter vector.

    ``objective_and_grad_fn`` takes ``theta`` (numpy) and returns
    ``(loss, grad)`` where ``grad`` has the same shape as ``theta``.

    ``bounds`` is the optional box constraint (decision 7-C / unification:
    the 2D corrector must run inside ``q ∈ [-T, T]^m``). Pass ``[(-T, T)]
    * m`` from the call site; ``None`` (default) leaves L-BFGS-B
    effectively unconstrained, preserving the previous behaviour for any
    legacy caller.
    """
    theta0 = np.asarray(theta0, dtype=np.float64).reshape(-1)

    if bounds is not None:
        # Defensive clamp so the warm-start is feasible.
        lows = np.asarray([float(lo) for (lo, _) in bounds], dtype=np.float64)
        highs = np.asarray([float(hi) for (_, hi) in bounds], dtype=np.float64)
        theta0 = np.clip(theta0, lows, highs)

    def fun_and_grad(theta_np: np.ndarray):
        loss, grad = objective_and_grad_fn(theta_np)
        return float(loss), np.asarray(grad, dtype=np.float64).reshape(-1)

    res = minimize(
        fun_and_grad,
        theta0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iter), "ftol": float(ftol), "gtol": float(gtol)},
    )
    return RAdaptResult(
        theta_final=np.asarray(res.x, dtype=np.float64),
        energy_final=float(res.fun),
        n_iter=int(res.nit),
        converged=bool(res.success),
        message=str(res.message),
    )


# --------------------------------------------------------------------------
# Convenience wrappers: r-adapt for the lshape reference (single (sigma1, sigma2)).
# --------------------------------------------------------------------------


def radapt_p3_one_sample(
    sigma1: float,
    sigma2: float,
    *,
    n_elem: int,
    p: int,
    max_iter: int = 200,
    theta_init: Optional[np.ndarray] = None,
) -> tuple[RAdaptResult, Array]:
    """Find the (uniform-style) r-adapted mesh that minimises lshape's
    **residual a-posteriori estimator η²** for a single (sigma1, sigma2).

    Returns ``(result, knots_x)`` for the OPTIMIZED axis. Both axes share
    the same logits (anchored uniform initial -> symmetric L-shape).
    Cross-axis coupling is NOT explored at this stage; we just optimise one
    axis. The other axis is held at uniform.

    NOTE: this used to minimise Ritz energy until the audit-driven Ritz
    elimination pass. The residual estimator is the consistent objective
    with the rest of the training pipeline.
    """
    from src.nonparametric.solver_2d import galerkin_solve_lshape
    from src.nonparametric.eta_estimator_2d import eta_squared_lshape

    half = int(n_elem) // 2
    if half * 2 != int(n_elem):
        raise ValueError(f"n_elem must be even; got {n_elem}")
    n_params = 2 * half

    # Uniform initial knots for the y-axis (held fixed).
    theta_y_l = jnp.zeros((half,), dtype=DEFAULT_DTYPE)
    theta_y_r = jnp.zeros((half,), dtype=DEFAULT_DTYPE)
    knots_y = theta_to_knots_lshape(theta_y_l, theta_y_r, p)

    n_elem_eff = p3_effective_n_elem(int(n_elem), int(p))

    def loss_fn(theta_flat):
        theta_x_l = theta_flat[:half]
        theta_x_r = theta_flat[half:]
        knots_x = theta_to_knots_lshape(theta_x_l, theta_x_r, p)
        res = galerkin_solve_lshape(
            knots_x, knots_y, p, n_elem_eff, n_elem_eff,
            sigma1, sigma2, q_K=p + 1, q_F=2,
        )
        return eta_squared_lshape(
            res.u_h, knots_x, knots_y, p, n_elem_eff, n_elem_eff,
            sigma1, sigma2, q_est=2,
        )

    loss_and_grad = jax.value_and_grad(loss_fn)

    def obj(theta_np):
        theta_jax = jnp.asarray(theta_np, dtype=DEFAULT_DTYPE)
        loss, grad = loss_and_grad(theta_jax)
        return float(loss), np.asarray(grad, dtype=np.float64)

    theta0 = theta_init if theta_init is not None else np.zeros(n_params, dtype=np.float64)
    result = radapt_local(obj, theta0, max_iter=max_iter)
    theta_x_l = jnp.asarray(result.theta_final[:half], dtype=DEFAULT_DTYPE)
    theta_x_r = jnp.asarray(result.theta_final[half:], dtype=DEFAULT_DTYPE)
    knots_x = theta_to_knots_lshape(theta_x_l, theta_x_r, p)
    return result, knots_x


__all__ = [
    "Array",
    "softmax_jax",
    "theta_to_knots_arctan",
    "theta_to_knots_lshape",
    "p3_effective_n_elem",
    "theta_uniform_arctan",
    "theta_uniform_p3_axis",
    "RAdaptResult",
    "radapt_local",
    "radapt_p3_one_sample",
]
