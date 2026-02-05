from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from power_quadrature import load_global_analytic_power, stiffness_global_gl

Array = jnp.ndarray


@functools.partial(jax.jit, static_argnames=("degree", "qK"))
def assemble_system_analytic(
    knots: Array,
    degree: int,
    beta: float,
    *,
    qK: int,
) -> tuple[Array, Array]:
    """Assemble (K,F) for the manufactured power-solution problem.

    Problem:
        -u'' = beta(1-beta) x^{beta-2},
        u(0)=0,
        u'(1)=beta.

    Notes
    -----
    - Stiffness is assembled using element-wise Gauss--Legendre quadrature.
    - RHS load integral is computed analytically.
    - Dirichlet at x=0 is imposed strongly.
    - Neumann at x=1 is handled in the RHS.
    """
    degree = int(degree)
    dtype = knots.dtype
    beta = jnp.asarray(beta, dtype=dtype)

    # Stiffness (GL)
    K = stiffness_global_gl(knots, degree, nq=int(qK))
    # RHS (analytic)
    F = load_global_analytic_power(knots, degree, beta)

    # Strong Dirichlet at x=0
    K = K.at[0, :].set(0.0).at[:, 0].set(0.0).at[0, 0].set(1.0)
    F = F.at[0].set(0.0)

    # Neumann at x=1: add beta to last dof (sigma=1)
    F = F.at[-1].add(beta)

    return K, F


__all__ = ["assemble_system_analytic"]
