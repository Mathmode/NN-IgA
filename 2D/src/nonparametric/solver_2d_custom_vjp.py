"""arctan / lshape Galerkin solvers using the EXPLICIT discrete adjoint
(``solve_system_2d`` with ``custom_vjp``) instead of ``jnp.linalg.solve``.

Parallel to ``solver_2d.galerkin_solve_arctan`` / ``galerkin_solve_lshape`` —
byte-for-byte the same assembly, BCs, and ``Solve2DResult``; the ONLY
difference is that the line ``u = jnp.linalg.solve(K, F)`` is replaced by
``u = solve_system_2d(K, F)``. Existing ``solver_2d`` is untouched.

Everything else is imported from ``solver_2d`` (assembly, Neumann
assembly, free masks, Dirichlet application, the result container) — no
duplication of the FE machinery.
"""
from __future__ import annotations

import functools

from common._precision import ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp

from common.solver_custom_vjp_2d import solve_system_2d
from src.nonparametric.solver_2d import (
    Solve2DResult,
    _apply_dirichlet,
    _assemble_2d,
    _assemble_neumann_1d,
    free_mask_arctan,
    free_mask_lshape,
)

Array = jnp.ndarray


@functools.partial(
    jax.jit,
    static_argnames=("p", "n_elem_x", "n_elem_y", "q_K", "q_F"),
)
def galerkin_solve_p2_explicit_adjoint(
    knots_x: Array,
    knots_y: Array,
    p: int,
    n_elem_x: int,
    n_elem_y: int,
    alpha: Array,
    s1: Array,
    s2: Array,
    *,
    q_K: int = 4,
    q_F: int = 50,
) -> Solve2DResult:
    """Same as ``solver_2d.galerkin_solve_arctan`` but solves the linear
    system through ``solve_system_2d`` (explicit discrete-adjoint
    ``custom_vjp``). Result is numerically identical for the same inputs.
    """
    import jax.numpy as jnp
    from src.nonparametric.arctan.pde import (
        f_sigma,
        g_sigma_neumann_right,
        g_sigma_neumann_top,
    )

    sigma_callable = lambda x, y: jnp.ones_like(x)
    f_callable = lambda x, y: f_sigma(x, y, alpha, s1, s2)
    K, F = _assemble_2d(
        knots_x, knots_y, p, n_elem_x, n_elem_y,
        sigma_callable, f_callable, q_K, q_F,
    )

    F_right = _assemble_neumann_1d(
        lambda y: g_sigma_neumann_right(y, alpha, s1, s2),
        knots_y, p, n_elem_y, q_F,
    )
    F_top = _assemble_neumann_1d(
        lambda x: g_sigma_neumann_top(x, alpha, s1, s2),
        knots_x, p, n_elem_x, q_F,
    )

    n_x = n_elem_x + p
    n_y = n_elem_y + p
    F_grid = F.reshape(n_x, n_y)
    F_grid = F_grid.at[n_x - 1, :].add(F_right)
    F_grid = F_grid.at[:, n_y - 1].add(F_top)
    F = F_grid.reshape(-1)

    free = free_mask_arctan(knots_x, knots_y, p)
    K, F = _apply_dirichlet(K, F, free.reshape(-1))

    u = solve_system_2d(K, F)                 # <-- explicit adjoint
    ritz_energy = 0.5 * u @ K @ u - u @ F
    return Solve2DResult(
        u_h=u.reshape(n_x, n_y),
        ritz_energy=ritz_energy,
        K=K,
        F=F,
    )


@functools.partial(
    jax.jit,
    static_argnames=("p", "n_elem_x", "n_elem_y", "q_K", "q_F"),
)
def galerkin_solve_p3_explicit_adjoint(
    knots_x: Array,
    knots_y: Array,
    p: int,
    n_elem_x: int,
    n_elem_y: int,
    sigma1: Array,
    sigma2: Array,
    *,
    q_K: int = 4,
    q_F: int = 2,
) -> Solve2DResult:
    """Same as ``solver_2d.galerkin_solve_lshape`` but solves the linear
    system through ``solve_system_2d`` (explicit discrete-adjoint
    ``custom_vjp``). Result is numerically identical for the same inputs.
    """
    from src.nonparametric.lshape.pde import f_lshape, sigma_field

    sigma_callable = lambda x, y: sigma_field(x, y, sigma1, sigma2)
    f_callable = lambda x, y: f_lshape(x, y)
    K, F = _assemble_2d(
        knots_x, knots_y, p, n_elem_x, n_elem_y,
        sigma_callable, f_callable, q_K, q_F,
    )

    free = free_mask_lshape(knots_x, knots_y, p)
    K, F = _apply_dirichlet(K, F, free.reshape(-1))

    u = solve_system_2d(K, F)                 # <-- explicit adjoint
    ritz_energy = 0.5 * u @ K @ u - u @ F

    n_x = n_elem_x + p
    n_y = n_elem_y + p
    return Solve2DResult(
        u_h=u.reshape(n_x, n_y),
        ritz_energy=ritz_energy,
        K=K,
        F=F,
    )


__all__ = [
    "galerkin_solve_p2_explicit_adjoint",
    "galerkin_solve_p3_explicit_adjoint",
]
