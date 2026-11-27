from __future__ import annotations

import functools

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp

from common.bspline_basis import basis_batch_for_degree
from common.quadrature import rule_gl_on_elements
from common.solver_custom_vjp_2d import solve_system_2d
from src.nonparametric.eta_estimator_2d import _endpoint_basis_right, _gather_local_coeffs
from src.nonparametric.solver_2d import (
    Solve2DResult,
    _apply_dirichlet,
    _assemble_2d,
    _assemble_neumann_1d,
    _element_data,
    free_mask_arctan,
)

SHARPNESS = 20.0
OFFSET = 0.5


def u_scalar(x, y, angle, t=SHARPNESS, s=OFFSET):
    return x * y * jnp.arctan(t * (x * jnp.cos(angle) + y * jnp.sin(angle) - s))


def _vec(fn):
    return jnp.vectorize(fn, excluded=(2, 3, 4))


def u_exact(x, y, angle, t=SHARPNESS, s=OFFSET):
    return _vec(u_scalar)(x, y, angle, t, s)


def grad_exact(x, y, angle, t=SHARPNESS, s=OFFSET):
    gx = _vec(jax.grad(u_scalar, argnums=0))(x, y, angle, t, s)
    gy = _vec(jax.grad(u_scalar, argnums=1))(x, y, angle, t, s)
    return gx, gy


def laplacian_exact(x, y, angle, t=SHARPNESS, s=OFFSET):
    uxx = jax.grad(jax.grad(u_scalar, argnums=0), argnums=0)
    uyy = jax.grad(jax.grad(u_scalar, argnums=1), argnums=1)
    return _vec(lambda a, b, c, d, e: uxx(a, b, c, d, e) + uyy(a, b, c, d, e))(x, y, angle, t, s)


def forcing(x, y, angle, t=SHARPNESS, s=OFFSET):
    return -laplacian_exact(x, y, angle, t, s)


def neumann_right(yv, angle, t=SHARPNESS, s=OFFSET):
    return grad_exact(jnp.ones_like(yv), yv, angle, t, s)[0]


def neumann_top(xv, angle, t=SHARPNESS, s=OFFSET):
    return grad_exact(xv, jnp.ones_like(xv), angle, t, s)[1]


@functools.partial(jax.jit, static_argnames=("p", "n_elem_x", "n_elem_y", "q_K", "q_F"))
def galerkin_solve_oblique(knots_x, knots_y, p, n_elem_x, n_elem_y, angle, *, q_K=4, q_F=20):
    K, F = _assemble_2d(knots_x, knots_y, p, n_elem_x, n_elem_y,
                        lambda x, y: jnp.ones_like(x), lambda x, y: forcing(x, y, angle), q_K, q_F)
    n_x, n_y = n_elem_x + p, n_elem_y + p
    G = F.reshape(n_x, n_y)
    G = G.at[n_x - 1, :].add(_assemble_neumann_1d(lambda y: neumann_right(y, angle), knots_y, p, n_elem_y, q_F))
    G = G.at[:, n_y - 1].add(_assemble_neumann_1d(lambda x: neumann_top(x, angle), knots_x, p, n_elem_x, q_F))
    K, F = _apply_dirichlet(K, G.reshape(-1), free_mask_arctan(knots_x, knots_y, p).reshape(-1))
    u = solve_system_2d(K, F)
    return Solve2DResult(u_h=u.reshape(n_x, n_y), ritz_energy=0.5 * u @ K @ u - u @ F, K=K, F=F)


@functools.partial(jax.jit, static_argnames=("p", "n_elem_x", "n_elem_y", "q_est"))
def eta_squared_oblique(C, knots_x, knots_y, p, n_elem_x, n_elem_y, angle, *, q_est=20):
    a_x, b_x, Ux, ix = _element_data(knots_x, p, n_elem_x)
    a_y, b_y, Uy, iy = _element_data(knots_y, p, n_elem_y)
    xq, wx = rule_gl_on_elements(a_x, b_x, q_est)
    yq, wy = rule_gl_on_elements(a_y, b_y, q_est)
    Nx, _, d2Nx = basis_batch_for_degree(p, xq, Ux)
    Ny, _, d2Ny = basis_batch_for_degree(p, yq, Uy)
    Cl = _gather_local_coeffs(C, ix, iy)
    lap = (jnp.einsum("xyij,xqi,yrj->xyqr", Cl, d2Nx, Ny) + jnp.einsum("xyij,xqi,yrj->xyqr", Cl, Nx, d2Ny))
    XX = jnp.broadcast_to(xq[:, None, :, None], lap.shape)
    YY = jnp.broadcast_to(yq[None, :, None, :], lap.shape)
    R = -lap - forcing(XX, YY, angle)
    hx, hy = b_x - a_x, b_y - a_y
    vol = jnp.sum((hx[:, None] ** 2 + hy[None, :] ** 2)
                  * jnp.sum(R * R * wx[:, None, :, None] * wy[None, :, None, :], axis=(2, 3)))
    gxr, _, dNxr = _endpoint_basis_right(knots_x, p, n_elem_x)
    rx = jnp.einsum("eqi,ei->eq", Ny, (dNxr @ C[gxr, :])[iy]) - neumann_right(yq, angle)
    gyt, _, dNyt = _endpoint_basis_right(knots_y, p, n_elem_y)
    ry = jnp.einsum("eqi,ei->eq", Nx, (C[:, gyt] @ dNyt)[ix]) - neumann_top(xq, angle)
    bnd = jnp.sum(hy * jnp.sum(wy * rx * rx, axis=1)) + jnp.sum(hx * jnp.sum(wx * ry * ry, axis=1))
    return vol + bnd


__all__ = ["SHARPNESS", "OFFSET", "u_exact", "grad_exact", "forcing", "neumann_right",
           "neumann_top", "galerkin_solve_oblique", "eta_squared_oblique"]
