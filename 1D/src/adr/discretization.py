from __future__ import annotations

import functools

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np

from common.bspline_basis import bspline_basis_local
from common.quadrature import rule_gl_on_elements
from src.adr.problem import exact, exact_dx, rhs
from src.nonparametric.knots import theta_to_knots, theta_to_sizes


def quad_order(p):
    return int(p) + 5


def knots_from_theta(theta, p, h_min):
    return theta_to_knots(theta, 0.0, 1.0, int(p), h_min=float(h_min))


def _points(knots, p, nq, n_sub=1):
    n = knots.shape[0] - 2 * int(p) - 1
    a, b = knots[p:p + n], knots[p + 1:p + n + 1]
    t = jnp.linspace(0.0, 1.0, int(n_sub) + 1, dtype=knots.dtype)
    sa = (a[:, None] + (b - a)[:, None] * t[None, :-1]).reshape(-1)
    sb = (a[:, None] + (b - a)[:, None] * t[None, 1:]).reshape(-1)
    xq, wq = rule_gl_on_elements(sa, sb, int(nq))
    N, dN, d2N, spans = bspline_basis_local(xq.reshape(-1), knots, int(p))
    idx = (spans - int(p))[:, None] + jnp.arange(int(p) + 1, dtype=jnp.int32)[None, :]
    shape = (n, int(n_sub) * int(nq))
    return (xq.reshape(shape), wq.reshape(shape), N.reshape(shape + (p + 1,)),
            dN.reshape(shape + (p + 1,)), d2N.reshape(shape + (p + 1,)), idx.reshape(shape + (p + 1,)), b - a)


def _field(coeffs, B, idx):
    return jnp.sum(B * coeffs[idx], axis=-1)


@functools.partial(jax.jit, static_argnames=("p", "nq"))
def solve(knots, p, eps, b, nq):
    xq, wq, N, dN, _, idx, _ = _points(knots, p, nq)
    n_ctrl = knots.shape[0] - p - 1
    loc = (eps * jnp.einsum("eqi,eq,eqj->eqij", dN, wq, dN)
           + b * jnp.einsum("eqi,eq,eqj->eqij", N, wq, dN)
           + jnp.einsum("eqi,eq,eqj->eqij", N, wq, N))
    A = jnp.zeros((n_ctrl, n_ctrl), knots.dtype).at[idx[..., :, None], idx[..., None, :]].add(loc)
    F = jnp.zeros((n_ctrl,), knots.dtype).at[idx].add(N * (wq * rhs(xq, eps, b))[..., None])
    u = jnp.linalg.solve(A[1:-1, 1:-1], F[1:-1])
    return jnp.concatenate([jnp.zeros((1,), knots.dtype), u, jnp.zeros((1,), knots.dtype)])


@functools.partial(jax.jit, static_argnames=("p", "nq"))
def eta2_elements(knots, p, u, eps, b, nq):
    xq, wq, N, dN, d2N, idx, h = _points(knots, p, nq)
    r = rhs(xq, eps, b) + eps * _field(u, d2N, idx) - b * _field(u, dN, idx) - _field(u, N, idx)
    rho = jnp.minimum(h / jnp.sqrt(eps), 1.0)
    return rho ** 2 * jnp.sum(wq * r * r, axis=1)


def eta2(knots, p, u, eps, b, nq):
    return jnp.sum(eta2_elements(knots, p, u, eps, b, nq))


@functools.partial(jax.jit, static_argnames=("p", "nq", "n_sub"))
def energy_errors(knots, p, u, eps, b, nq=12, n_sub=8):
    xq, wq, N, dN, _, idx, _ = _points(knots, p, nq, n_sub)
    e = exact(xq, eps, b) - _field(u, N, idx)
    de = exact_dx(xq, eps, b) - _field(u, dN, idx)
    err2 = jnp.sum(wq * (eps * de * de + e * e))
    ref2 = jnp.sum(wq * (eps * exact_dx(xq, eps, b) ** 2 + exact(xq, eps, b) ** 2))
    return jnp.sqrt(err2), jnp.sqrt(err2 / ref2)


def gamma_loc(sizes, p):
    h = np.asarray(sizes, dtype=np.float64)
    w = 2 * int(p) + 1
    if h.size <= w:
        return float(h.max() / h.min())
    windows = np.lib.stride_tricks.sliding_window_view(h, w)
    return float(np.max(windows.max(axis=1) / windows.min(axis=1)))


def sizes_from_theta(theta, h_min):
    return theta_to_sizes(theta, float(h_min))


__all__ = ["quad_order", "knots_from_theta", "solve", "eta2_elements", "eta2",
           "energy_errors", "gamma_loc", "sizes_from_theta"]
