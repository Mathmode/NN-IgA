from __future__ import annotations

"""IGA discretization helpers for the singular experiment."""

import functools
import math
from typing import NamedTuple

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from common.bspline_basis import safe_div, basis_p2_batch as _bspline_p2_batch, basis_p3_batch as _bspline_p3_batch
from common.quadrature import monomial_integral, rule_gl_on_elements
from src.nonparametric.knots import theta_to_knots
from src.nonparametric.pde import BETA
from src.nonparametric.solver import solve_system

Array = jnp.ndarray


class PowerEvalCache(NamedTuple):
    knots: Array
    sizes: Array
    a: Array
    b: Array
    xq: Array
    wq: Array
    windows: Array
    active_ids: Array
    basis: Array
    dbasis: Array


# Deduplicated: use canonical implementations from bspline_basis.py
basis_p2_batch = _bspline_p2_batch
basis_p3_batch = _bspline_p3_batch


def ders_basis_funs_local_single(x: Array, knot_window: Array, degree: int) -> tuple[Array, Array, Array]:
    p = int(degree)
    n_ders = min(2, p)
    dtype = knot_window.dtype
    span = p

    ndu = jnp.zeros((p + 1, p + 1), dtype=dtype)
    left = jnp.zeros((p + 1,), dtype=dtype)
    right = jnp.zeros((p + 1,), dtype=dtype)
    ndu = ndu.at[0, 0].set(1.0)

    for j in range(1, p + 1):
        left = left.at[j].set(x - knot_window[span + 1 - j])
        right = right.at[j].set(knot_window[span + j] - x)
        saved = jnp.asarray(0.0, dtype=dtype)
        for r in range(j):
            ndu = ndu.at[j, r].set(right[r + 1] + left[j - r])
            temp = safe_div(ndu[r, j - 1], ndu[j, r])
            ndu = ndu.at[r, j].set(saved + right[r + 1] * temp)
            saved = left[j - r] * temp
        ndu = ndu.at[j, j].set(saved)

    ders = jnp.zeros((n_ders + 1, p + 1), dtype=dtype)
    ders = ders.at[0, :].set(ndu[:, p])

    for r in range(p + 1):
        a = jnp.zeros((2, p + 1), dtype=dtype)
        a = a.at[0, 0].set(1.0)
        s1 = 0
        s2 = 1
        for k in range(1, n_ders + 1):
            a = a.at[s2, :].set(0.0)
            d = jnp.asarray(0.0, dtype=dtype)
            rk = r - k
            pk = p - k
            if r >= k:
                a = a.at[s2, 0].set(safe_div(a[s1, 0], ndu[pk + 1, rk]))
                d = d + a[s2, 0] * ndu[rk, pk]

            j1 = 1 if rk >= -1 else -rk
            j2 = (k - 1) if (r - 1) <= pk else (p - r)
            for j in range(j1, j2 + 1):
                a = a.at[s2, j].set(safe_div(a[s1, j] - a[s1, j - 1], ndu[pk + 1, rk + j]))
                d = d + a[s2, j] * ndu[rk + j, pk]

            if r <= pk:
                a = a.at[s2, k].set(-safe_div(a[s1, k - 1], ndu[pk + 1, r]))
                d = d + a[s2, k] * ndu[r, pk]

            ders = ders.at[k, r].set(d)
            s1, s2 = s2, s1

    for k in range(1, n_ders + 1):
        factor = 1.0
        for j in range(p - k + 1, p + 1):
            factor *= j
        ders = ders.at[k, :].set(ders[k, :] * factor)

    n = ders[0]
    dn = ders[1] if p >= 1 else jnp.zeros((p + 1,), dtype=dtype)
    d2n = ders[2] if p >= 2 else jnp.zeros((p + 1,), dtype=dtype)
    return n, dn, d2n


def basis_generic_batch(x: Array, knot_windows: Array, degree: int) -> tuple[Array, Array, Array]:
    x = jnp.asarray(x)
    if x.ndim == 1:
        x = x[:, None]

    def elem_eval(x_elem: Array, window: Array):
        return jax.vmap(lambda x_q: ders_basis_funs_local_single(x_q, window, degree))(x_elem)

    n, dn, d2n = jax.vmap(elem_eval)(x, knot_windows)
    return n, dn, d2n


def basis_batch(x: Array, knot_windows: Array, degree: int) -> tuple[Array, Array, Array]:
    if degree == 2:
        return basis_p2_batch(x, knot_windows)
    if degree == 3:
        return basis_p3_batch(x, knot_windows)
    if degree == 4:
        return basis_generic_batch(x, knot_windows, degree)
    raise ValueError(f"Unsupported degree={degree}; expected 2, 3, or 4.")


def stiffness_quadrature_order(degree: int) -> int:
    return int(degree) + 1


@functools.partial(jax.jit, static_argnames=("degree",))
def local_knot_windows(knots: Array, degree: int) -> Array:
    n_elem = knots.shape[0] - 2 * degree - 1

    def take_window(eid):
        return lax.dynamic_slice_in_dim(knots, eid, 2 * degree + 2, axis=0)

    return jax.vmap(take_window)(jnp.arange(n_elem, dtype=jnp.int32))


@functools.partial(jax.jit, static_argnames=("degree", "quadrature_order"))
def build_eval_cache(theta: Array, degree: int, quadrature_order: int, h_min: float) -> PowerEvalCache:
    knots, sizes = theta_to_knots(theta, 0.0, 1.0, degree, h_min=h_min, return_h=True)
    n_elem = sizes.shape[0]
    a = knots[degree : degree + n_elem]
    b = knots[degree + 1 : degree + n_elem + 1]
    xq, wq = rule_gl_on_elements(a, b, int(quadrature_order))
    windows = local_knot_windows(knots, degree)
    basis, dbasis, _d2basis = basis_batch(xq, windows, degree)
    active_ids = jnp.arange(n_elem, dtype=jnp.int32)[:, None] + jnp.arange(degree + 1, dtype=jnp.int32)[None, :]
    return PowerEvalCache(
        knots=knots,
        sizes=sizes,
        a=a,
        b=b,
        xq=xq,
        wq=wq,
        windows=windows,
        active_ids=active_ids,
        basis=basis,
        dbasis=dbasis,
    )


@functools.partial(jax.jit, static_argnames=("degree",))
def basis_poly_coeffs_in_x(cache: PowerEvalCache, degree: int) -> Array:
    degree = int(degree)
    dtype = cache.knots.dtype
    r = jnp.arange(degree + 1, dtype=dtype)
    t = 0.5 * (1.0 + jnp.cos((2.0 * r + 1.0) * jnp.pi / (2.0 * (degree + 1.0))))
    x = cache.a[:, None] + cache.sizes[:, None] * t[None, :]
    basis_s, _dbasis_s, _d2basis_s = basis_batch(x, cache.windows, degree)
    k = jnp.arange(degree + 1, dtype=dtype)
    vand = jnp.power(x[:, :, None], k[None, None, :])
    return jnp.linalg.solve(vand, basis_s)


@functools.partial(jax.jit, static_argnames=("degree",))
def assemble_reduced_system_from_cache(
    cache: PowerEvalCache,
    degree: int,
    beta: float = BETA,
) -> tuple[Array, Array]:
    n_basis = cache.active_ids.shape[0] + int(degree)
    dtype = cache.knots.dtype
    beta_t = jnp.asarray(beta, dtype=dtype)

    k_loc = jnp.einsum("eqi,eq,eqj->eij", cache.dbasis, cache.wq, cache.dbasis)
    coeffs_x = basis_poly_coeffs_in_x(cache, degree)
    k = jnp.arange(int(degree) + 1, dtype=dtype)
    alpha = beta_t - 2.0
    forcing_coeff = beta_t * (1.0 - beta_t)
    moments = monomial_integral(cache.a[:, None], cache.b[:, None], alpha + k[None, :])
    f_loc = forcing_coeff * jnp.einsum("ekj,ek->ej", coeffs_x, moments)

    k_full = jnp.zeros((n_basis, n_basis), dtype=dtype)
    k_full = k_full.at[cache.active_ids[:, :, None], cache.active_ids[:, None, :]].add(k_loc)

    f_full = jnp.zeros((n_basis,), dtype=dtype)
    f_full = f_full.at[cache.active_ids.reshape(-1)].add(f_loc.reshape(-1))
    f_full = f_full.at[-1].add(beta_t)

    return k_full[1:, 1:], f_full[1:]


@functools.partial(jax.jit, static_argnames=("degree", "quadrature_order"))
def assemble_reduced_system(
    theta: Array,
    degree: int,
    quadrature_order: int,
    h_min: float,
    beta: float = BETA,
) -> tuple[Array, Array]:
    cache = build_eval_cache(theta, degree, quadrature_order, h_min)
    return assemble_reduced_system_from_cache(cache, degree, beta)


@jax.jit
def expand_dirichlet_zero(u_free: Array) -> Array:
    return jnp.concatenate([jnp.zeros((1,), dtype=u_free.dtype), u_free], axis=0)


@functools.partial(jax.jit, static_argnames=("degree", "quadrature_order"))
def solve_state(
    theta: Array,
    degree: int,
    quadrature_order: int,
    h_min: float,
    beta: float = BETA,
) -> tuple[Array, PowerEvalCache]:
    cache = build_eval_cache(theta, degree, quadrature_order, h_min)
    k_ff, f_f = assemble_reduced_system_from_cache(cache, degree, beta)
    u_free = solve_system(k_ff, f_f)
    u_full = expand_dirichlet_zero(u_free)
    return u_full, cache


@functools.lru_cache(maxsize=8)
def monomial_fit_data_np(degree: int) -> tuple[np.ndarray, np.ndarray]:
    t_nodes = (np.arange(degree + 1, dtype=np.float64) + 0.5) / float(degree + 1)
    vand = np.vander(t_nodes, N=degree + 1, increasing=True)
    inv_vand = np.linalg.inv(vand)
    return t_nodes, inv_vand


def monomial_fit_data(degree: int, dtype=DEFAULT_DTYPE) -> tuple[Array, Array]:
    t_nodes_np, inv_vand_np = monomial_fit_data_np(int(degree))
    return jnp.asarray(t_nodes_np, dtype=dtype), jnp.asarray(inv_vand_np, dtype=dtype)


@functools.partial(jax.jit, static_argnames=("degree",))
def element_solution_coeffs_t(u_full: Array, cache: PowerEvalCache, degree: int) -> Array:
    t_nodes, inv_vand = monomial_fit_data(degree, u_full.dtype)
    x_sample = cache.a[:, None] + cache.sizes[:, None] * t_nodes[None, :]
    basis_s, _dbasis_s, _d2basis_s = basis_batch(x_sample, cache.windows, degree)
    u_loc = u_full[cache.active_ids]
    values = jnp.einsum("eqi,ei->eq", basis_s, u_loc)
    return jnp.einsum("ij,ej->ei", inv_vand, values)


def differentiate_poly_t_wrt_x(coeffs_t: Array, sizes: Array, order: int) -> Array:
    if order == 0:
        return coeffs_t

    deg = coeffs_t.shape[1] - 1
    if deg < order:
        return jnp.zeros((coeffs_t.shape[0], 1), dtype=coeffs_t.dtype)

    factors = [float(math.prod(range(k - order + 1, k + 1))) for k in range(order, deg + 1)]
    scale = jnp.asarray(factors, dtype=coeffs_t.dtype)[None, :]
    return coeffs_t[:, order:] * scale / (sizes[:, None] ** order)


__all__ = [
    "Array",
    "PowerEvalCache",
    "stiffness_quadrature_order",
    "build_eval_cache",
    "basis_poly_coeffs_in_x",
    "assemble_reduced_system_from_cache",
    "assemble_reduced_system",
    "solve_state",
    "expand_dirichlet_zero",
    "element_solution_coeffs_t",
    "differentiate_poly_t_wrt_x",
]
