from __future__ import annotations

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp

from src.config import EPSILON_DENOM, T_SCHEDULE
from src.nonparametric.discretization import (
    assemble_reduced_system_from_cache,
    build_eval_cache,
    stiffness_quadrature_order,
)
from src.nonparametric.solver import solve_system
from src.parametric.positional_density_network import cell_xi, forward_scalar, gauge_fix, saturate


def ritz_energy(theta, p, q_order, h_min, beta):
    cache = build_eval_cache(theta, int(p), int(q_order), float(h_min))
    K, F = assemble_reduced_system_from_cache(cache, int(p), beta)
    u = solve_system(K, F)
    return -0.5 * jnp.dot(F, u)


def network_theta(params, beta, p, N):
    z = gauge_fix(forward_scalar(params, beta, cell_xi(int(N)), int(N)))
    return saturate(z, float(T_SCHEDULE[int(p)][int(N)]))


def make_per_beta_ritz_loss(p, N, h_min):
    q = stiffness_quadrature_order(int(p))

    def per_beta(params, beta, denom):
        beta_t = jnp.asarray(beta, dtype=DEFAULT_DTYPE)
        J = ritz_energy(network_theta(params, beta_t, p, N), p, q, h_min, beta_t)
        return J / (jnp.asarray(denom, DEFAULT_DTYPE) + jnp.asarray(EPSILON_DENOM, DEFAULT_DTYPE))

    return per_beta


def make_uniform_ritz_fn(p, N, h_min):
    q = stiffness_quadrature_order(int(p))
    theta0 = jnp.zeros((int(N),), dtype=DEFAULT_DTYPE)
    one = lambda beta: jnp.abs(ritz_energy(theta0, p, q, h_min, jnp.asarray(beta, DEFAULT_DTYPE)))
    return jax.jit(jax.vmap(one))


def make_val_forward_ritz(p, N, h_min, n_sub=10):
    from src.h1_metric import _h1_error_squared
    from src.nonparametric.discretization import solve_state

    q = stiffness_quadrature_order(int(p))
    per_beta = make_per_beta_ritz_loss(p, N, h_min)

    def one(params, beta, denom):
        beta_t = jnp.asarray(beta, dtype=DEFAULT_DTYPE)
        u_full, cache = solve_state(network_theta(params, beta_t, p, N), int(p), q, float(h_min), beta_t)
        _, en_sq = _h1_error_squared(u_full, cache, int(p), beta_t, n_sub=int(n_sub))
        h1_rel = jnp.sqrt(jnp.maximum(en_sq, 0.0) / (beta_t * beta_t / (2.0 * beta_t - 1.0)))
        return per_beta(params, beta_t, denom), h1_rel

    return jax.jit(lambda params, betas, denoms: jax.vmap(lambda b, d: one(params, b, d))(betas, denoms))


__all__ = [
    "ritz_energy",
    "network_theta",
    "make_per_beta_ritz_loss",
    "make_uniform_ritz_fn",
    "make_val_forward_ritz",
]
