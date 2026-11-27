from __future__ import annotations

import time

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.config import H_MIN_SCHEDULE, T_SCHEDULE
from src.nonparametric.discretization import solve_state, stiffness_quadrature_order
from src.nonparametric.quadrature_analytic import residual_loss_terms_from_state
from src.parametric.ritz_loss import ritz_energy


def h_min_for(p, N):
    return float(H_MIN_SCHEDULE[int(p)].get(int(N), 1e-8))


def bound_for(p, N):
    return float(T_SCHEDULE[int(p)][int(N)])


def eta_objective(p, N, beta):
    q, h = stiffness_quadrature_order(int(p)), h_min_for(p, N)

    def f(theta):
        u, cache = solve_state(theta, int(p), q, h, beta)
        return residual_loss_terms_from_state(u, cache, int(p), beta).total_loss

    return f


def ritz_objective(p, N, beta):
    q, h = stiffness_quadrature_order(int(p)), h_min_for(p, N)
    return lambda theta: ritz_energy(theta, int(p), q, h, beta)


OBJECTIVES = {"eta": eta_objective, "ritz": ritz_objective}


_COMPILED = {}


def compiled_objective(kind, p, N):
    key = (kind, int(p), int(N))
    if key not in _COMPILED:
        q, h = stiffness_quadrature_order(int(p)), h_min_for(p, N)
        if kind == "eta":
            def f(theta, beta):
                u, cache = solve_state(theta, int(p), q, h, beta)
                return residual_loss_terms_from_state(u, cache, int(p), beta).total_loss
        else:
            def f(theta, beta):
                return ritz_energy(theta, int(p), q, h, beta)
        _COMPILED[key] = (jax.jit(f), jax.jit(jax.value_and_grad(lambda theta, beta, scale: f(theta, beta) / scale)))
    return _COMPILED[key]


def _scale(kind, p, N, beta):
    f, _ = compiled_objective(kind, p, N)
    return abs(float(f(jnp.zeros((int(N),), dtype=DEFAULT_DTYPE), float(beta))))


def normalized_objective(kind, p, N, beta):
    f, _ = compiled_objective(kind, p, N)
    scale = _scale(kind, p, N, beta)
    return lambda theta: f(theta, float(beta)) / scale


def normalized_value_and_grad(kind, p, N, beta):
    _, vg = compiled_objective(kind, p, N)
    b, scale = float(beta), _scale(kind, p, N, beta)
    return lambda theta: vg(theta, b, scale)


def adam_minimize(value_and_grad, theta0, iters, T, lr=1e-2, decay=1e-2):
    opt = optax.adam(optax.exponential_decay(lr, int(iters), decay, end_value=lr * decay))
    state = opt.init(theta0)
    theta, best_val, best_theta = theta0, np.inf, theta0
    for _ in range(int(iters)):
        v, g = value_and_grad(theta)
        v = float(v)
        if np.isfinite(v) and v < best_val:
            best_val, best_theta = v, theta
        upd, state = opt.update(g, state, theta)
        theta = jnp.clip(optax.apply_updates(theta, upd), -T, T)
    return best_theta


def prolong(theta):
    return jnp.repeat(jnp.asarray(theta), 2)


def run_ladder(kind, p, beta, levels, iters, lr=1e-2):
    theta = jnp.zeros((int(levels[0]),), dtype=DEFAULT_DTYPE)
    out, elapsed = {}, 0.0
    for i, N in enumerate(int(n) for n in levels):
        if i > 0:
            theta = prolong(theta)
        vg = normalized_value_and_grad(kind, p, N, beta)
        jax.block_until_ready(vg(theta))
        t0 = time.perf_counter()
        theta = adam_minimize(vg, theta, iters, bound_for(p, N), lr)
        theta.block_until_ready()
        elapsed += time.perf_counter() - t0
        out[N] = (theta, elapsed)
    return out


def run_cold(kind, p, beta, N, iters, lr=1e-2):
    vg = normalized_value_and_grad(kind, p, N, beta)
    theta0 = jnp.zeros((int(N),), dtype=DEFAULT_DTYPE)
    jax.block_until_ready(vg(theta0))
    t0 = time.perf_counter()
    theta = adam_minimize(vg, theta0, iters, bound_for(p, N), lr)
    theta.block_until_ready()
    return theta, time.perf_counter() - t0


__all__ = ["h_min_for", "bound_for", "eta_objective", "ritz_objective", "compiled_objective", "normalized_objective",
           "normalized_value_and_grad", "adam_minimize",
           "prolong", "run_ladder", "run_cold"]
