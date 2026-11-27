from __future__ import annotations

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np

from common.h1_seminorm_2d import open_uniform_knots
from src.nonparametric.r_adapt import p3_effective_n_elem, theta_to_knots_lshape
from src.nonparametric.solver_2d import galerkin_solve_arctan, galerkin_solve_lshape
from src.parametric.positional_density_network_2d import (
    knots_p2_from_network_axis,
    knots_p3_from_network_axis,
)


def _solve(experiment, kx, ky, p, n_elem, sigma, q_K, q_F, solver="dense"):
    if experiment == "arctan":
        if solver == "fdm":
            from src.nonparametric.solver_2d_fdm import galerkin_solve_arctan_fdm
            return galerkin_solve_arctan_fdm(kx, ky, p, n_elem, n_elem, sigma[0], sigma[1], sigma[2], q_K=q_K, q_F=q_F)
        return galerkin_solve_arctan(kx, ky, p, n_elem, n_elem, sigma[0], sigma[1], sigma[2], q_K=q_K, q_F=q_F)
    n_eff = p3_effective_n_elem(n_elem, p)
    return galerkin_solve_lshape(kx, ky, p, n_eff, n_eff, sigma[0], sigma[1], q_K=q_K, q_F=q_F)


def network_solve(params, sigma, experiment, p, n_elem, q_K, q_F, solver="dense"):
    build = knots_p2_from_network_axis if experiment == "arctan" else knots_p3_from_network_axis
    kx = build(params, sigma, n_elem, p, axis_id=0.0)
    ky = build(params, sigma, n_elem, p, axis_id=1.0)
    return _solve(experiment, kx, ky, p, n_elem, sigma, q_K, q_F, solver), kx, ky


def uniform_ritz(sigma, experiment, p, N, q_K, q_F):
    if experiment == "arctan":
        kn = jnp.asarray(open_uniform_knots(int(N), int(p)), DEFAULT_DTYPE)
    else:
        z = jnp.zeros((int(N) // 2,), DEFAULT_DTYPE)
        kn = theta_to_knots_lshape(z, z, int(p))
    return _solve(experiment, kn, kn, int(p), int(N), sigma, q_K, q_F).ritz_energy


def make_batch_loss_ritz(*, experiment, p, n_elem, q_K, q_F, epsilon=0.0, solver="dense"):
    def per(params, sigma, denom):
        r, _, _ = network_solve(params, sigma, experiment, p, n_elem, q_K, q_F, solver)
        return r.ritz_energy / (jnp.asarray(denom, DEFAULT_DTYPE) + jnp.asarray(epsilon, DEFAULT_DTYPE))

    return lambda params, sigmas, denoms: jnp.mean(jax.vmap(lambda s, d: per(params, s, d))(sigmas, denoms))


def make_val_forward_ritz(*, experiment, p, n_elem, q_K, q_F, epsilon=0.0, solver="dense"):
    def one(params, sigma, denom):
        r, kx, ky = network_solve(params, sigma, experiment, p, n_elem, q_K, q_F, solver)
        return r.ritz_energy / (jnp.asarray(denom, DEFAULT_DTYPE) + jnp.asarray(epsilon, DEFAULT_DTYPE)), r.u_h, kx, ky

    return lambda params, sigmas, denoms: jax.vmap(lambda s, d: one(params, s, d))(sigmas, denoms)


def ritz_unif_denoms(params_arr, N, p, *, experiment, q_K, q_F, chunk=32):
    f = jax.jit(jax.vmap(lambda s: jnp.abs(uniform_ritz(s, experiment, p, N, q_K, q_F))))
    arr = np.asarray(params_arr, dtype=np.float64)
    outs = [np.asarray(f(jnp.asarray(arr[i:i + chunk], DEFAULT_DTYPE))) for i in range(0, len(arr), chunk)]
    return np.concatenate(outs) if outs else np.zeros((0,))


def make_ritz_denoms_fn(train_params, val_params, *, experiment, p, q_K, q_F, chunk=32):
    cache = {}

    def denoms_fn(N):
        N = int(N)
        if N not in cache:
            kw = dict(experiment=experiment, q_K=q_K, q_F=q_F, chunk=chunk)
            cache[N] = (ritz_unif_denoms(train_params, N, p, **kw),
                        None if val_params is None else ritz_unif_denoms(val_params, N, p, **kw))
        return cache[N]

    return denoms_fn


__all__ = ["network_solve", "uniform_ritz", "make_batch_loss_ritz", "make_val_forward_ritz",
           "ritz_unif_denoms", "make_ritz_denoms_fn"]
