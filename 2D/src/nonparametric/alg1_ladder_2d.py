from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

from common._precision import DEFAULT_DTYPE, ensure_double_precision
from src.config import T_ARCTAN, T_LSHAPE, T_LSHAPE_BY_P

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from src.nonparametric.lshape.schedule import policy_at
from src.nonparametric.eta_estimator_2d import eta_squared_arctan, eta_squared_lshape
from src.nonparametric.oblique import eta_squared_oblique, galerkin_solve_oblique
from src.nonparametric.r_adapt import p3_effective_n_elem, theta_to_knots_arctan, theta_to_knots_lshape
from src.nonparametric.solver_2d import galerkin_solve_arctan, galerkin_solve_lshape


@dataclass(frozen=True)
class Problem:
    n_blocks: int
    block_size: Callable[[int], int]
    knots: Callable
    solve: Callable
    eta2: Callable
    bound: Callable[[int], float]
    mesh_schedule: tuple = ()


def _tensor_knots(theta, N, p):
    tx, ty = jnp.split(theta, 2)
    return theta_to_knots_arctan(tx, p), theta_to_knots_arctan(ty, p), N


def _lshape_knots(theta, N, p, h_min=0.0):
    xl, xr, yl, yr = jnp.split(theta, 4)
    return theta_to_knots_lshape(xl, xr, p, h_min=h_min), theta_to_knots_lshape(yl, yr, p, h_min=h_min), p3_effective_n_elem(N, p)


def arctan_problem(nu, q_F=50, q_est=50):
    a, s1, s2 = (jnp.asarray(v, DEFAULT_DTYPE) for v in nu)
    return Problem(
        2, lambda N: N, _tensor_knots,
        lambda kx, ky, p, n: galerkin_solve_arctan(kx, ky, p, n, n, a, s1, s2, q_K=p + 1, q_F=q_F),
        lambda r, kx, ky, p, n: eta_squared_arctan(r.u_h, kx, ky, p, n, n, a, s1, s2, q_est=q_est),
        lambda p: float(T_ARCTAN))


def lshape_problem(nu, q_F=4, q_est=4, mesh_schedule=()):
    s1, s2 = (jnp.asarray(v, DEFAULT_DTYPE) for v in nu)
    return Problem(
        4, lambda N: N // 2,
        lambda theta, N, p: _lshape_knots(theta, N, p, policy_at(mesh_schedule, N)[1] if mesh_schedule else 0.0),
        lambda kx, ky, p, n: galerkin_solve_lshape(kx, ky, p, n, n, s1, s2, q_K=p + 1, q_F=q_F),
        lambda r, kx, ky, p, n: eta_squared_lshape(r.u_h, kx, ky, p, n, n, s1, s2, q_est=q_est),
        lambda p: float(T_LSHAPE_BY_P.get(int(p), T_LSHAPE)), mesh_schedule)


def oblique_problem(angle, q_F=20, q_est=20):
    ang = jnp.asarray(angle, DEFAULT_DTYPE)
    return Problem(
        2, lambda N: N, _tensor_knots,
        lambda kx, ky, p, n: galerkin_solve_oblique(kx, ky, p, n, n, ang, q_K=p + 1, q_F=q_F),
        lambda r, kx, ky, p, n: eta_squared_oblique(r.u_h, kx, ky, p, n, n, ang, q_est=q_est),
        lambda p: float(T_ARCTAN))


def objective(problem, loss, p, N):
    def f(theta):
        kx, ky, n = problem.knots(theta, N, p)
        r = problem.solve(kx, ky, p, n)
        return problem.eta2(r, kx, ky, p, n) if loss == "eta" else r.ritz_energy
    scale = abs(float(f(jnp.zeros(problem.n_blocks * problem.block_size(N), DEFAULT_DTYPE))))
    return jax.jit(jax.value_and_grad(lambda theta: f(theta) / scale))


def lbfgs(value_and_grad, theta0, maxiter, T):
    def fun(x):
        v, g = value_and_grad(jnp.asarray(x, DEFAULT_DTYPE))
        return float(v), np.asarray(g, dtype=np.float64)
    x0 = np.asarray(theta0, np.float64)
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=[(-float(T), float(T))] * x0.size,
                   options={"maxiter": int(maxiter), "ftol": 0.0, "gtol": 1e-9})
    return jnp.asarray(res.x, DEFAULT_DTYPE), int(res.nit), stop_reason(res, maxiter), int(np.sum(np.abs(res.x) >= float(T)))


def stop_reason(res, maxiter):
    msg = str(res.message).upper()
    if "PROJECTED_GRADIENT" in msg:
        return "gradient"
    if "ABNORMAL" in msg or "REL_REDUCTION" in msg:
        return "line_search"
    if int(res.nit) >= int(maxiter) or "ITERATIONS" in msg:
        return "maxiter"
    return msg.replace(" ", "_").lower()


def prolong(theta, n_blocks):
    return jnp.concatenate([jnp.repeat(b, 2) for b in jnp.split(theta, n_blocks)])


def run_ladder(problem, loss, p, levels, maxiter):
    theta = jnp.zeros(problem.n_blocks * problem.block_size(levels[0]), DEFAULT_DTYPE)
    out, elapsed = {}, 0.0
    for i, N in enumerate(int(n) for n in levels):
        if i > 0:
            theta = prolong(theta, problem.n_blocks)
        vg = objective(problem, loss, p, N)
        jax.block_until_ready(vg(theta))
        t0 = time.perf_counter()
        bound = policy_at(problem.mesh_schedule, N)[0] if problem.mesh_schedule else problem.bound(p)
        theta, nit, stop, n_active = lbfgs(vg, jnp.clip(theta, -bound, bound), maxiter, bound)
        elapsed += time.perf_counter() - t0
        out[N] = dict(theta=theta, time_s=elapsed, nit=nit, stop=stop, n_active=n_active)
    return out


def solution(problem, theta, p, N):
    kx, ky, n = problem.knots(theta, N, p)
    r = problem.solve(kx, ky, p, n)
    return r, kx, ky, n, float(problem.eta2(r, kx, ky, p, n))


def mesh_quality(kx, ky, p):
    hx, hy = np.diff(np.unique(np.asarray(kx))), np.diff(np.unique(np.asarray(ky)))
    w = 2 * p + 1
    loc = max(float(np.max(np.lib.stride_tricks.sliding_window_view(h, min(w, h.size)).max(1)
                            / np.lib.stride_tricks.sliding_window_view(h, min(w, h.size)).min(1))) for h in (hx, hy))
    ar = float(np.max(np.maximum(hx[:, None] / hy[None, :], hy[None, :] / hx[:, None])))
    return loc, ar


__all__ = ["Problem", "arctan_problem", "lshape_problem", "oblique_problem", "objective",
           "lbfgs", "stop_reason", "prolong", "run_ladder", "solution", "mesh_quality"]
