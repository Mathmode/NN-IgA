"""Tests for src.nonparametric.eta_estimator_2d."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from common.h1_seminorm_2d import open_uniform_knots
from src.nonparametric.eta_estimator_2d import (
    eta_squared_arctan,
    eta_squared_lshape,
    eta_uniform_squared_arctan,
    eta_uniform_squared_lshape,
)
from src.nonparametric.r_adapt import p3_effective_n_elem, theta_to_knots_lshape
from src.nonparametric.solver_2d import galerkin_solve_arctan, galerkin_solve_lshape


# --------------------------------------------------------------------------
# arctan — basic sanity
# --------------------------------------------------------------------------


def test_p2_eta_is_positive_finite():
    sigma = jnp.array([5.0, 0.5, 0.5])
    val = eta_uniform_squared_arctan(sigma, 8, 3, q_est=20)
    val_f = float(val)
    assert val_f > 0.0
    assert np.isfinite(val_f)


def test_p2_eta_decreases_under_h_refinement():
    """For smooth α=5 manufactured solution, refining the uniform mesh
    must drop η² substantially. Test at fixed p=3."""
    sigma = jnp.array([5.0, 0.5, 0.5])
    vals = {}
    for N in [4, 8, 16]:
        vals[N] = float(eta_uniform_squared_arctan(sigma, N, 3, q_est=20))
    print(f"\nη² @ p=3 uniform (alpha=5): {vals}")
    assert vals[8] < vals[4]
    assert vals[16] < vals[8]
    # Expect at least 5× reduction from N=4 to N=16
    assert vals[16] < vals[4] / 5.0


def test_p2_eta_jit_compilable():
    """Verify that eta_squared_arctan runs inside jax.jit without errors."""
    knots = jnp.asarray(open_uniform_knots(8, 3))
    sigma = jnp.array([5.0, 0.5, 0.5])
    res = galerkin_solve_arctan(knots, knots, 3, 8, 8, sigma[0], sigma[1], sigma[2], q_K=4, q_F=20)

    @jax.jit
    def call(C, kx, ky, a, s1, s2):
        return eta_squared_arctan(C, kx, ky, 3, 8, 8, a, s1, s2, q_est=20)

    val = float(call(res.u_h, knots, knots, sigma[0], sigma[1], sigma[2]))
    assert np.isfinite(val)
    assert val > 0


# --------------------------------------------------------------------------
# arctan — autodiff correctness (CRITICAL)
# --------------------------------------------------------------------------


def _loss_p2_from_logits(logits_x, logits_y, sigma, N, p):
    """Build knots from raw "logits" via softmax+cumsum and return η²."""
    from src.nonparametric.r_adapt import theta_to_knots_arctan
    knots_x = theta_to_knots_arctan(logits_x, p)
    knots_y = theta_to_knots_arctan(logits_y, p)
    res = galerkin_solve_arctan(
        knots_x, knots_y, p, N, N,
        sigma[0], sigma[1], sigma[2], q_K=p + 1, q_F=20,
    )
    return eta_squared_arctan(
        res.u_h, knots_x, knots_y, p, N, N,
        sigma[0], sigma[1], sigma[2], q_est=20,
    )


def test_p2_eta_gradient_matches_finite_difference():
    """jax.grad of η² wrt logits matches central finite differences (rel < 1e-3)."""
    N, p = 6, 2
    sigma = jnp.array([3.0, 0.5, 0.5])
    rng = np.random.default_rng(0)
    logits_x = jnp.asarray(0.2 * rng.standard_normal(N))
    logits_y = jnp.asarray(0.15 * rng.standard_normal(N))

    grad_ad = jax.grad(lambda lx, ly: _loss_p2_from_logits(lx, ly, sigma, N, p), argnums=0)(logits_x, logits_y)
    eps = 1e-5
    grad_fd = []
    for i in range(N):
        lp = logits_x.at[i].add(eps)
        lm = logits_x.at[i].add(-eps)
        d = (_loss_p2_from_logits(lp, logits_y, sigma, N, p)
             - _loss_p2_from_logits(lm, logits_y, sigma, N, p)) / (2 * eps)
        grad_fd.append(float(d))
    grad_fd = jnp.asarray(grad_fd)
    grad_ad = jnp.asarray(grad_ad)
    num = jnp.linalg.norm(grad_ad - grad_fd)
    den = jnp.linalg.norm(grad_fd) + 1e-30
    rel = float(num / den)
    print(f"\nP2 grad FD rel error: {rel:.3e}  (||AD||={float(jnp.linalg.norm(grad_ad)):.3e})")
    assert rel < 1e-3


def test_p2_eta_vmap_matches_sequential():
    """vmap over sigma yields identical results to sequential."""
    N, p = 8, 3
    knots = jnp.asarray(open_uniform_knots(N, p))
    sigmas = jnp.array([[5.0, 0.5, 0.5],
                        [10.0, 0.3, 0.7],
                        [15.0, 0.6, 0.4]])

    def one(s):
        res = galerkin_solve_arctan(knots, knots, p, N, N, s[0], s[1], s[2], q_K=p + 1, q_F=20)
        return eta_squared_arctan(res.u_h, knots, knots, p, N, N, s[0], s[1], s[2], q_est=20)

    batched = jax.vmap(one)(sigmas)
    seq = jnp.array([one(s) for s in sigmas])
    assert jnp.allclose(batched, seq, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# lshape — sanity
# --------------------------------------------------------------------------


def test_p3_eta_is_positive_finite():
    sigma = jnp.array([1.0, 1.0])
    val = eta_uniform_squared_lshape(sigma, 10, 2, q_est=2)
    val_f = float(val)
    assert val_f > 0.0
    assert np.isfinite(val_f)


def test_p3_eta_decreases_under_h_refinement():
    """For uniform (1, 1) σ, refining the mesh must drop η²."""
    sigma = jnp.array([1.0, 1.0])
    vals = {}
    for N in [10, 20, 40]:
        vals[N] = float(eta_uniform_squared_lshape(sigma, N, 2, q_est=2))
    print(f"\nη² @ p=2 uniform (sigma=(1,1)): {vals}")
    assert vals[40] < vals[10]


def test_p3_eta_responds_to_sigma_variation():
    """η² at high-contrast (0.1, 10) should differ noticeably from (1, 1)."""
    e_uniform = float(eta_uniform_squared_lshape(jnp.array([1.0, 1.0]), 10, 2, q_est=2))
    e_contrast = float(eta_uniform_squared_lshape(jnp.array([0.1, 10.0]), 10, 2, q_est=2))
    rel = abs(e_uniform - e_contrast) / max(e_uniform, e_contrast)
    print(f"\nP3 η² sensitivity to σ: rel diff (1,1) vs (0.1, 10) = {rel:.3e}")
    assert rel > 0.1


def test_p3_eta_vmap_matches_sequential():
    N, p = 10, 2
    half = N // 2
    n_eff = p3_effective_n_elem(N, p)
    knots = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), p)
    sigmas = jnp.array([[1.0, 1.0], [0.1, 10.0], [5.0, 0.2]])

    def one(s):
        res = galerkin_solve_lshape(knots, knots, p, n_eff, n_eff, s[0], s[1], q_K=p + 1, q_F=2)
        return eta_squared_lshape(res.u_h, knots, knots, p, n_eff, n_eff, s[0], s[1], q_est=2)

    batched = jax.vmap(one)(sigmas)
    seq = jnp.array([one(s) for s in sigmas])
    assert jnp.allclose(batched, seq, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# Smoke: Loss = η²/η_unif² is bounded and around 1.0 for uniform mesh
# --------------------------------------------------------------------------


def test_p2_residual_loss_at_uniform_is_one():
    """η²(θ_uniform)/η²_uniform == 1 (sanity for the σ-balancing normalizer)."""
    N, p = 8, 3
    sigma = jnp.array([5.0, 0.5, 0.5])
    knots = jnp.asarray(open_uniform_knots(N, p))
    res = galerkin_solve_arctan(knots, knots, p, N, N, sigma[0], sigma[1], sigma[2], q_K=p + 1, q_F=20)
    num = eta_squared_arctan(res.u_h, knots, knots, p, N, N, sigma[0], sigma[1], sigma[2], q_est=20)
    den = eta_uniform_squared_arctan(sigma, N, p, q_est=20)
    rel = float(num / den)
    assert abs(rel - 1.0) < 1e-10


def test_p3_residual_loss_at_uniform_is_one():
    N, p = 10, 2
    half = N // 2
    n_eff = p3_effective_n_elem(N, p)
    sigma = jnp.array([1.0, 1.0])
    knots = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), p)
    res = galerkin_solve_lshape(knots, knots, p, n_eff, n_eff, sigma[0], sigma[1], q_K=p + 1, q_F=2)
    num = eta_squared_lshape(res.u_h, knots, knots, p, n_eff, n_eff, sigma[0], sigma[1], q_est=2)
    den = eta_uniform_squared_lshape(sigma, N, p, q_est=2)
    rel = float(num / den)
    assert abs(rel - 1.0) < 1e-10


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v", "-s"])
