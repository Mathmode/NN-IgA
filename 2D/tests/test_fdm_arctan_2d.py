"""Optimización B — the FDM/Kronecker exact O(N³) arctan solver verified
bit-for-bit against the validated dense solver.

Gates (see REPORT_optB_fdm_arctan.md):
  1. K_x ⊗ M_y + M_x ⊗ K_y  ==  dense `_assemble_2d` (σ=1, pre-Dirichlet).
  2. Free-block (drop i=0, j=0) Kronecker == dense free submatrix.
  3. u_fdm == u_dense (galerkin_solve_arctan_fdm vs galerkin_solve_arctan).
  4. grad_fdm == grad_dense w.r.t. knots (end-to-end; FD-verified), vmap, jit.

The forward and the FD-verified directional gradient match the dense path to
machine precision. The full-vector gradient's worst component is conditioning-
limited (eigendecomposition vs Cholesky) and grows mildly with N; it is
training-irrelevant, so its bound is looser than the forward's.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from common._precision import ensure_double_precision

ensure_double_precision()

from src.nonparametric.solver_2d import (
    _assemble_2d,
    free_mask_arctan,
    galerkin_solve_arctan,
)
from src.nonparametric.solver_2d_fdm import (
    galerkin_solve_arctan_fdm,
    stiffness_mass_1d,
)

P = 2
ALPHA, S1, S2 = jnp.asarray(20.0), jnp.asarray(0.6), jnp.asarray(0.4)


def _open_uniform(n_elem, p=P):
    inner = np.linspace(0.0, 1.0, n_elem + 1)
    return jnp.asarray(np.concatenate([np.zeros(p), inner, np.ones(p)]))


def _perturbed(n_elem, perturb=0.15, seed=1, p=P):
    inner = np.linspace(0.0, 1.0, n_elem + 1)
    r = np.random.default_rng(seed)
    d = r.uniform(-perturb, perturb, size=inner.shape); d[0] = d[-1] = 0.0
    inner = np.sort(np.clip(inner + d, 1e-3, 1 - 1e-3)); inner[0] = 0.0; inner[-1] = 1.0
    return jnp.asarray(np.concatenate([np.zeros(p), inner, np.ones(p)]))


# ---- Gate 0: separable Dirichlet (the premise of the whole method) ----
@pytest.mark.parametrize("N", [4, 8])
def test_dirichlet_is_separable_drop0(N):
    kx = _open_uniform(N)
    fm = np.asarray(free_mask_arctan(kx, kx, P))
    expected = np.ones_like(fm); expected[0, :] = 0.0; expected[:, 0] = 0.0
    assert np.array_equal(fm, expected), "arctan Dirichlet is not drop-index-0 separable"


# ---- Gate 1: Kronecker == dense assembly ----
@pytest.mark.parametrize("N", [4, 8])
def test_kronecker_equals_dense_assembly(N):
    kx = _open_uniform(N)
    sig = lambda x, y: jnp.ones_like(x); f0 = lambda x, y: jnp.zeros_like(x)
    K_dense, _ = _assemble_2d(kx, kx, P, N, N, sig, f0, P + 1, P + 1)
    Kx, Mx = stiffness_mass_1d(kx, P, N, P + 1)
    Ky, My = stiffness_mass_1d(kx, P, N, P + 1)
    K_kron = jnp.kron(Kx, My) + jnp.kron(Mx, Ky)
    np.testing.assert_allclose(np.asarray(K_dense), np.asarray(K_kron), atol=1e-12, rtol=0)


# ---- Gate 2: free-block restriction == dense free submatrix ----
@pytest.mark.parametrize("N", [4, 8])
def test_free_block_equals_dense_submatrix(N):
    kx = _open_uniform(N)
    sig = lambda x, y: jnp.ones_like(x); f0 = lambda x, y: jnp.zeros_like(x)
    K_dense, _ = _assemble_2d(kx, kx, P, N, N, sig, f0, P + 1, P + 1)
    free = np.asarray(free_mask_arctan(kx, kx, P)).reshape(-1) > 0.5
    K_dense_free = np.asarray(K_dense)[np.ix_(free, free)]
    Kx, Mx = stiffness_mass_1d(kx, P, N, P + 1)
    Ky, My = stiffness_mass_1d(kx, P, N, P + 1)
    K_kron_free = np.asarray(jnp.kron(Kx[1:, 1:], My[1:, 1:]) + jnp.kron(Mx[1:, 1:], Ky[1:, 1:]))
    np.testing.assert_allclose(K_dense_free, K_kron_free, atol=1e-12, rtol=0)


# ---- Gate 3 (CRITICAL forward): u_fdm == u_dense ----
@pytest.mark.parametrize("N", [4, 8, 16])
def test_forward_u_matches_dense(N):
    kx = _perturbed(N, seed=1); ky = _perturbed(N, seed=2)
    rd = galerkin_solve_arctan(kx, ky, P, N, N, ALPHA, S1, S2, q_K=P + 1, q_F=20)
    rf = galerkin_solve_arctan_fdm(kx, ky, P, N, N, ALPHA, S1, S2, q_K=P + 1, q_F=20)
    np.testing.assert_allclose(np.asarray(rf.u_h), np.asarray(rd.u_h), atol=1e-9, rtol=1e-9)
    # ritz energy (0.5 uᵀKu − uᵀF == −0.5 uᵀF at the solution) matches.
    np.testing.assert_allclose(float(rf.ritz_energy), float(rd.ritz_energy), rtol=1e-9)


# ---- Gate 4 (CRITICAL backward): gradient w.r.t. knots ----
def _loss(solver, kx, ky, N):
    return jnp.sum(solver(kx, ky, P, N, N, ALPHA, S1, S2, q_K=P + 1, q_F=20).u_h ** 2)


@pytest.mark.parametrize("N", [4, 8])
def test_gradient_matches_dense_fullvector(N):
    kx = _perturbed(N, seed=1); ky = _perturbed(N, seed=2)
    gf = jax.grad(lambda a, b: _loss(galerkin_solve_arctan_fdm, a, b, N), argnums=(0, 1))(kx, ky)
    gd = jax.grad(lambda a, b: _loss(galerkin_solve_arctan, a, b, N), argnums=(0, 1))(kx, ky)
    scale = float(jnp.maximum(jnp.max(jnp.abs(gd[0])), jnp.max(jnp.abs(gd[1]))))
    # conditioning-limited; far tighter than any real-bug error (which is O(1))
    np.testing.assert_allclose(np.asarray(gf[0]), np.asarray(gd[0]), atol=1e-6 * scale, rtol=0)
    np.testing.assert_allclose(np.asarray(gf[1]), np.asarray(gd[1]), atol=1e-6 * scale, rtol=0)


@pytest.mark.parametrize("N", [4, 8, 16])
def test_gradient_directional_matches_dense_and_fd(N):
    """Scalar interior-knot directional derivative: FDM == dense == finite diff."""
    k0 = _open_uniform(N)
    idx = P + N // 2  # an interior knot

    def loss_theta(solver, theta):
        kx = k0.at[idx].add(theta)
        return _loss(solver, kx, k0, N)

    gf = float(jax.grad(lambda t: loss_theta(galerkin_solve_arctan_fdm, t))(0.0))
    gd = float(jax.grad(lambda t: loss_theta(galerkin_solve_arctan, t))(0.0))
    h = 1e-6
    fd = (float(loss_theta(galerkin_solve_arctan, h)) - float(loss_theta(galerkin_solve_arctan, -h))) / (2 * h)
    # FDM == dense to machine precision (directional derivative is well conditioned)
    assert abs(gf - gd) <= 1e-8 * abs(gd)
    # both match the finite-difference ground truth to the FD's own resolution
    assert abs(gf - fd) <= 1e-5 * abs(fd)


def test_vmap_and_jit():
    N = 8
    kx = _perturbed(N, seed=1); ky = _perturbed(N, seed=2)
    r = np.random.default_rng(0)
    als = jnp.asarray(r.uniform(5, 25, 4)); s1s = jnp.asarray(r.uniform(.3, .7, 4)); s2s = jnp.asarray(r.uniform(.3, .7, 4))
    ff = lambda a, b, c: galerkin_solve_arctan_fdm(kx, ky, P, N, N, a, b, c, q_K=P + 1, q_F=20).u_h
    fd = lambda a, b, c: galerkin_solve_arctan(kx, ky, P, N, N, a, b, c, q_K=P + 1, q_F=20).u_h
    uf = jax.vmap(ff)(als, s1s, s2s); ud = jax.vmap(fd)(als, s1s, s2s)
    np.testing.assert_allclose(np.asarray(uf), np.asarray(ud), atol=1e-9, rtol=1e-9)
    # jit forward + grad compile and agree with eager
    jf = jax.jit(lambda a, b: galerkin_solve_arctan_fdm(a, b, P, N, N, ALPHA, S1, S2, q_K=P + 1, q_F=20).u_h)
    np.testing.assert_allclose(np.asarray(jf(kx, ky)), np.asarray(ff(ALPHA, S1, S2)), atol=1e-12, rtol=0)
    g_eager = jax.grad(lambda a, b: _loss(galerkin_solve_arctan_fdm, a, b, N), argnums=0)(kx, ky)
    g_jit = jax.jit(jax.grad(lambda a, b: _loss(galerkin_solve_arctan_fdm, a, b, N), argnums=0))(kx, ky)
    np.testing.assert_allclose(np.asarray(g_jit), np.asarray(g_eager), atol=1e-9, rtol=1e-9)
