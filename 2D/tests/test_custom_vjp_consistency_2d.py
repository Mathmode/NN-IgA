"""Consistency tests for the 2D discrete-adjoint solve (Optimización A).

``galerkin_solve_arctan`` / ``galerkin_solve_lshape`` now solve the SPD Galerkin
system with ``common.solver_custom_vjp_2d.solve_system_2d`` (Cholesky forward +
EXPLICIT discrete-adjoint backward, reusing the forward factor) instead of
``jnp.linalg.solve`` (LU + JAX's implicit adjoint). The two are the SAME maths;
this suite proves it numerically — the central safety net of the change.

Three solve routes are compared on the SAME SPD system:
  * ``solve_system_2d``           — the custom_vjp now wired into training.
  * ``jnp.linalg.solve``          — the dense path it REPLACES.
  * ``solve_system_2d_autodiff``  — same Cholesky forward, NO custom_vjp
                                    (JAX differentiates the factorization).

A subtlety, verified here and documented for future readers
---------------------------------------------------------------------------
For the RHS gradient ``dF`` and the forward solution ``u`` all three routes
agree to machine precision. For the MATRIX gradient ``dK`` of a loss
``L(solve(K, F))``:

  * ``solve_system_2d`` and ``jnp.linalg.solve`` both return the full general-
    matrix adjoint ``dK = -lam u^T`` (NOT symmetric in general) — and they match
    each other to machine precision. This is the gradient training actually
    sees, and it is identical before/after the swap.
  * ``solve_system_2d_autodiff`` differentiates through ``jnp.linalg.cholesky``,
    whose VJP returns the SYMMETRIC gradient, i.e. ``dK_autodiff == sym(dK)``.

The full-matrix difference between the custom_vjp and the Cholesky-autodiff
reference is therefore the ANTISYMMETRIC part of ``-lam u^T`` — a convention
artifact, not an error. It is invisible to training: the stiffness matrix is a
symmetric function of the mesh knots, so ``dK/dknots`` is symmetric and the
antisymmetric part of ``dK`` contracts to zero. We assert this end-to-end
(``test_end_to_end_symmetric_K_gradient_identical``): for any symmetric ``K(theta)``
the parameter gradient is bit-identical across all three routes.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

# Importing the module enables float64 (ensure_double_precision) and is the same
# entry point training now uses.
from common.solver_custom_vjp_2d import solve_system_2d, solve_system_2d_autodiff

FWD_TOL = 1e-10      # forward solve: Cholesky vs LU -> machine precision in double
GRAD_TOL = 1e-9      # gradients: machine precision in double

_ROUTES = {
    "custom": solve_system_2d,
    "linalg": lambda K, F: jnp.linalg.solve(K, F),
    "autodiff": solve_system_2d_autodiff,
}


def _sym(A):
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def _spd_synthetic(n, seed):
    """Well-conditioned SPD K = A A^T + n I and a RHS F."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    K = A @ A.T + n * np.eye(n)
    F = rng.standard_normal(n)
    return jnp.asarray(K), jnp.asarray(F)


def _open_uniform_knots(n_elem, p):
    inner = np.linspace(0.0, 1.0, n_elem + 1)
    return jnp.asarray(np.concatenate([np.zeros(p), inner, np.ones(p)]))


def _spd_real_arctan(p=2, N=4):
    """A REAL post-Dirichlet SPD (K, F) assembled by galerkin_solve_arctan."""
    from src.nonparametric.solver_2d import galerkin_solve_arctan
    kx = _open_uniform_knots(N, p)
    ky = _open_uniform_knots(N, p)
    res = galerkin_solve_arctan(
        kx, ky, p, N, N,
        jnp.asarray(5.0), jnp.asarray(0.5), jnp.asarray(0.5),
        q_K=p + 1, q_F=10,
    )
    K = jnp.asarray(np.asarray(res.K))
    F = jnp.asarray(np.asarray(res.F))
    return K, F


def _cases():
    return {"synthetic_n12": _spd_synthetic(12, 0),
            "real_arctan_N4": _spd_real_arctan()}


def _grads(K, F):
    """(dK, dF) of L = sum(solve(K,F)**2) for each route, as numpy."""
    out = {}
    for name, solve in _ROUTES.items():
        loss = lambda K_, F_, s=solve: jnp.sum(s(K_, F_) ** 2)
        dK, dF = jax.grad(loss, argnums=(0, 1))(K, F)
        out[name] = (np.asarray(dK), np.asarray(dF))
    return out


# ---------------------------------------------------------------------------
# 2.1  Forward identical
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", list(_cases()))
def test_forward_identical(case):
    K, F = _cases()[case]
    u = {k: np.asarray(s(K, F)) for k, s in _ROUTES.items()}
    np.testing.assert_allclose(u["custom"], u["linalg"], atol=FWD_TOL, rtol=FWD_TOL)
    np.testing.assert_allclose(u["custom"], u["autodiff"], atol=FWD_TOL, rtol=FWD_TOL)


# ---------------------------------------------------------------------------
# 2.2  Backward: dF identical across ALL three; dK custom==linalg (full),
#      and sym(dK) consistent with the Cholesky-autodiff reference.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", list(_cases()))
def test_dF_identical(case):
    K, F = _cases()[case]
    g = _grads(K, F)
    np.testing.assert_allclose(g["custom"][1], g["linalg"][1], atol=GRAD_TOL, rtol=GRAD_TOL)
    np.testing.assert_allclose(g["custom"][1], g["autodiff"][1], atol=GRAD_TOL, rtol=GRAD_TOL)


@pytest.mark.parametrize("case", list(_cases()))
def test_dK_custom_equals_linalg_full(case):
    """The custom_vjp reproduces the EXACT general-matrix adjoint that
    jnp.linalg.solve used — so swapping the solver leaves training gradients
    bit-identical (full dK, not just the symmetric part)."""
    K, F = _cases()[case]
    g = _grads(K, F)
    np.testing.assert_allclose(g["custom"][0], g["linalg"][0], atol=GRAD_TOL, rtol=GRAD_TOL)


@pytest.mark.parametrize("case", list(_cases()))
def test_dK_symmetric_part_matches_autodiff(case):
    """The Cholesky-autodiff reference symmetrizes dK; its value equals the
    symmetric part of the custom_vjp's dK to machine precision."""
    K, F = _cases()[case]
    g = _grads(K, F)
    dK_custom, dK_autodiff, dK_linalg = g["custom"][0], g["autodiff"][0], g["linalg"][0]
    # autodiff route returns a symmetric dK
    np.testing.assert_allclose(dK_autodiff, _sym(dK_autodiff), atol=GRAD_TOL, rtol=GRAD_TOL)
    # and it equals sym(custom) == sym(linalg)
    np.testing.assert_allclose(_sym(dK_custom), dK_autodiff, atol=GRAD_TOL, rtol=GRAD_TOL)
    np.testing.assert_allclose(_sym(dK_linalg), dK_autodiff, atol=GRAD_TOL, rtol=GRAD_TOL)


# ---------------------------------------------------------------------------
# The training-faithful proof: for a SYMMETRIC K(theta) (as assembled from
# knots), the parameter gradient is identical across all three solve routes —
# the antisymmetric dK convention difference contracts to zero.
# ---------------------------------------------------------------------------
def test_end_to_end_symmetric_K_gradient_identical():
    n = 12
    rng = np.random.default_rng(7)
    A0 = jnp.asarray(rng.standard_normal((n, n)))
    A1 = jnp.asarray(rng.standard_normal((n, n)))
    F = jnp.asarray(rng.standard_normal(n))
    eye = jnp.eye(n)

    def pipeline(solve, theta):
        A = A0 + theta * A1
        K = A @ A.T + n * eye          # symmetric SPD function of theta
        return jnp.sum(solve(K, F) ** 2)

    theta0 = 0.3
    grads = {k: float(jax.grad(lambda t, s=s: pipeline(s, t))(theta0))
             for k, s in _ROUTES.items()}
    assert abs(grads["custom"] - grads["linalg"]) <= GRAD_TOL * (1 + abs(grads["linalg"]))
    assert abs(grads["custom"] - grads["autodiff"]) <= GRAD_TOL * (1 + abs(grads["autodiff"]))


# ---------------------------------------------------------------------------
# 2.3  vmap compatibility (training maps over parameter samples).
# ---------------------------------------------------------------------------
def test_vmap_forward_and_grad():
    n, B = 10, 5
    Ks, Fs = [], []
    for b in range(B):
        K, F = _spd_synthetic(n, 100 + b)
        Ks.append(K); Fs.append(F)
    K_batch = jnp.stack(Ks); F_batch = jnp.stack(Fs)   # (B,n,n), (B,n)

    # forward under vmap matches the dense path sample-by-sample
    u_custom = np.asarray(jax.vmap(solve_system_2d)(K_batch, F_batch))
    u_linalg = np.asarray(jax.vmap(lambda K, F: jnp.linalg.solve(K, F))(K_batch, F_batch))
    np.testing.assert_allclose(u_custom, u_linalg, atol=FWD_TOL, rtol=FWD_TOL)

    # gradient of a loss that uses vmap internally
    def batched_loss(solve, K_b, F_b):
        return jnp.sum(jax.vmap(solve)(K_b, F_b) ** 2)
    dK_c, dF_c = jax.grad(lambda K, F: batched_loss(solve_system_2d, K, F), argnums=(0, 1))(K_batch, F_batch)
    dK_l, dF_l = jax.grad(lambda K, F: batched_loss(lambda a, b: jnp.linalg.solve(a, b), K, F), argnums=(0, 1))(K_batch, F_batch)
    np.testing.assert_allclose(np.asarray(dF_c), np.asarray(dF_l), atol=GRAD_TOL, rtol=GRAD_TOL)
    np.testing.assert_allclose(np.asarray(dK_c), np.asarray(dK_l), atol=GRAD_TOL, rtol=GRAD_TOL)


# ---------------------------------------------------------------------------
# 2.4  jit compatibility.
# ---------------------------------------------------------------------------
def test_jit_forward_and_grad():
    K, F = _spd_synthetic(12, 3)
    u_eager = np.asarray(solve_system_2d(K, F))
    u_jit = np.asarray(jax.jit(solve_system_2d)(K, F))
    np.testing.assert_allclose(u_eager, u_jit, atol=FWD_TOL, rtol=FWD_TOL)

    loss = lambda K_, F_: jnp.sum(solve_system_2d(K_, F_) ** 2)
    g_eager = jax.grad(loss, argnums=(0, 1))(K, F)
    g_jit = jax.jit(jax.grad(loss, argnums=(0, 1)))(K, F)
    np.testing.assert_allclose(np.asarray(g_eager[0]), np.asarray(g_jit[0]), atol=GRAD_TOL, rtol=GRAD_TOL)
    np.testing.assert_allclose(np.asarray(g_eager[1]), np.asarray(g_jit[1]), atol=GRAD_TOL, rtol=GRAD_TOL)
