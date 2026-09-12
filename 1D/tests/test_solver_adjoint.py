"""Finite-difference verification of the adjoint gradient.

The training loss `reduced_loss(theta, p, q, h_min, beta)` flows through the
custom_vjp Cholesky solve in `src.nonparametric.exp1.solver`. JAX autodiff
through this primitive must agree with central finite differences.

Pass criterion: relative error < 1e-4 across a small parameter sweep.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from common._precision import DEFAULT_DTYPE
from src.nonparametric.quadrature_analytic import reduced_loss


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float(np.linalg.norm(a - b))


def _central_diff(loss_fn, theta, h):
    n = theta.shape[0]
    g = np.zeros(n, dtype=np.float64)
    for i in range(n):
        e = jnp.zeros_like(theta).at[i].set(h)
        fp = float(loss_fn(theta + e))
        fm = float(loss_fn(theta - e))
        g[i] = (fp - fm) / (2.0 * h)
    return g


@pytest.mark.parametrize("p, N", [(2, 8), (2, 16), (3, 8), (3, 16)])
def test_adjoint_matches_fd(p, N):
    rng = np.random.default_rng(0)
    h_min = 1e-8
    q_order = p + 1
    beta = 1.7
    h_fd = 1e-5

    def loss(theta):
        return reduced_loss(theta, p, q_order, h_min, beta=beta)

    grad_fn = jax.grad(loss)

    worst = 0.0
    for seed in range(2):
        theta = jnp.asarray(rng.standard_normal(N) * 0.5, dtype=DEFAULT_DTYPE)
        g_auto = np.asarray(jax.device_get(grad_fn(theta)), dtype=np.float64)
        g_fd = _central_diff(loss, theta, h_fd)
        rel = _rel_err(g_auto, g_fd)
        worst = max(worst, rel)
    assert worst < 1e-4, (
        f"adjoint vs FD relative error too large: {worst:.3e} (target < 1e-4) "
        f"for (p={p}, N={N})"
    )
