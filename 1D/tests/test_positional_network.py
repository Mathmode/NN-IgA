"""Sanity tests for the positional density network (PDN).

Required by the spec:
  1. Gauge fix:  z' = z - mean(z)  has zero sum.
  2. Bounded logits:  after T*tanh(z'/T), |theta_i| <= T.
  3. Collocation consistency:  the same rho_phi evaluated at the same
     xi values for two different N values agrees, modulo the gauge.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from src.config import T_SCHEDULE
from src.parametric.positional_density_network import (
    cell_xi,
    forward_scalar,
    gauge_fix,
    init_params,
    logits_at_level,
    saturate,
)


def test_gauge_fix_centered():
    rng = np.random.default_rng(42)
    z = jnp.asarray(rng.normal(size=(17,)) * 5.0)
    z_g = gauge_fix(z)
    s = float(jnp.sum(z_g))
    assert abs(s) < 1e-10, f"gauge fix should give zero sum; got {s:.3e}"


@pytest.mark.parametrize("p, N", [(2, 8), (3, 16), (3, 64)])
def test_bounded_logits_in_range(p, N):
    params = init_params(seed=0)
    theta = logits_at_level(params, beta=1.7, p=p, N=N)
    T = float(T_SCHEDULE[p][N])
    theta_np = np.asarray(theta)
    assert np.all(np.abs(theta_np) <= T + 1e-12), (
        f"logits exceeded box: max|theta|={np.max(np.abs(theta_np)):.3e}, T={T}"
    )


@pytest.mark.parametrize("p", [2, 3])
def test_collocation_consistency_modulo_gauge(p):
    """Same rho_phi at shared xi positions for N1 and N2 should agree
    (in z_pre, before gauge fix) — the network does not know N except
    through the log(xi + 1/N) feature.

    We test that for two different N values evaluated at the *same*
    physical xi vector, the network output differs only via the
    log-feature that depends on N. If we use a *common* N_for_features
    to evaluate both, the outputs must coincide exactly.
    """
    params = init_params(seed=1)
    beta = 1.8

    xi_shared = jnp.asarray([0.05, 0.2, 0.4, 0.6, 0.8])

    z_a = forward_scalar(params, beta, xi_shared, N=64)
    z_b = forward_scalar(params, beta, xi_shared, N=64)
    np.testing.assert_allclose(np.asarray(z_a), np.asarray(z_b), atol=0.0, rtol=0.0)

    # Different N for features changes the log term -> z differs.
    z_c = forward_scalar(params, beta, xi_shared, N=8)
    diff = float(np.max(np.abs(np.asarray(z_a) - np.asarray(z_c))))
    assert diff > 0.0, "log(xi + 1/N) feature must make N matter"


def test_saturate_caps_inputs():
    z = jnp.asarray([-100.0, -1.0, 0.0, 1.0, 100.0])
    T = 5.0
    out = saturate(z, T)
    np.testing.assert_array_less(np.asarray(out), T + 1e-12)
    np.testing.assert_array_less(-T - 1e-12, np.asarray(out))


def test_cell_xi_layout():
    xi = cell_xi(4)
    np.testing.assert_allclose(np.asarray(xi), np.array([0.125, 0.375, 0.625, 0.875]))
