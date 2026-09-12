"""Tests for src.nonparametric.arctan.pde."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.nonparametric.arctan.pde import (
    d2u_axis,
    du_axis,
    f_sigma,
    g_sigma_neumann_right,
    g_sigma_neumann_top,
    grad_u_sigma,
    u_axis,
    u_sigma,
)


def test_axis_u_value_at_zero_is_zero():
    """u_j(0) = arctan(-alpha s) + arctan(alpha s) = 0."""
    for s in (0.1, 0.5, 0.9):
        for a in (1.0, 5.0, 20.0):
            assert float(u_axis(jnp.asarray(0.0), a, s)) == pytest.approx(0.0, abs=1e-14)


def test_du_axis_matches_jax_grad():
    """Analytic du = jax.grad(u)."""
    for s, a in [(0.5, 10.0), (0.3, 5.0), (0.9, 20.0)]:
        for t in (0.0, 0.1, 0.5, 0.7, 1.0):
            tt = jnp.asarray(t)
            du_ana = float(du_axis(tt, a, s))
            du_ad = float(jax.grad(lambda x: u_axis(x, a, s))(tt))
            assert du_ana == pytest.approx(du_ad, rel=1e-12, abs=1e-12)


def test_d2u_axis_matches_jax_grad():
    """Analytic d2u = jax.grad(jax.grad(u))."""
    for s, a in [(0.5, 10.0), (0.3, 5.0), (0.9, 20.0)]:
        for t in (0.0, 0.1, 0.5, 0.7, 1.0):
            tt = jnp.asarray(t)
            d2u_ana = float(d2u_axis(tt, a, s))
            d2u_ad = float(jax.grad(jax.grad(lambda x: u_axis(x, a, s)))(tt))
            assert d2u_ana == pytest.approx(d2u_ad, rel=1e-10, abs=1e-12)


def test_u_sigma_zero_on_left_bottom():
    """u_sigma(0, y) = 0 and u_sigma(x, 0) = 0 for all y, x."""
    for y in (0.1, 0.3, 0.7, 0.9):
        for a in (1.0, 5.0, 20.0):
            for s in [(0.3, 0.4), (0.5, 0.5), (0.7, 0.8)]:
                s1, s2 = s
                v = float(u_sigma(jnp.asarray(0.0), jnp.asarray(y), a, s1, s2))
                assert v == pytest.approx(0.0, abs=1e-14)
                v = float(u_sigma(jnp.asarray(y), jnp.asarray(0.0), a, s1, s2))
                assert v == pytest.approx(0.0, abs=1e-14)


def test_grad_u_sigma_matches_jax_grad():
    """grad_u_sigma matches jax.grad of u_sigma componentwise."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.05, 0.95, size=(20, 2))
    for s1, s2, a in [(0.3, 0.5, 5.0), (0.5, 0.5, 10.0), (0.7, 0.2, 20.0)]:
        for p in pts:
            x, y = jnp.asarray(p[0]), jnp.asarray(p[1])
            gx, gy = grad_u_sigma(x, y, a, s1, s2)
            gx_ad = jax.grad(lambda xx: u_sigma(xx, y, a, s1, s2))(x)
            gy_ad = jax.grad(lambda yy: u_sigma(x, yy, a, s1, s2))(y)
            assert float(gx) == pytest.approx(float(gx_ad), rel=1e-12, abs=1e-12)
            assert float(gy) == pytest.approx(float(gy_ad), rel=1e-12, abs=1e-12)


def test_f_sigma_equals_minus_laplacian_via_jax():
    """f_sigma analytic = -(d^2 u/dx^2 + d^2 u/dy^2) computed by autodiff,
    at 50 random points and 3 sigmas."""
    rng = np.random.default_rng(42)
    pts = rng.uniform(0.05, 0.95, size=(50, 2))
    cases = [
        (5.0, 0.3, 0.5),
        (10.0, 0.5, 0.5),
        (20.0, 0.7, 0.2),
    ]
    for a, s1, s2 in cases:
        for p in pts:
            x, y = jnp.asarray(p[0]), jnp.asarray(p[1])
            u_xx = jax.grad(jax.grad(lambda xx: u_sigma(xx, y, a, s1, s2)))(x)
            u_yy = jax.grad(jax.grad(lambda yy: u_sigma(x, yy, a, s1, s2)))(y)
            lap = float(u_xx + u_yy)
            f_ana = float(f_sigma(x, y, a, s1, s2))
            assert f_ana == pytest.approx(-lap, rel=1e-9, abs=1e-10), (
                f"alpha={a} s={s1, s2} pt={p}: f_ana={f_ana} -Δu={-lap}"
            )


def test_neumann_data_matches_gradient():
    """g_sigma at x=1: should equal partial_x u at x=1. Same on y=1."""
    a, s1, s2 = 10.0, 0.5, 0.5
    ys = jnp.linspace(0.1, 0.9, 5)
    for y in ys:
        g = float(g_sigma_neumann_right(y, a, s1, s2))
        gx, _ = grad_u_sigma(jnp.asarray(1.0), y, a, s1, s2)
        assert g == pytest.approx(float(gx), rel=1e-12, abs=1e-12)
    xs = jnp.linspace(0.1, 0.9, 5)
    for x in xs:
        g = float(g_sigma_neumann_top(x, a, s1, s2))
        _, gy = grad_u_sigma(x, jnp.asarray(1.0), a, s1, s2)
        assert g == pytest.approx(float(gy), rel=1e-12, abs=1e-12)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
