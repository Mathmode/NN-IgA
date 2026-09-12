"""Tests for src.nonparametric.lshape.pde."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.nonparametric.lshape.pde import f_lshape, sigma_field


def test_sigma_field_canonical_regions():
    """sigma(0.25, 0.75) = 1 (top-left).
       sigma(0.25, 0.25) = sigma1 (bottom-left).
       sigma(0.75, 0.75) = sigma2 (top-right)."""
    sigma1, sigma2 = 0.5, 7.0
    # Top-left
    assert float(sigma_field(0.25, 0.75, sigma1, sigma2)) == pytest.approx(1.0, abs=1e-14)
    # Bottom-left
    assert float(sigma_field(0.25, 0.25, sigma1, sigma2)) == pytest.approx(sigma1, abs=1e-14)
    # Top-right
    assert float(sigma_field(0.75, 0.75, sigma1, sigma2)) == pytest.approx(sigma2, abs=1e-14)


def test_sigma_field_jit_compilable():
    @jax.jit
    def s(x, y, s1, s2):
        return sigma_field(x, y, s1, s2)
    v = float(s(jnp.asarray(0.25), jnp.asarray(0.75), 0.5, 7.0))
    assert v == pytest.approx(1.0)


def test_sigma_field_differentiable_in_parameters():
    """sigma is linear in (sigma1, sigma2), so grad_{s1} sigma = indicator of
    bottom-left region."""
    grad = jax.grad(sigma_field, argnums=(2, 3))
    g_s1, g_s2 = grad(jnp.asarray(0.25), jnp.asarray(0.25), 0.5, 7.0)
    assert float(g_s1) == pytest.approx(1.0, abs=1e-14)
    assert float(g_s2) == pytest.approx(0.0, abs=1e-14)


def test_sigma_field_vectorized():
    """sigma must broadcast over array inputs."""
    xs = jnp.array([0.25, 0.25, 0.75])
    ys = jnp.array([0.25, 0.75, 0.75])
    vals = np.asarray(sigma_field(xs, ys, 2.0, 3.0))
    expected = np.array([2.0, 1.0, 3.0])
    assert np.allclose(vals, expected, atol=1e-14)


def test_forcing_is_constant_one():
    xs = jnp.linspace(0.05, 0.95, 7)
    ys = jnp.linspace(0.05, 0.95, 7)
    XX, YY = jnp.meshgrid(xs, ys, indexing="ij")
    f = np.asarray(f_lshape(XX, YY))
    assert np.allclose(f, 1.0, atol=1e-14)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
