"""Tests for src.parametric.positional_density_network_2d."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.parametric.positional_density_network_2d import cell_midpoints, forward, init_params, knots_p2_from_network, knots_p3_from_network_axis, params_from_flat_dict, params_to_flat_dict


def test_init_params_shape_p2():
    """arctan: sigma_dim=3, input = 3 + 2 = 5, hidden 10,10, output 1."""
    p = init_params(seed=0, sigma_dim=3, hidden_dims=(10, 10))
    assert len(p.layers) == 3
    assert p.layers[0][0].shape == (5, 10)
    assert p.layers[1][0].shape == (10, 10)
    assert p.layers[2][0].shape == (10, 1)


def test_init_params_shape_p3():
    p = init_params(seed=0, sigma_dim=2, hidden_dims=(10, 10))
    assert p.layers[0][0].shape == (4, 10)


def test_forward_returns_finite():
    p = init_params(seed=0, sigma_dim=3, hidden_dims=(10, 10))
    z = forward(p, jnp.asarray([5.0, 0.5, 0.5]), cell_midpoints(8), axis_id=0.0)
    assert z.shape == (8,)
    assert np.all(np.isfinite(np.asarray(z)))


def test_lecun_init_activation_scale():
    """LeCun std=sqrt(1/fan_in) -> first layer's pre-activation typical
    magnitude ~ O(1)."""
    rng = np.random.default_rng(0)
    p = init_params(seed=0, sigma_dim=3, hidden_dims=(10, 10))
    z = []
    for _ in range(50):
        sig = jnp.asarray(rng.uniform(-1, 1, size=3))
        out = forward(p, sig, cell_midpoints(8), axis_id=0.0)
        z.append(np.asarray(out))
    z = np.concatenate(z)
    # Sanity: output not exploding (no NaN/inf, scale within an order of magnitude)
    assert np.std(z) < 5.0


def test_knots_p2_endpoints():
    p = init_params(seed=0, sigma_dim=3, hidden_dims=(10, 10))
    sigma = jnp.asarray([5.0, 0.5, 0.5])
    knots = np.asarray(knots_p2_from_network(p, sigma, 4, 2))
    assert knots.shape == (9,)
    assert np.allclose(knots[:3], 0.0)
    assert np.allclose(knots[-3:], 1.0)
    # Interior ascending
    interior = knots[3:6]
    assert np.all(np.diff(interior) > 0)


def test_knots_p3_fixed_node_at_half():
    p = init_params(seed=0, sigma_dim=2, hidden_dims=(10, 10))
    sigma = jnp.asarray([1.0, 1.0])
    knots = np.asarray(knots_p3_from_network_axis(p, sigma, 4, 2, axis_id=0.0))
    # Should have 0.5 as one of the interior knots
    interior = knots[3:-3]
    # Middle index of interior must be 0.5 (since both halves have equal element count)
    assert interior[len(interior) // 2] == pytest.approx(0.5, abs=1e-14)


def test_knots_differentiable_through_sigma():
    """The full pipeline (network -> softmax -> cumsum -> knots) is differentiable
    w.r.t. sigma."""
    p = init_params(seed=0, sigma_dim=3, hidden_dims=(10, 10))

    def knots_sum(sigma):
        knots = knots_p2_from_network(p, sigma, 8, 2)
        return jnp.sum(knots)

    g = jax.grad(knots_sum)(jnp.asarray([10.0, 0.5, 0.5]))
    assert np.all(np.isfinite(np.asarray(g)))


def test_serialization_roundtrip(tmp_path):
    p = init_params(seed=7, sigma_dim=3, hidden_dims=(10, 10))
    d = params_to_flat_dict(p)
    np.savez(tmp_path / "ckpt.npz", **d)
    loaded = np.load(tmp_path / "ckpt.npz")
    p2 = params_from_flat_dict(loaded)
    for (W1, b1), (W2, b2) in zip(p.layers, p2.layers):
        assert np.allclose(np.asarray(W1), np.asarray(W2))
        assert np.allclose(np.asarray(b1), np.asarray(b2))


def test_axis_id_changes_output():
    """For the same sigma & xi, different axis_id gives different logits."""
    p = init_params(seed=0, sigma_dim=3, hidden_dims=(10, 10))
    sigma = jnp.asarray([5.0, 0.5, 0.5])
    xi = cell_midpoints(4)
    z0 = forward(p, sigma, xi, axis_id=0.0)
    z1 = forward(p, sigma, xi, axis_id=1.0)
    diff = np.max(np.abs(np.asarray(z0) - np.asarray(z1)))
    assert diff > 1e-5


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
