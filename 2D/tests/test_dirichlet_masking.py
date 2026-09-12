"""Tests for src.shared.dirichlet_masking."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from common.dirichlet_masking import (
    free_node_mask,
    lshape_dirichlet_node_mask,
    lshape_removed_quadrant_mask,
    sign_indicator,
    step_ge,
    step_le,
)


def test_sign_indicator_canonical_points():
    # x clearly below threshold -> ~0
    assert float(sign_indicator(jnp.asarray(0.3), 0.5)) == pytest.approx(0.0, abs=1e-14)
    # x clearly above -> ~1
    assert float(sign_indicator(jnp.asarray(0.7), 0.5)) == pytest.approx(1.0, abs=1e-14)
    # x exactly at threshold -> 0.5 (this case is handled separately in practice)
    assert float(sign_indicator(jnp.asarray(0.5), 0.5)) == pytest.approx(0.5, abs=1e-14)


def test_removed_quadrant_mask_canonical_points():
    """Strict-interior canonical points per the prompt's examples."""
    # Inside removed quadrant: (0.7, 0.3) -> 1
    assert float(lshape_removed_quadrant_mask(0.7, 0.3)) == pytest.approx(1.0, abs=1e-14)
    # Outside (top-left): (0.3, 0.7) -> 0
    assert float(lshape_removed_quadrant_mask(0.3, 0.7)) == pytest.approx(0.0, abs=1e-14)
    # Outside (bottom-left): (0.3, 0.3) -> 0
    assert float(lshape_removed_quadrant_mask(0.3, 0.3)) == pytest.approx(0.0, abs=1e-14)
    # Outside (top-right): (0.7, 0.7) -> 0
    assert float(lshape_removed_quadrant_mask(0.7, 0.7)) == pytest.approx(0.0, abs=1e-14)


def test_removed_quadrant_mask_includes_internal_boundary():
    """Closed-region semantics: nodes on the L-shape boundary segments
    {x=0.5, y<=0.5} U {x>=0.5, y=0.5} are tagged Dirichlet."""
    # Right edge of bottom-left square (= L-shape boundary)
    assert float(lshape_removed_quadrant_mask(0.5, 0.3)) == pytest.approx(1.0, abs=1e-14)
    # Bottom edge of top-right square (= L-shape boundary)
    assert float(lshape_removed_quadrant_mask(0.7, 0.5)) == pytest.approx(1.0, abs=1e-14)
    # Reentrant corner: included (u = 0)
    assert float(lshape_removed_quadrant_mask(0.5, 0.5)) == pytest.approx(1.0, abs=1e-14)
    # But (0.5, y > 0.5) is FREE (left edge of top-left square is interior of L-shape)
    assert float(lshape_removed_quadrant_mask(0.5, 0.7)) == pytest.approx(0.0, abs=1e-14)
    # And (x < 0.5, 0.5) is FREE (bottom edge of top-left is interior of L-shape)
    assert float(lshape_removed_quadrant_mask(0.3, 0.5)) == pytest.approx(0.0, abs=1e-14)


def test_step_ge_le_basic():
    assert float(step_ge(0.7, 0.5)) == 1.0
    assert float(step_ge(0.5, 0.5)) == 1.0      # closed
    assert float(step_ge(0.3, 0.5)) == 0.0
    assert float(step_le(0.3, 0.5)) == 1.0
    assert float(step_le(0.5, 0.5)) == 1.0      # closed
    assert float(step_le(0.7, 0.5)) == 0.0


def test_dirichlet_mask_includes_external_boundary():
    # External boundary of unit square is Dirichlet=0.
    for xy in [(0.0, 0.4), (1.0, 0.7), (0.3, 0.0), (0.3, 1.0), (0.0, 0.0)]:
        m = float(lshape_dirichlet_node_mask(xy[0], xy[1]))
        assert m == pytest.approx(1.0, abs=1e-14), f"node {xy} not flagged"


def test_dirichlet_mask_excludes_interior_free_nodes():
    # Interior nodes outside the removed quadrant should be FREE.
    for xy in [(0.3, 0.4), (0.4, 0.7), (0.6, 0.8), (0.25, 0.25)]:
        m = float(lshape_dirichlet_node_mask(xy[0], xy[1]))
        assert m == pytest.approx(0.0, abs=1e-14), f"node {xy} wrongly flagged"


def test_free_mask_is_complement():
    xs = np.array([0.3, 0.7, 0.0, 1.0, 0.4])
    ys = np.array([0.4, 0.3, 0.5, 0.5, 0.6])
    d = np.asarray(lshape_dirichlet_node_mask(xs, ys))
    f = np.asarray(free_node_mask(xs, ys))
    assert np.allclose(d + f, 1.0, atol=1e-14)


def test_mask_jit_compilable():
    """Mask must be usable inside jax.jit."""
    @jax.jit
    def masked_sum(xs, ys, vals):
        m = lshape_dirichlet_node_mask(xs, ys)
        return jnp.sum(vals * m)

    xs = jnp.linspace(0.0, 1.0, 11)
    ys = jnp.linspace(0.0, 1.0, 11)
    XX, YY = jnp.meshgrid(xs, ys, indexing="ij")
    out = masked_sum(XX.reshape(-1), YY.reshape(-1), jnp.ones(11 * 11))
    assert float(out) > 0.0


def test_mask_differentiable_through_coords():
    """The mask must be differentiable through jax.grad (gradient is zero
    on the interior of each region, which is the expected r-adapt behaviour)."""
    def loss(x, y):
        return lshape_removed_quadrant_mask(x, y).sum()

    g = jax.grad(loss, argnums=(0, 1))(jnp.asarray(0.7), jnp.asarray(0.3))
    # Should produce finite gradients (zero on the interior under sign trick)
    for gi in g:
        assert jnp.isfinite(gi)


def test_removed_quadrant_on_grid():
    """Validate mask on a 5x5 nodal grid against the CLOSED-quadrant rule."""
    xs = np.linspace(0, 1, 5)
    ys = np.linspace(0, 1, 5)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    mask = np.asarray(lshape_removed_quadrant_mask(XX, YY))
    # Closed quadrant: x >= 0.5 AND y <= 0.5.
    expected = ((XX >= 0.5 - 1e-12) & (YY <= 0.5 + 1e-12)).astype(np.float64)
    assert np.allclose(mask, expected, atol=1e-14)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
