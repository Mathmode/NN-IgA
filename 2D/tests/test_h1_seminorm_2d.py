"""Tests for src.shared.h1_seminorm_2d."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from common.h1_seminorm_2d import (
    h1_seminorm_sq_2d,
    h1_seminorm_sq_analytic,
    n_basis_open_uniform,
    open_uniform_knots,
)


# --------------------------------------------------------------------------
# Knot vector sanity
# --------------------------------------------------------------------------


def test_open_uniform_knots_p2_n4():
    knots = open_uniform_knots(4, 2)
    # Multiplicity (p+1=3) at both endpoints.
    assert np.allclose(knots[:3], 0.0)
    assert np.allclose(knots[-3:], 1.0)
    # Interior: 0.25, 0.5, 0.75
    assert np.allclose(knots[3:6], [0.25, 0.5, 0.75])
    # Length = n_elem + 2 p + 1 = 4 + 5 = 9
    assert knots.shape == (9,)


def test_n_basis_count():
    # For open-uniform with n_elem elements and degree p: n_func = n_elem + p
    assert n_basis_open_uniform(4, 2) == 6
    assert n_basis_open_uniform(8, 3) == 11


# --------------------------------------------------------------------------
# Seminorm-zero tests (u_h matches analytic exactly)
# --------------------------------------------------------------------------


def test_constant_function_zero_seminorm():
    """u_h ≡ c (all coefs = c), analytic gradient = 0 -> seminorm = 0."""
    n_elem = 4
    p = 2
    knots = open_uniform_knots(n_elem, p)
    n = n_basis_open_uniform(n_elem, p)
    C = 3.14 * np.ones((n, n), dtype=np.float64)  # any constant
    grad_zero = lambda x, y: (jnp.zeros_like(x), jnp.zeros_like(y))
    val = h1_seminorm_sq_2d(C, knots, knots, p, grad_zero, quad_points_per_dim=8)
    assert val < 1e-22, f"expected ~0, got {val}"


def test_zero_solution_matches_analytic_seminorm():
    """u_h ≡ 0, analytic gradient -> seminorm reduces to ||grad u||^2."""
    n_elem = 4
    p = 2
    knots = open_uniform_knots(n_elem, p)
    n = n_basis_open_uniform(n_elem, p)
    C = np.zeros((n, n), dtype=np.float64)

    # Smooth gradient field: u = sin(pi x) sin(pi y)
    # grad = (pi cos(pi x) sin(pi y), pi sin(pi x) cos(pi y))
    pi = float(jnp.pi)

    def grad(x, y):
        return (pi * jnp.cos(pi * x) * jnp.sin(pi * y),
                pi * jnp.sin(pi * x) * jnp.cos(pi * y))

    val_fe = h1_seminorm_sq_2d(C, knots, knots, p, grad, quad_points_per_dim=16)
    val_ref = h1_seminorm_sq_analytic(grad, quad_points_per_dim=16, n_subdiv_per_dim=4)
    # Analytic value: int_0^1 int_0^1 (pi^2 (cos^2 sin^2 + sin^2 cos^2)) dx dy
    # = pi^2 * 2 * (1/4) = pi^2 / 2.
    expected = pi * pi / 2.0
    assert val_ref == pytest.approx(expected, rel=1e-12)
    assert val_fe == pytest.approx(expected, rel=1e-8)


# --------------------------------------------------------------------------
# Arctangent (sharp at alpha=20) — quadrature convergence
# --------------------------------------------------------------------------


def _arctan_grad(alpha, s1, s2):
    """grad of u(x, y) = (atan(alpha (x - s1)) + atan(alpha s1))
                            * (atan(alpha (y - s2)) + atan(alpha s2))."""
    a = float(alpha)
    s1 = float(s1)
    s2 = float(s2)

    def g(x, y):
        u_x = jnp.arctan(a * (x - s1)) + jnp.arctan(jnp.asarray(a * s1, dtype=x.dtype))
        u_y = jnp.arctan(a * (y - s2)) + jnp.arctan(jnp.asarray(a * s2, dtype=y.dtype))
        du_x_dx = a / (1.0 + (a * (x - s1)) ** 2)
        du_y_dy = a / (1.0 + (a * (y - s2)) ** 2)
        return (du_x_dx * u_y, u_x * du_y_dy)

    return g


def test_arctan_q50_vs_q80_convergence():
    """For alpha=20 (sharp), GL 50^2 and 80^2 on a 32x32 sub-cell tessellation
    must agree to < 1e-10 in relative norm."""
    grad = _arctan_grad(alpha=20.0, s1=0.5, s2=0.5)
    val50 = h1_seminorm_sq_analytic(grad, quad_points_per_dim=50, n_subdiv_per_dim=32)
    val80 = h1_seminorm_sq_analytic(grad, quad_points_per_dim=80, n_subdiv_per_dim=32)
    rel = abs(val50 - val80) / abs(val80)
    assert rel < 1e-10, f"q=50 vs q=80 relative gap = {rel:.3e}"


def test_arctan_subdiv_refinement_converges():
    """At alpha=10, refinement of the sub-cell tessellation (16x16 -> 64x64)
    yields a value that converges to < 1e-12 rel diff."""
    grad = _arctan_grad(alpha=10.0, s1=0.3, s2=0.6)
    v_coarse = h1_seminorm_sq_analytic(grad, quad_points_per_dim=50, n_subdiv_per_dim=16)
    v_fine = h1_seminorm_sq_analytic(grad, quad_points_per_dim=50, n_subdiv_per_dim=64)
    rel = abs(v_coarse - v_fine) / abs(v_fine)
    assert rel < 1e-12, f"16x16 vs 64x64 relative gap = {rel:.3e}"


# --------------------------------------------------------------------------
# Hand-computed sanity on a single quadrant
# --------------------------------------------------------------------------


def test_seminorm_of_planar_gradient_single_elem():
    """For grad(u) = (1, 2) constant on Omega = (0, 1)^2: seminorm^2 = 1 + 4 = 5."""
    grad = lambda x, y: (jnp.ones_like(x), 2.0 * jnp.ones_like(y))
    val = h1_seminorm_sq_analytic(grad, quad_points_per_dim=4, n_subdiv_per_dim=2)
    assert val == pytest.approx(5.0, abs=1e-14)


def test_seminorm_against_fe_with_constant_solution():
    """Pass u_h = 0 with analytic grad = (1, 2): same answer as the analytic ref."""
    n_elem = 4
    p = 2
    knots = open_uniform_knots(n_elem, p)
    n = n_basis_open_uniform(n_elem, p)
    C = np.zeros((n, n))
    grad = lambda x, y: (jnp.ones_like(x), 2.0 * jnp.ones_like(y))
    val_fe = h1_seminorm_sq_2d(C, knots, knots, p, grad, quad_points_per_dim=4)
    val_ref = h1_seminorm_sq_analytic(grad, quad_points_per_dim=4, n_subdiv_per_dim=4)
    assert val_fe == pytest.approx(val_ref, rel=1e-12)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
