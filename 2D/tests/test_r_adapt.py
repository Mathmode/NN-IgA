"""Tests for src.nonparametric.r_adapt."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from src.nonparametric.r_adapt import (
    radapt_local,
    radapt_p3_one_sample,
    softmax_jax,
    theta_to_knots_arctan,
    theta_to_knots_lshape,
)


def test_softmax_jax_unit():
    out = softmax_jax(jnp.array([0.0, 0.0, 0.0]))
    assert float(jnp.sum(out)) == pytest.approx(1.0, abs=1e-14)
    assert all(float(v) == pytest.approx(1 / 3, abs=1e-14) for v in out)


def test_theta_to_knots_p2_endpoints_and_multiplicity():
    theta = jnp.zeros((4,))  # uniform
    p = 2
    knots = np.asarray(theta_to_knots_arctan(theta, p))
    # Multiplicity p+1 at both endpoints
    assert np.allclose(knots[:3], 0.0)
    assert np.allclose(knots[-3:], 1.0)
    # Interior knots equispaced: 0.25, 0.5, 0.75
    assert np.allclose(knots[3:6], [0.25, 0.5, 0.75])


def test_theta_to_knots_p3_fixed_node_at_half():
    """Uniform logits on both halves -> knot vector contains 0.5 with
    multiplicity p (C^0 basis at the σ-discontinuity per
    audit_p3_multipatch.md, enabled by the gradient-safe ``safe_div``
    fix in commit ``safe_div_robust_multiplicity``)."""
    theta_l = jnp.zeros((4,))
    theta_r = jnp.zeros((4,))
    p = 2
    knots = np.asarray(theta_to_knots_lshape(theta_l, theta_r, p))
    # Multiplicity p+1 at endpoints.
    assert np.allclose(knots[:3], 0.0)
    assert np.allclose(knots[-3:], 1.0)
    # Interior: (4-1) + p + (4-1) = 3 + p + 3.
    interior = knots[3:-3]
    assert interior.shape == (3 + p + 3,)
    # Multiplicity p at 0.5 sits in the middle.
    mid_lo = (interior.shape[0] - p) // 2
    assert np.allclose(interior[mid_lo:mid_lo + p], 0.5)
    # Half intervals retain the equispaced positions on each side of 0.5.
    assert np.isclose(interior[2], 0.375)
    assert np.isclose(interior[mid_lo + p], 0.625)


def test_radapt_local_minimises_quadratic():
    """Smoke: minimize f(x) = (x - 3)^2 starting from 0."""
    def obj(x):
        return float((x[0] - 3.0) ** 2), np.array([2.0 * (x[0] - 3.0)])
    res = radapt_local(obj, np.array([0.0]), max_iter=200)
    assert res.converged
    assert res.theta_final[0] == pytest.approx(3.0, abs=1e-6)
    assert res.energy_final == pytest.approx(0.0, abs=1e-12)


@pytest.mark.slow
def test_radapt_p3_decreases_ritz_energy():
    """Run a few L-BFGS iterations and verify the energy decreases relative
    to the uniform starting point."""
    from src.nonparametric.r_adapt import p3_effective_n_elem
    n_elem = 8
    p = 2
    sigma1, sigma2 = 0.1, 10.0
    n_eff = p3_effective_n_elem(n_elem, p)
    # Baseline energy at uniform start.
    from src.nonparametric.solver_2d import galerkin_solve_lshape
    knots_u = theta_to_knots_lshape(jnp.zeros((4,)), jnp.zeros((4,)), p)
    j_u = float(
        galerkin_solve_lshape(
            knots_u, knots_u, p, n_eff, n_eff,
            jnp.asarray(sigma1), jnp.asarray(sigma2), q_K=p + 1, q_F=2,
        ).ritz_energy
    )

    result, _ = radapt_p3_one_sample(
        sigma1, sigma2, n_elem=n_elem, p=p, max_iter=30
    )
    print(
        f"\nP3 radapt sigma=({sigma1}, {sigma2}): "
        f"uniform J = {j_u:.4e}, optimized J = {result.energy_final:.4e}"
    )
    # Optimization should reduce (or hold) the energy.
    assert result.energy_final <= j_u + 1e-10


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v", "-s"])
