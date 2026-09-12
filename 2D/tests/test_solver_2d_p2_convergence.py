"""arctan solver: convergence rate on a smooth manufactured solution."""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from src.nonparametric.arctan.pde import grad_u_sigma
from src.nonparametric.solver_2d import galerkin_solve_arctan
from common.h1_seminorm_2d import (
    h1_seminorm_sq_2d,
    h1_seminorm_sq_analytic,
    open_uniform_knots,
)


def _solve_and_h1_rel(alpha, s1, s2, *, p, n_elem):
    knots = open_uniform_knots(n_elem, p)
    res = galerkin_solve_arctan(
        jnp.asarray(knots),
        jnp.asarray(knots),
        p,
        n_elem,
        n_elem,
        jnp.asarray(alpha),
        jnp.asarray(s1),
        jnp.asarray(s2),
        q_K=p + 1,
        q_F=40,
    )

    grad_fn = lambda x, y: grad_u_sigma(x, y, alpha, s1, s2)
    num = h1_seminorm_sq_2d(
        np.asarray(res.u_h), knots, knots, p, grad_fn, quad_points_per_dim=40
    )
    den = h1_seminorm_sq_analytic(
        grad_fn, quad_points_per_dim=40, n_subdiv_per_dim=8
    )
    h1_rel = math.sqrt(num / den)
    return float(h1_rel)


@pytest.mark.parametrize("p", [2, 3])
def test_uniform_h_convergence_smooth_arctangent(p):
    """For alpha=5 (moderate), uniform mesh refinement reduces H1_rel
    monotonically. The first refinement (N=4 -> 8) may be pre-asymptotic at p=2;
    we require strict descent there and a clean h^p rate at N=8 -> 16."""
    alpha, s1, s2 = 5.0, 0.5, 0.5
    e4 = _solve_and_h1_rel(alpha, s1, s2, p=p, n_elem=4)
    e8 = _solve_and_h1_rel(alpha, s1, s2, p=p, n_elem=8)
    e16 = _solve_and_h1_rel(alpha, s1, s2, p=p, n_elem=16)
    print(f"\np={p}: H1_rel  N=4: {e4:.3e}  N=8: {e8:.3e}  N=16: {e16:.3e}")
    # Pre-asymptotic: just monotone descent.
    assert e8 < e4, f"4 -> 8 not monotone: {e4}, {e8}"
    # Asymptotic h^p rate: expect at least h^p / 2 reduction (2x at p=2, 4x at p=3).
    expected_factor = 2.0 ** p / 2.0
    assert e16 < e8 / expected_factor, (
        f"8 -> 16 below expected (h^{p}/2 = {expected_factor}x reduction): {e8}, {e16}"
    )


def test_p2_smoke_alpha5_n8_in_expected_range():
    """At alpha=5, s1=s2=0.5, p=3, N=8 uniform: H1_rel should be < 0.5 (loose).
    Aballay 4.3.1 at N=64 reports 0.046 uniform; we're far coarser here."""
    e = _solve_and_h1_rel(5.0, 0.5, 0.5, p=3, n_elem=8)
    print(f"\nP2 smoke alpha=5 p=3 N=8: H1_rel = {e:.3e}")
    assert 0.0 < e < 0.5


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v", "-s"])
