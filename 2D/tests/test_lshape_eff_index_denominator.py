"""Regression test for audit defect D1 (remediation T1).

The L-shape effectivity index MUST be

    I_eff = eta / |u_h - u_ref|_{sigma-H1}        (estimator over ERROR)

and NOT

    I_eff = eta / |u_ref|_{sigma-H1}              (estimator over reference NORM),

which was the shipped bug (Table 4 column 0.86..0.00).  ``_eff_index`` divides
by ``sqrt`` of its argument, so the correct denominator to pass is the SQUARED
absolute error ``h1_abs**2`` (returned as the 2nd element of
``_direct_h1_error_lshape``), whereas the bug passed ``ref_norm_sq`` (the 3rd
element).

This test exercises the two load-bearing functions (`_direct_h1_error_lshape`,
`_eff_index`) directly with a synthetic reference, and asserts:
  (a) the corrected identity  eff_index == eta / abs_error;
  (b) the error and the reference norm genuinely differ, so the bug would have
      produced a numerically different (wrong) value.
It needs no 60 MB reference pickle and runs in well under a second.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import jax.numpy as jnp

from src.nonparametric.r_adapt import p3_effective_n_elem, theta_to_knots_lshape
from src.nonparametric.solver_2d import galerkin_solve_lshape
from src.nonparametric.eta_estimator_2d import eta_squared_lshape
from src.parametric.evaluation_v2 import _direct_h1_error_lshape, _eff_index


def _uniform_solution(p: int, N: int, s1: float, s2: float):
    """Uniform-mesh immersed L-shape solve; returns (u_coeffs, knots, n_eff)."""
    half = int(N) // 2
    z = jnp.zeros((half,), dtype=jnp.float64)
    knots = theta_to_knots_lshape(z, z, p)
    n_eff = p3_effective_n_elem(int(N), p)
    res = galerkin_solve_lshape(knots, knots, p, n_eff, n_eff, s1, s2, q_K=p + 1, q_F=2)
    return np.asarray(res.u_h), np.asarray(knots), n_eff


def test_lshape_eff_index_uses_error_not_reference_norm():
    p, s1, s2 = 2, 1.0, 1.0

    # Reference = a finer uniform solve (N=16); candidate = a coarser one (N=8).
    # They differ, so the sigma-weighted seminorm error is strictly positive and
    # distinct from the reference norm.
    u_ref, kx_ref, _ = _uniform_solution(p, 16, s1, s2)
    ref_entry = {"knots_x": kx_ref, "knots_y": kx_ref, "u_coeffs": u_ref, "p": p}

    u_cand, kx_cand, n_eff = _uniform_solution(p, 8, s1, s2)

    rel, h1_abs, ref_norm_sq, cand_norm_sq = _direct_h1_error_lshape(
        u_cand, kx_cand, kx_cand, p, s1, s2, ref_entry
    )
    eta = math.sqrt(float(eta_squared_lshape(
        jnp.asarray(u_cand), jnp.asarray(kx_cand), jnp.asarray(kx_cand),
        p, n_eff, n_eff, jnp.asarray(s1), jnp.asarray(s2), q_est=4,
    )))

    assert h1_abs > 0.0 and math.isfinite(h1_abs)
    assert ref_norm_sq > 0.0 and math.isfinite(ref_norm_sq)
    # Error and reference norm must genuinely differ, else the test is vacuous.
    assert not math.isclose(h1_abs ** 2, ref_norm_sq, rel_tol=0.2)

    # (a) The corrected effectivity index divides eta by the ABSOLUTE ERROR.
    eff_corrected = _eff_index(eta, h1_abs ** 2)          # SQUARED error argument
    assert eff_corrected == pytest.approx(eta / h1_abs, rel=1e-12)

    # (b) The old (buggy) denominator (reference norm squared) gives a different,
    #     wrong value.  Guard against silent regression to it.
    eff_bug = _eff_index(eta, ref_norm_sq)
    assert not math.isclose(eff_corrected, eff_bug, rel_tol=1e-3)
    # rel is defined as sqrt(err^2 / ref_norm_sq), so eff_bug == eff_corrected * rel.
    assert eff_bug == pytest.approx(eff_corrected * rel, rel=1e-9)


def test_eff_index_contract_is_eta_over_sqrt_arg():
    """`_eff_index` must implement eta / sqrt(arg); NaN guard on non-positive."""
    assert _eff_index(2.0, 4.0) == pytest.approx(1.0)
    assert math.isnan(_eff_index(2.0, 0.0))
    assert math.isnan(_eff_index(2.0, float("nan")))
    assert math.isnan(_eff_index(float("nan"), 4.0))
