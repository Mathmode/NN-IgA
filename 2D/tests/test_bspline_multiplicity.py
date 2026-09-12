"""Tests for B-spline basis with arbitrary knot multiplicity.

The ``common.bspline_basis.safe_div`` helper was refactored to use the
double-where pattern so that ``jax.grad`` produces finite gradients on
the unselected branch even when the denominator is structurally zero
(repeated knots). This unlocks multiplicity-p knots at the σ-discontinuity
in lshape (giving a C^0 basis there, matching the physics).

Tests verify:
  - basis values + gradients are finite for multiplicity-p knots,
  - C^0 continuity is enforced (values match left/right of the interface),
  - ∂N has a real jump at the multiplicity-p knot (basis is NOT C^1),
  - multiplicity-1 (production singular / arctan path) is unchanged (regression).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from common.bspline_basis import bspline_basis_local, safe_div


# --------------------------------------------------------------------------
# Direct safe_div regression
# --------------------------------------------------------------------------


def test_safe_div_value_matches_normal_division_for_nonzero_den():
    """For den != 0 the new safe_div returns num / den exactly."""
    for num, den in [(1.0, 2.0), (5.0, 0.1), (-3.7, 1.2), (1e-15, 1e-15)]:
        ours = float(safe_div(jnp.asarray(num), jnp.asarray(den)))
        ref = float(num) / float(den)
        assert ours == pytest.approx(ref, rel=1e-15, abs=1e-300)


def test_safe_div_zero_den_returns_zero_value_and_gradient():
    """For den == 0, safe_div returns 0 and zero gradient (no NaN)."""
    val = float(safe_div(jnp.asarray(1.0), jnp.asarray(0.0)))
    assert val == 0.0
    g = jax.grad(lambda x: safe_div(x, jnp.asarray(0.0)))(jnp.asarray(1.0))
    assert float(g) == 0.0
    g_den = jax.grad(lambda d: safe_div(jnp.asarray(1.0), d))(jnp.asarray(0.0))
    assert np.isfinite(float(g_den))


# --------------------------------------------------------------------------
# B-spline basis with multiplicity-p interior knot
# --------------------------------------------------------------------------


def _knots_mult_p(p: int) -> jnp.ndarray:
    """Open-uniform on [0, 1] with multiplicity p at the interior knot 0.5
    plus single knots at 0.25 and 0.75 (just for diversity)."""
    return jnp.concatenate([
        jnp.zeros(p + 1),
        jnp.array([0.25]),
        jnp.full(p, 0.5),
        jnp.array([0.75]),
        jnp.ones(p + 1),
    ])


@pytest.mark.parametrize("p", [2, 3])
def test_evaluate_basis_mult_p_finite(p):
    knots = _knots_mult_p(p)
    x = jnp.linspace(0.01, 0.99, 50)
    N, dN, d2N, _ = bspline_basis_local(x, knots, p)
    assert jnp.all(jnp.isfinite(N)), "N has NaN/Inf"
    assert jnp.all(jnp.isfinite(dN)), "dN has NaN/Inf"
    assert jnp.all(jnp.isfinite(d2N)), "d2N has NaN/Inf"


@pytest.mark.parametrize("p", [2, 3])
def test_partition_of_unity_with_mult_p(p):
    knots = _knots_mult_p(p)
    x = jnp.linspace(0.01, 0.99, 50)
    N, _, _, _ = bspline_basis_local(x, knots, p)
    pou_err = float(jnp.max(jnp.abs(N.sum(axis=-1) - 1.0)))
    # B-spline POU is exact in theory; with the double-where ε we get
    # tiny single-precision rounding (~1e-7).
    assert pou_err < 1e-6, f"POU err = {pou_err:.3e}"


@pytest.mark.parametrize("p", [2, 3])
def test_grad_basis_wrt_knots_finite_with_mult_p(p):
    knots = _knots_mult_p(p)
    x = jnp.array([0.30, 0.49, 0.51, 0.70])

    def loss(k):
        Nk, _, _, _ = bspline_basis_local(x, k, p)
        return Nk.sum()

    g = jax.grad(loss)(knots)
    assert jnp.all(jnp.isfinite(g)), f"NaN/Inf gradient at p={p}: {g}"


@pytest.mark.parametrize("p", [2, 3])
def test_grad_dbasis_wrt_knots_finite_with_mult_p(p):
    knots = _knots_mult_p(p)
    x = jnp.array([0.30, 0.49, 0.51, 0.70])

    def loss(k):
        _, dNk, _, _ = bspline_basis_local(x, k, p)
        return dNk.sum()

    g = jax.grad(loss)(knots)
    assert jnp.all(jnp.isfinite(g)), f"NaN/Inf dN-gradient at p={p}: {g}"


# --------------------------------------------------------------------------
# C^0 vs C^1 character at the multiplicity-p knot
# --------------------------------------------------------------------------


def _global_eval(coeffs, x, knots, p):
    """Evaluate u(x) = Σ_i coeffs[i] * B_i(x) for an array of x values.

    ``bspline_basis_local`` returns the (p+1) active basis functions and
    their span; we scatter the local coefficients globally to assemble u(x)
    and du/dx.
    """
    N, dN, _, spans = bspline_basis_local(x, knots, p)
    # spans[k] is the global span; active functions are indices
    # [span - p, span] = span - p, span - p + 1, ..., span.
    def local_idx(s):
        return s - p + jnp.arange(p + 1, dtype=jnp.int32)
    idx = jax.vmap(local_idx)(spans)         # (n_x, p+1)
    coeffs_local = coeffs[idx]                # (n_x, p+1)
    u = jnp.sum(N * coeffs_local, axis=-1)
    du = jnp.sum(dN * coeffs_local, axis=-1)
    return u, du


@pytest.mark.parametrize("p", [2, 3])
def test_global_function_is_C0_at_mult_p_knot(p):
    """Global function ``u(x) = Σ c_i B_i(x)`` is continuous at the
    multiplicity-p knot (C^0) for arbitrary coefficients."""
    knots = _knots_mult_p(p)
    n_basis = int(knots.shape[0] - p - 1)
    rng = np.random.default_rng(7)
    coeffs = jnp.asarray(rng.standard_normal(n_basis))
    eps = 1e-7
    x_l = jnp.array([0.5 - eps])
    x_r = jnp.array([0.5 + eps])
    u_l, _ = _global_eval(coeffs, x_l, knots, p)
    u_r, _ = _global_eval(coeffs, x_r, knots, p)
    assert float(jnp.abs(u_l[0] - u_r[0])) < 1e-5, (
        f"u not C^0 at 0.5 (p={p}): u(0.5-)={float(u_l[0])}, u(0.5+)={float(u_r[0])}"
    )


@pytest.mark.parametrize("p", [2, 3])
def test_global_derivative_jumps_at_mult_p_knot(p):
    """Global derivative ``du/dx`` should have a real jump at the
    multiplicity-p knot (basis is C^0, NOT C^1)."""
    knots = _knots_mult_p(p)
    n_basis = int(knots.shape[0] - p - 1)
    rng = np.random.default_rng(11)
    coeffs = jnp.asarray(rng.standard_normal(n_basis))
    eps = 1e-4
    x_l = jnp.array([0.5 - eps])
    x_r = jnp.array([0.5 + eps])
    _, du_l = _global_eval(coeffs, x_l, knots, p)
    _, du_r = _global_eval(coeffs, x_r, knots, p)
    jump = float(jnp.abs(du_l[0] - du_r[0]))
    assert jump > 1e-2, (
        f"expected real du jump at mult-p knot (p={p}); jump={jump:.3e}"
    )


# --------------------------------------------------------------------------
# Regression: multiplicity-1 path unchanged
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p", [2, 3])
def test_no_regression_mult_1(p):
    """With mult-1 knots (1D / 2D production path) the new safe_div is a
    no-op vs the old single-where (since |den| > eps everywhere)."""
    knots = jnp.concatenate([
        jnp.zeros(p + 1),
        jnp.array([0.25, 0.5, 0.75]),
        jnp.ones(p + 1),
    ])
    x = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    N, dN, d2N, _ = bspline_basis_local(x, knots, p)
    assert jnp.all(jnp.isfinite(N))
    assert jnp.allclose(N.sum(axis=-1), 1.0, atol=1e-12)

    g = jax.grad(lambda k: bspline_basis_local(x, k, p)[0].sum())(knots)
    assert jnp.all(jnp.isfinite(g))


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
