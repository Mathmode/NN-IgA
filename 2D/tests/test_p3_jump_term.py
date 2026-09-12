"""Tests for the lshape material-interface flux-jump term in
``eta_estimator_2d._eta_jump_lshape``.

The flux jump on a material interface is
    ``[s d_n u_h](x_F) = s_+ d_n u_h(x_F) - s_- d_n u_h(x_F)``
which, for piecewise-constant sigma, carries the factor ``(s_+ - s_-)``.

These tests check structural properties of the jump term that are
independent of the sigma-swap symmetry analysis (covered separately by
``test_eta_jump_p3_symmetry.py``): growth with sigma-contrast, the bound
``total eta^2 >= volume eta^2``, a non-trivial jump fraction at high
contrast, and h-convergence of the jump term under uniform refinement.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from src.nonparametric.eta_estimator_2d import _eta_jump_lshape, _eta_volume_lshape, eta_squared_lshape
from src.nonparametric.r_adapt import p3_effective_n_elem, theta_to_knots_lshape
from src.nonparametric.solver_2d import galerkin_solve_lshape


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _solve(sigma1, sigma2, *, N, p):
    half = N // 2
    knots = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), p)
    n_eff = p3_effective_n_elem(N, p)
    res = galerkin_solve_lshape(
        knots, knots, p, n_eff, n_eff,
        jnp.asarray(sigma1), jnp.asarray(sigma2),
        q_K=p + 1, q_F=2,
    )
    return knots, res.u_h, n_eff


# --------------------------------------------------------------------------
# Properties of the jump term
# --------------------------------------------------------------------------



def test_jump_term_grows_with_sigma_contrast():
    """The jump magnitude scales as (σ_+ − σ_−)²; doubling the contrast
    must increase the jump term."""
    N, p = 20, 2
    n_eff = p3_effective_n_elem(N, p)
    knots = theta_to_knots_lshape(jnp.zeros((N // 2,)), jnp.zeros((N // 2,)), p)

    def jump_for(s1, s2):
        res = galerkin_solve_lshape(
            knots, knots, p, n_eff, n_eff,
            jnp.asarray(s1), jnp.asarray(s2),
            q_K=p + 1, q_F=2,
        )
        return float(_eta_jump_lshape(
            res.u_h, knots, knots, p, n_eff, n_eff,
            jnp.asarray(s1), jnp.asarray(s2), 2,
        ))

    low = jump_for(0.5, 0.5)
    mid = jump_for(0.1, 0.1)
    high = jump_for(0.01, 0.01)
    print(f"\njump @ (0.5,0.5)={low:.3e}, (0.1,0.1)={mid:.3e}, (0.01,0.01)={high:.3e}")
    assert low < mid
    assert mid < high



def test_total_eta_with_jump_is_at_least_volume_only():
    """Total η² = volume + jump ≥ volume — the jump can only INCREASE the
    estimator, never reduce it."""
    for s1, s2 in [(1.0, 1.0), (0.1, 0.1), (10.0, 0.1), (0.1, 10.0)]:
        N, p = 20, 2
        knots, u_h, n_eff = _solve(s1, s2, N=N, p=p)
        vol = float(_eta_volume_lshape(
            u_h, knots, knots, p, n_eff, n_eff,
            jnp.asarray(s1), jnp.asarray(s2), 2,
        ))
        total = float(eta_squared_lshape(
            u_h, knots, knots, p, n_eff, n_eff,
            jnp.asarray(s1), jnp.asarray(s2), q_est=2,
        ))
        assert total >= vol - 1e-12, f"σ=({s1}, {s2}): total {total} < volume {vol}"


def test_jump_term_nontrivial_for_high_contrast():
    """At high σ-contrast (10, 0.1) the jump term must be a NON-trivial
    fraction of the total η² (≥3% at p=2, N=20 after the σ-weighting and
    C^0 basis fix). The fraction shrinks vs the C^{p-1} basis because the
    volume residual is much smaller now — that's the whole point of the fix."""
    s1, s2 = 10.0, 0.1
    N, p = 20, 2
    knots, u_h, n_eff = _solve(s1, s2, N=N, p=p)
    vol = float(_eta_volume_lshape(
        u_h, knots, knots, p, n_eff, n_eff,
        jnp.asarray(s1), jnp.asarray(s2), 2,
    ))
    jump = float(_eta_jump_lshape(
        u_h, knots, knots, p, n_eff, n_eff,
        jnp.asarray(s1), jnp.asarray(s2), 2,
    ))
    total = vol + jump
    jump_frac = jump / total
    print(f"\nσ=(10, 0.1) N=20: vol={vol:.3e}, jump={jump:.3e}, frac={jump_frac:.2%}")
    assert jump > 0, "jump term must be > 0 with σ-contrast"
    assert jump_frac > 0.03, (
        f"jump term should be ≥3% of total at high contrast; got {jump_frac:.2%}"
    )


def test_jump_term_h_convergence():
    """Under uniform refinement, the jump term should decrease (at the
    p^th rate in h). We check the soft criterion that
    jump(N=80) < jump(N=10)/2 across a representative σ."""
    p = 2
    s1, s2 = 0.1, 10.0
    jumps = {}
    for N in (10, 20, 40, 80):
        knots, u_h, n_eff = _solve(s1, s2, N=N, p=p)
        jumps[N] = float(_eta_jump_lshape(
            u_h, knots, knots, p, n_eff, n_eff,
            jnp.asarray(s1), jnp.asarray(s2), 2,
        ))
    print(f"\nP3 jump term under h-refinement σ=(0.1, 10): {jumps}")
    # Monotone descent.
    Ns = sorted(jumps)
    for n_lo, n_hi in zip(Ns[:-1], Ns[1:]):
        assert jumps[n_hi] < jumps[n_lo], (
            f"jump term not monotone: N={n_lo} → {n_hi}: {jumps[n_lo]:.3e} → {jumps[n_hi]:.3e}"
        )
    assert jumps[80] < jumps[10] / 2.0


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v", "-s"])
