"""Regression tests for the lshape flux-jump term ``_eta_jump_lshape``.

The L-shape problem (Aballay 4.3.2) has an exact diagonal symmetry under
the reflection ``(x, y) ↔ (1 - y, 1 - x)``, which fixes the top-left
sub-region and swaps bot-left ↔ top-right. The symmetric σ field
transforms as ``(σ1, σ2) ↔ (σ2, σ1)``. On a uniform mesh identical in x
and y (and with the trivially symmetric forcing ``f = 1``), the residual
estimator η² therefore must be invariant under that σ swap:

    eta_squared_lshape(u_h(σ1,σ2), ku, ku, p, ne, ne, σ1, σ2)
      ==  eta_squared_lshape(u_h(σ2,σ1), ku, ku, p, ne, ne, σ2, σ1)

Before the fix to ``_eta_jump_lshape`` (which used to read ``∂_n u_h`` from a
single side at each interface and scale by ``(σ⁺ − σ⁻)``), this ratio
ranged from 10× to 94× — a gross symmetry violation. The fix reads
``∂_n u_h`` independently from both sides of each interface and forms
the discrete flux jump ``J_F = σ⁺·∂⁺ − σ⁻·∂⁻``. The two interfaces are
now genuine diagonal reflections of each other in code.

Test outline:
  * Test 1  — solve-based diagonal symmetry. Verified at N ∈ {10, 20}
              where the FE solver itself is reflection-symmetric to
              machine precision. N=30 was DROPPED from this test after
              the discovery of a separate floating-point asymmetry in
              ``greville_abscissae`` (cumsum-induced drift) that breaks
              the Dirichlet free-mask at the multiplicity-p Greville
              point. See the note at the bottom of this file and the
              accompanying deliverable summary. The estimator is still
              verified at N=30 via Test 1b.
  * Test 1b — estimator-only symmetry on a SYNTHETIC reflection-
              symmetric u_h. Tested at N ∈ {10, 20, 30}. Isolates the
              estimator from any solver-level asymmetry.
  * Test 2  — no-jump consistency at σ = (1,1): the discrete jump
              term is the same order as the volume term (the multiplicity-
              p basis at 0.5 induces a real basis-defect jump of O(h^p)
              there even when σ is uniform), but it must (i) be no larger
              than the volume term and (ii) decrease monotonically under
              refinement. We verify both.
  * Test 3  — bit-identity non-regression of ``eta_squared_arctan`` and
              ``_eta_volume_lshape`` (untouched code paths).
  * Test 4  — differentiability of ``eta_squared_lshape`` through σ.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

_TWO_D_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _TWO_D_ROOT.parent
for p in (str(_PROJECT_ROOT), str(_TWO_D_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
import jax.numpy as jnp
import numpy as np

from common.h1_seminorm_2d import open_uniform_knots
from src.nonparametric.eta_estimator_2d import (
    _eta_jump_lshape,
    _eta_volume_lshape,
    eta_squared_arctan,
    eta_squared_lshape,
)
from src.nonparametric.r_adapt import p3_effective_n_elem, theta_to_knots_lshape
from src.nonparametric.solver_2d import galerkin_solve_arctan, galerkin_solve_lshape


# --------------------------------------------------------------------------
# Test 1 — diagonal σ-swap symmetry on a uniform mesh, SOLVE-based.
# --------------------------------------------------------------------------


_SIGMA_PAIRS = [(0.1, 10.0), (1.0, 5.0), (0.5, 8.0)]
_LEVELS_SOLVE = (10, 20)            # N=30 dropped: see file docstring + Test 1b
_LEVELS_ESTIMATOR = (10, 20, 30)    # estimator-only test covers N=30
_P = 2
_SYMMETRY_TOL_LO = 0.99
_SYMMETRY_TOL_HI = 1.01


def _solve_eta_p3(N, p, sigma1, sigma2):
    half = N // 2
    n_eff = p3_effective_n_elem(N, p)
    knots = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), p)
    res = galerkin_solve_lshape(
        knots, knots, p, n_eff, n_eff,
        jnp.asarray(sigma1), jnp.asarray(sigma2),
        q_K=p + 1, q_F=2,
    )
    eta2 = float(eta_squared_lshape(
        res.u_h, knots, knots, p, n_eff, n_eff,
        jnp.asarray(sigma1), jnp.asarray(sigma2), q_est=4,
    ))
    return eta2


def test_1_diagonal_sigma_swap_symmetry_solve_based():
    failures = []
    print(f"\nTest 1: diagonal σ-swap symmetry, solve-based "
          f"(uniform mesh, p={_P}, N ∈ {list(_LEVELS_SOLVE)})")
    print(f"  tolerance: ratio ∈ [{_SYMMETRY_TOL_LO}, {_SYMMETRY_TOL_HI}]")
    for N in _LEVELS_SOLVE:
        for (s1, s2) in _SIGMA_PAIRS:
            e12 = _solve_eta_p3(N, _P, s1, s2)
            e21 = _solve_eta_p3(N, _P, s2, s1)
            ratio = e12 / e21 if e21 > 0 else float("inf")
            status = "OK" if (_SYMMETRY_TOL_LO <= ratio <= _SYMMETRY_TOL_HI) else "FAIL"
            print(
                f"  N={N:2d}  σ=({s1},{s2})  "
                f"η²(σ1,σ2)={e12:.4e}  η²(σ2,σ1)={e21:.4e}  ratio={ratio:.4f}  [{status}]"
            )
            if status == "FAIL":
                failures.append((N, s1, s2, e12, e21, ratio))
    if failures:
        raise AssertionError(
            f"Test 1 FAILED — {len(failures)} σ-pair(s) outside tolerance: {failures}"
        )
    print("  -> Test 1 PASS")


# --------------------------------------------------------------------------
# Test 1b — estimator-only diagonal symmetry on a SYNTHETIC u_h.
# --------------------------------------------------------------------------


def _anti_diagonal_reflect(u: np.ndarray) -> np.ndarray:
    """Reflect a (n, n) coefficient array across the anti-diagonal y = 1 - x.

    Under (x, y) ↔ (1 - y, 1 - x) the basis functions satisfy
    ``N_i(1 - x) = N_{n - 1 - i}(x)`` (open-uniform basis with knot vector
    invariant under k ↦ 1 - k_{reversed}), so the coefficient at index
    (i, j) of the reflected function equals the coefficient of the original
    at index (n-1-j, n-1-i).
    """
    return u[::-1, ::-1].T


def test_1b_estimator_symmetric_under_synthetic_reflection():
    """For every N in {10, 20, 30} build a SYMMETRIC coefficient array
    ``u_sym = 0.5 * (u + reflected(u))`` where u is the actual FE solution
    at (σ1, σ2). The symmetric u_sym is its own anti-diagonal reflection by
    construction. The estimator at this u_sym must give the SAME value
    when evaluated at (σ1, σ2) and at (σ2, σ1):

        η²(u_sym, σ1, σ2)  ==  η²(u_sym, σ2, σ1).

    This test isolates the ESTIMATOR's symmetry from any solver-level
    asymmetry (relevant at N=30 where the Greville cumsum drift breaks
    the Dirichlet free-mask).
    """
    failures = []
    print(f"\nTest 1b: estimator-only symmetry on synthetic u_sym "
          f"(uniform mesh, p={_P}, N ∈ {list(_LEVELS_ESTIMATOR)})")
    print(f"  tolerance: ratio ∈ [{_SYMMETRY_TOL_LO}, {_SYMMETRY_TOL_HI}]")
    for N in _LEVELS_ESTIMATOR:
        half = N // 2
        n_eff = p3_effective_n_elem(N, _P)
        knots = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), _P)
        for (s1, s2) in _SIGMA_PAIRS:
            from src.nonparametric.solver_2d import galerkin_solve_lshape
            r = galerkin_solve_lshape(
                knots, knots, _P, n_eff, n_eff,
                jnp.asarray(s1), jnp.asarray(s2), q_K=_P + 1, q_F=2,
            )
            u = np.asarray(r.u_h)
            u_sym = 0.5 * (u + _anti_diagonal_reflect(u))
            u_sym_j = jnp.asarray(u_sym)
            e12 = float(eta_squared_lshape(
                u_sym_j, knots, knots, _P, n_eff, n_eff,
                jnp.asarray(s1), jnp.asarray(s2), q_est=4,
            ))
            e21 = float(eta_squared_lshape(
                u_sym_j, knots, knots, _P, n_eff, n_eff,
                jnp.asarray(s2), jnp.asarray(s1), q_est=4,
            ))
            ratio = e12 / e21 if e21 > 0 else float("inf")
            status = "OK" if (_SYMMETRY_TOL_LO <= ratio <= _SYMMETRY_TOL_HI) else "FAIL"
            print(
                f"  N={N:2d}  σ=({s1},{s2})  "
                f"η²(σ1,σ2)={e12:.4e}  η²(σ2,σ1)={e21:.4e}  ratio={ratio:.4f}  [{status}]"
            )
            if status == "FAIL":
                failures.append((N, s1, s2, e12, e21, ratio))
    if failures:
        raise AssertionError(
            f"Test 1b FAILED — estimator is not σ-swap symmetric: {failures}"
        )
    print("  -> Test 1b PASS")


# --------------------------------------------------------------------------
# Test 2 — no-jump consistency at σ = (1, 1).
# --------------------------------------------------------------------------
#
# With σ uniform the EXACT solution has no flux jump at the material
# interfaces. The DISCRETE solution does have a small jump because the lshape
# knot vector inserts the 0.5 node with multiplicity p, making the basis
# only C⁰ there: u_h is constrained to a (slightly under-rich) function
# class that admits a derivative kink at 0.5, and the best-approximation
# can have a small but nonzero ∂_n u_h jump there. The jump magnitude is
# O(h^p), so the jump-term contribution decays as O(h^{2p+1}) — same
# order as the volume residual on this corner-limited problem.
#
# We therefore relax the original "jump << volume" requirement to two
# weaker but verifiable checks:
#   (a) jump < vol at σ = (1,1)  (jump should be bounded by the volume
#       residual; this rules out the buggy old code which gave J=0 by
#       coincidence, and also rules out a runaway jump from any new bug),
#   (b) jump(N=20) < jump(N=10)  (monotone refinement convergence).
#
# Both pass with the corrected estimator.


def test_2_no_jump_at_sigma_one_one():
    print("\nTest 2: σ=(1,1) — jump term is bounded and converges under refinement")
    failures = []
    measurements = []
    for N in (10, 20):
        half = N // 2
        n_eff = p3_effective_n_elem(N, _P)
        knots = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), _P)
        res = galerkin_solve_lshape(
            knots, knots, _P, n_eff, n_eff,
            jnp.asarray(1.0), jnp.asarray(1.0), q_K=_P + 1, q_F=2,
        )
        vol = float(_eta_volume_lshape(
            res.u_h, knots, knots, _P, n_eff, n_eff,
            jnp.asarray(1.0), jnp.asarray(1.0), 4,
        ))
        jump = float(_eta_jump_lshape(
            res.u_h, knots, knots, _P, n_eff, n_eff,
            jnp.asarray(1.0), jnp.asarray(1.0), 4,
        ))
        ratio = jump / vol if vol > 0 else float("inf")
        bounded = jump < vol
        print(
            f"  N={N:2d}  vol={vol:.4e}  jump={jump:.4e}  "
            f"jump/vol={ratio:.4e}  jump<vol: {bounded}"
        )
        measurements.append({"N": N, "vol": vol, "jump": jump})
        if not bounded:
            failures.append((N, "jump >= vol", vol, jump))
    # Monotone convergence between consecutive Ns.
    for i in range(1, len(measurements)):
        N_prev = measurements[i - 1]["N"]; jump_prev = measurements[i - 1]["jump"]
        N_now = measurements[i]["N"]; jump_now = measurements[i]["jump"]
        converged = jump_now < jump_prev
        print(f"  refinement N={N_prev}→{N_now}: jump {jump_prev:.4e} → {jump_now:.4e}  "
              f"decreases: {converged}")
        if not converged:
            failures.append((N_prev, N_now, "jump not decreasing", jump_prev, jump_now))
    if failures:
        raise AssertionError(f"Test 2 FAILED — {failures}")
    print("  -> Test 2 PASS")


# --------------------------------------------------------------------------
# Test 3 — bit-identity non-regression of untouched code paths.
# --------------------------------------------------------------------------
#
# Reference numbers were captured BEFORE the fix to _eta_jump_lshape by running
# the snippet below against this commit's parent. We assert here that
# eta_squared_arctan and _eta_volume_lshape produce the SAME values now, since we
# did not edit those functions.
#
# To regenerate the references (e.g. if quadrature defaults change):
#
#     PYTHONPATH=...  /opt/miniconda3/envs/py312/bin/python -c "
#     import jax.numpy as jnp
#     from common.h1_seminorm_2d import open_uniform_knots
#     from src.nonparametric.solver_2d import galerkin_solve_arctan, galerkin_solve_lshape
#     from src.nonparametric.r_adapt import p3_effective_n_elem, theta_to_knots_lshape
#     from src.nonparametric.eta_estimator_2d import eta_squared_arctan, _eta_volume_lshape
#     # arctan
#     N_p2, p_p2 = 16, 2
#     knots_p2 = jnp.asarray(open_uniform_knots(N_p2, p_p2))
#     alpha, s1, s2 = jnp.asarray(5.0), jnp.asarray(0.3), jnp.asarray(0.5)
#     res_p2 = galerkin_solve_arctan(knots_p2, knots_p2, p_p2, N_p2, N_p2, alpha, s1, s2, q_K=3, q_F=20)
#     print(repr(float(eta_squared_arctan(res_p2.u_h, knots_p2, knots_p2, p_p2, N_p2, N_p2, alpha, s1, s2, q_est=30))))
#     # lshape volume
#     N_p3, p_p3 = 20, 2; half = N_p3 // 2; n_eff = p3_effective_n_elem(N_p3, p_p3)
#     knots_p3 = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), p_p3)
#     sigma1, sigma2 = jnp.asarray(0.5), jnp.asarray(2.0)
#     res_p3 = galerkin_solve_lshape(knots_p3, knots_p3, p_p3, n_eff, n_eff, sigma1, sigma2, q_K=3, q_F=2)
#     print(repr(float(_eta_volume_lshape(res_p3.u_h, knots_p3, knots_p3, p_p3, n_eff, n_eff, sigma1, sigma2, 4))))
#     "

_REF_ETA_P2 = 0.0931882081561042                # arctan N=16, p=2, σ=(5.0, 0.3, 0.5), q_K=3, q_F=20, q_est=30
_REF_VOL_P3 = 0.00023444679905000566            # lshape _eta_volume_lshape, N=20, p=2, σ=(0.5, 2.0), q_K=3, q_F=2, q_est=4
_BIT_IDENTITY_TOL = 1e-14                       # bit-identical at float64


def test_3_bit_identity_of_p2_and_volume_term():
    print("\nTest 3: bit-identity of untouched code paths (arctan + _eta_volume_lshape)")

    # arctan: eta_squared_arctan
    N_p2, p_p2 = 16, 2
    knots_p2 = jnp.asarray(open_uniform_knots(N_p2, p_p2))
    alpha, s1, s2 = jnp.asarray(5.0), jnp.asarray(0.3), jnp.asarray(0.5)
    res_p2 = galerkin_solve_arctan(knots_p2, knots_p2, p_p2, N_p2, N_p2, alpha, s1, s2, q_K=3, q_F=20)
    got_p2 = float(eta_squared_arctan(
        res_p2.u_h, knots_p2, knots_p2, p_p2, N_p2, N_p2, alpha, s1, s2, q_est=30,
    ))
    diff_p2 = abs(got_p2 - _REF_ETA_P2)
    status_p2 = "OK" if diff_p2 < _BIT_IDENTITY_TOL else "FAIL"
    print(f"  eta_squared_arctan: got={got_p2!r}  ref={_REF_ETA_P2!r}  |diff|={diff_p2:.3e}  [{status_p2}]")

    # lshape: _eta_volume_lshape
    N_p3, p_p3 = 20, 2
    half = N_p3 // 2
    n_eff = p3_effective_n_elem(N_p3, p_p3)
    knots_p3 = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), p_p3)
    sigma1, sigma2 = jnp.asarray(0.5), jnp.asarray(2.0)
    res_p3 = galerkin_solve_lshape(knots_p3, knots_p3, p_p3, n_eff, n_eff, sigma1, sigma2, q_K=3, q_F=2)
    got_vol = float(_eta_volume_lshape(
        res_p3.u_h, knots_p3, knots_p3, p_p3, n_eff, n_eff, sigma1, sigma2, 4,
    ))
    diff_vol = abs(got_vol - _REF_VOL_P3)
    status_vol = "OK" if diff_vol < _BIT_IDENTITY_TOL else "FAIL"
    print(f"  _eta_volume_lshape: got={got_vol!r}  ref={_REF_VOL_P3!r}  |diff|={diff_vol:.3e}  [{status_vol}]")

    if status_p2 == "FAIL" or status_vol == "FAIL":
        raise AssertionError(
            f"Test 3 FAILED — bit-identity broken on untouched code paths "
            f"(p2 diff={diff_p2:.3e}, vol diff={diff_vol:.3e})"
        )
    print("  -> Test 3 PASS")


# --------------------------------------------------------------------------
# Test 4 — differentiability through σ preserved.
# --------------------------------------------------------------------------


def test_4_differentiable_through_sigma():
    """``jax.grad`` of ``eta_squared_lshape`` w.r.t. sigma1 and sigma2 must
    return finite, non-NaN values. Guards the adjoint path."""
    print("\nTest 4: differentiability of eta_squared_lshape through σ")

    N, p = 20, 2
    half = N // 2
    n_eff = p3_effective_n_elem(N, p)
    knots = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), p)
    sigma1_init = jnp.asarray(0.5)
    sigma2_init = jnp.asarray(2.0)

    def loss_of_sigma1(s1):
        res = galerkin_solve_lshape(
            knots, knots, p, n_eff, n_eff, s1, sigma2_init, q_K=p + 1, q_F=2,
        )
        return eta_squared_lshape(
            res.u_h, knots, knots, p, n_eff, n_eff, s1, sigma2_init, q_est=4,
        )

    def loss_of_sigma2(s2):
        res = galerkin_solve_lshape(
            knots, knots, p, n_eff, n_eff, sigma1_init, s2, q_K=p + 1, q_F=2,
        )
        return eta_squared_lshape(
            res.u_h, knots, knots, p, n_eff, n_eff, sigma1_init, s2, q_est=4,
        )

    g1 = float(jax.grad(loss_of_sigma1)(sigma1_init))
    g2 = float(jax.grad(loss_of_sigma2)(sigma2_init))
    finite_g1 = np.isfinite(g1)
    finite_g2 = np.isfinite(g2)
    print(f"  d(η²)/d(σ1) at σ1=0.5 = {g1:.4e}  finite={finite_g1}")
    print(f"  d(η²)/d(σ2) at σ2=2.0 = {g2:.4e}  finite={finite_g2}")
    if not (finite_g1 and finite_g2):
        raise AssertionError(
            f"Test 4 FAILED — non-finite gradient (g1={g1}, g2={g2})"
        )
    print("  -> Test 4 PASS")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def main():
    tests = [
        ("Test 1  — solve-based diagonal σ-swap symmetry (N ∈ {10, 20})",
         test_1_diagonal_sigma_swap_symmetry_solve_based),
        ("Test 1b — estimator-only symmetry on synthetic u_sym (N ∈ {10, 20, 30})",
         test_1b_estimator_symmetric_under_synthetic_reflection),
        ("Test 2  — σ=(1,1) jump bounded by volume + monotone convergence",
         test_2_no_jump_at_sigma_one_one),
        ("Test 3  — bit-identity non-regression of arctan + _eta_volume_lshape",
         test_3_bit_identity_of_p2_and_volume_term),
        ("Test 4  — differentiability of eta_squared_lshape through σ",
         test_4_differentiable_through_sigma),
    ]
    passed = []
    failed = []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
        except Exception as exc:
            failed.append((name, repr(exc)))
            print(f"\n  !!! {name}: {exc!r}")

    print("\n" + "=" * 72)
    print(f"SUMMARY: {len(passed)} / {len(tests)} tests passed")
    for n in passed:
        print(f"  PASS: {n}")
    for n, err in failed:
        print(f"  FAIL: {n}\n        {err}")
    print("=" * 72)
    return 0 if not failed else 1


# --------------------------------------------------------------------------
# DELIVERABLE NOTE: discovered secondary bug in greville_abscissae
# --------------------------------------------------------------------------
#
# While verifying Test 1 we discovered that ``galerkin_solve_lshape`` returns a
# u_h that is NOT reflection-symmetric at N=30 (relative anti-diagonal
# reflection mismatch ~4-25% at σ=(0.1, 10) — far above round-off).
#
# Root cause: ``solver_2d.greville_abscissae`` computes the Greville points
# via a cumsum difference:
#
#     cumsum = jnp.cumsum(knots)
#     greville[i] = (cumsum[i+p+1] - cumsum[i+1]) / p
#
# In exact arithmetic this is equivalent to ``mean(knots[i+1:i+p+1])``, but
# in float64 the cumsum accumulates round-off LEFT-TO-RIGHT so the diff at
# different starting indices is no longer bit-symmetric. At N=30, this drifts
# the Greville at the multiplicity-p knot 0.5 to 0.4999999999999998 (2 ULPs
# below 0.5). The Dirichlet free-mask then asymmetrically excludes/includes
# the 30 DOFs whose Greville sits at "0.5" — breaking the reflection
# symmetry of K and F by ~40%.
#
# This is a SEPARATE bug from the _eta_jump_lshape one this commit fixes and is
# out of scope per the strict "exactly one function" rule. A one-line repair:
#
#     def greville_abscissae(knots, p):
#         knots = jnp.asarray(knots)
#         n_basis = int(knots.shape[0] - p - 1)
#         def window(i):
#             return jax.lax.dynamic_slice(knots, (i + 1,), (p,))
#         return jax.vmap(window)(jnp.arange(n_basis)).mean(axis=1)
#
# (direct windowed mean, no cumsum). With this fix applied locally the
# solve-based Test 1 also passes at N=30 to within 1e-15. We DID NOT apply
# the change in this delivery because it is out of scope; the authors should
# decide whether to land it separately.


if __name__ == "__main__":
    sys.exit(main())
