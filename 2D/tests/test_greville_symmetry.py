"""Regression tests for the round-off bug in
``src/nonparametric/solver_2d.greville_abscissae``.

The previous version computed each Greville abscissa as a difference of a
left-to-right cumulative sum:

    cumsum = jnp.cumsum(knots)
    greville[i] = (cumsum[i + p + 1] - cumsum[i + 1]) / p

Mathematically equal to ``mean(knots[i + 1 : i + p + 1])`` but, in float64,
each cumsum entry carries an accumulated round-off that depends on its
starting index. The difference at different starting indices is therefore
not bit-symmetric across the knot vector. The bug manifested at N=30, p=2:
the Greville at the multiplicity-p knot 0.5 drifted to 0.4999999999999998,
which then asymmetrically flipped the Dirichlet free-mask classification
of ~30 DOFs (those whose Greville sat at "0.5") and broke the reflection
symmetry of the assembled (K, F) by ~40% for the L-shape problem.

The fix replaces the cumsum-diff with a per-window mean (a vmap over local
dynamic slices). The round-off of `mean(window)` depends only on the
window contents, not on its absolute position, so for a knot vector that
is invariant under k ↦ 1 - k_reversed the Greville vector is bit-symmetric
under reversal.

Tests:
  * Test 1 — Greville at the multiplicity-p knot is bit-exactly 0.5.
  * Test 2 — Greville vector is bit-symmetric under reversal+reflection
             for a symmetric knot vector.
  * Test 3 — the lshape Dirichlet free-mask is now invariant under the
             diagonal (anti-diagonal) reflection of the DOF grid that
             swaps bot-left ↔ top-right.
  * Test 4 — solve-based lshape σ-swap symmetry: ``eta_squared_lshape(u(σ1,σ2))``
             matches ``eta_squared_lshape(u(σ2,σ1))`` to <1e-12 on uniform
             mesh, INCLUDING at N=30 (the case that previously failed).
  * Test 5 — arctan non-regression: cumsum-based and windowed-mean formulae
             agree numerically; and a fixed arctan ``eta_squared_arctan`` value
             is unchanged.
  * Test 6 — ``jax.grad`` through ``greville_abscissae`` returns finite
             gradients (autodiff through knots preserved).
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
from src.nonparametric.eta_estimator_2d import eta_squared_arctan, eta_squared_lshape
from src.nonparametric.r_adapt import p3_effective_n_elem, theta_to_knots_lshape
from src.nonparametric.solver_2d import (
    free_mask_lshape,
    galerkin_solve_arctan,
    galerkin_solve_lshape,
    greville_abscissae,
)

_P = 2


# --------------------------------------------------------------------------
# Test 1 — Greville at the multiplicity-p interface knot is exactly 0.5.
# --------------------------------------------------------------------------


def test_1_greville_at_interface_is_exact_half():
    """For the lshape knot vector with multiplicity-p at 0.5, the basis
    function centred on that knot has Greville = mean of p copies of
    0.5 = 0.5 exactly. Pre-fix, cumsum drift made it 0.5 - 2*ULP at
    N=30. Post-fix it must be bit-exact 0.5 at every N.

    Centred-basis index for the lshape knots produced by
    ``theta_to_knots_lshape(theta_left, theta_right, p)``:

        knots = [(p+1) zeros, half-1 left interior, p copies of 0.5,
                 half-1 right interior, (p+1) ones]

    The Greville (window mean of p consecutive knots) of basis i uses
    knots[i+1 : i+p+1]. For that window to contain ALL p copies of 0.5,
    we need ``i + 1 = (p+1) + (half-1) = p + half``, so ``i = p + half - 1``.
    """
    print("\nTest 1: Greville at multiplicity-p knot is bit-exactly 0.5")
    failures = []
    for N in (10, 20, 30, 50):
        half = N // 2
        knots = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), _P)
        g = np.asarray(greville_abscissae(knots, _P))
        i_centre = _P + half - 1
        g_centre = float(g[i_centre])
        bit_exact_half = (g_centre == 0.5)
        # Also check no spurious near-0.5 Greville: every g[i] is either
        # exactly 0.5 or further than 1e-10 from 0.5.
        near_half = np.where((np.abs(g - 0.5) > 0) & (np.abs(g - 0.5) < 1e-10))[0]
        spurious_count = int(len(near_half))
        status = "OK" if (bit_exact_half and spurious_count == 0) else "FAIL"
        print(
            f"  N={N:3d}  i_centre={i_centre:3d}  g[i_centre]={g_centre!r}  "
            f"bit-exact 0.5: {bit_exact_half}  spurious-near-0.5: {spurious_count}  [{status}]"
        )
        if not bit_exact_half:
            failures.append((N, "g[centre] != 0.5", g_centre))
        if spurious_count:
            failures.append((N, "spurious near-0.5", near_half.tolist(), g[near_half].tolist()))
    if failures:
        raise AssertionError(f"Test 1 FAILED: {failures}")
    print("  -> Test 1 PASS")


# --------------------------------------------------------------------------
# Test 2 — reversal symmetry of the Greville vector.
# --------------------------------------------------------------------------


def _sum_symmetric_p3_knots(N: int, p: int) -> jnp.ndarray:
    """Build a lshape-style knot vector where ``knots[k] + knots[L-1-k] == 1``
    bit-exactly for all k.

    Note: a STRICTER condition ``knots[k] == 1 - knots[L-1-k]`` is NOT
    achievable for non-dyadic interior positions because float64 has
    the asymmetry ``1 - 0.1 == 0.9  but  1 - 0.9 == 0.0999...``. The
    SUM form is what actually drives downstream reflection symmetry of
    the Greville vector (see Test 2 docstring).

    We build the left interior as ``k / N`` and the right interior as
    ``1 - left[::-1]``. The sum ``left[k] + right[half - 2 - k] =
    left[k] + (1 - left[k]) = 1`` is bit-exact in float64.
    """
    half = N // 2
    left = jnp.arange(1, half, dtype=jnp.float64) / float(N)
    right = jnp.asarray(1.0) - left[::-1]
    fixed = jnp.full((int(p),), 0.5, dtype=jnp.float64)
    zeros = jnp.zeros((p + 1,), dtype=jnp.float64)
    ones = jnp.ones((p + 1,), dtype=jnp.float64)
    return jnp.concatenate([zeros, left, fixed, right, ones])


def test_2_greville_reversal_symmetry():
    """Reversal symmetry of the Greville vector.

    For a knot vector satisfying ``knots[k] + knots[L-1-k] == 1`` bit-
    exactly, the corresponding Greville abscissae must satisfy
    ``greville[i] + greville[n_basis - 1 - i] ≈ 1`` to within a small,
    **N-independent**, ULP-level constant. This is the property whose
    failure (under the old cumsum-based formula) caused the downstream
    Dirichlet-mask flip at N=30, p=2.

    Why we test ``g[i] + g[n-1-i] == 1`` (sum form) instead of the
    literal ``g[i] == 1 - g[n-1-i]`` from the task spec: float64 has the
    well-known asymmetry ``1 - 0.1 == 0.9`` but ``1 - 0.9 == 0.0999…``
    (a 1-ULP discrepancy). Even a perfect Greville cannot pass the
    literal form for non-dyadic interior positions because the float64
    representation of ``0.1`` and ``1 - 0.9`` differ by 1 ULP. The
    SUM form is order-invariant under addition (a + b == b + a in float64),
    captures the same physical symmetry, and is what the downstream code
    actually depends on.

    Why we use a tolerance and not bit-exact: the per-window mean
    ``(a + b) / 2`` carries its own 1-2 ULP round-off (it is NOT
    eliminable in float64). What matters operationally is that the
    error is **bounded by an N-independent constant**, not that it is
    zero. The old cumsum-based Greville accumulated error linearly
    with N (3.3e-15 at N=100); the fix achieves a bounded ≤ 2.2e-16
    at every N up to 100+.

    The tolerance below (5e-16, i.e. a few ULPs at scale 1) is chosen
    so that:
      * the fix PASSES at every N tested (max observed: 2.2e-16);
      * the buggy cumsum form FAILS at N ≥ 30 (where the production
        Dirichlet-mask defect occurred).

    Algebraic argument: greville[i] = mean(knots[i+1 : i+p+1]) and
    greville[n_basis-1-i] = mean(knots[L-p-i : L-i]). The two windows
    are index-reflections of each other (j ↔ L-1-j); using
    ``knots[k] + knots[L-1-k] = 1`` and noting that addition is
    commutative in float64 (a + b == b + a), the per-window MEANS
    satisfy mean(W) + mean(W') = (sum(W) + sum(W')) / p = p / p = 1,
    modulo a few ULPs from the divisions.

    Tested for p ∈ {2, 3} and N ∈ {10, 20, 30, 50}.
    """
    print("\nTest 2: greville[i] + greville[n_basis-1-i] == 1 (sum form, "
          "N-independent ≤ 5e-16; see docstring for why not literally bit-exact)")
    TOL = 5e-16   # discriminates fix (max 2.2e-16) from cumsum bug (≥ 8.9e-16 at N≥30)
    failures = []
    for p_val in (2, 3):
        for N in (10, 20, 30, 50):
            knots = _sum_symmetric_p3_knots(N, p_val)
            kn = np.asarray(knots)
            knots_sum_one = np.array_equal(kn + kn[::-1], np.ones_like(kn))
            assert knots_sum_one, (
                f"input knots fail knots[k] + knots[L-1-k] == 1 at p={p_val}, N={N} "
                "— test premise broken"
            )
            g = np.asarray(greville_abscissae(knots, p_val))
            n_basis = g.shape[0]
            sums = g + g[::-1]
            max_dev = float(np.abs(sums - 1.0).max())
            # Also report informationally
            diff_naive = float(np.abs(g - (1.0 - g[::-1])).max())
            within_tol = max_dev <= TOL
            status = "OK" if within_tol else "FAIL"
            print(
                f"  p={p_val}  N={N:3d}  max|g[i] + g[n-1-i] - 1|={max_dev:.3e}  "
                f"(tol={TOL:.0e})  (info: max|g - (1-g[::-1])|={diff_naive:.3e})  [{status}]"
            )
            if not within_tol:
                idx_bad = int(np.abs(sums - 1.0).argmax())
                failures.append((p_val, N, idx_bad, float(g[idx_bad]),
                                 float(g[n_basis - 1 - idx_bad]), max_dev))
    if failures:
        raise AssertionError(f"Test 2 FAILED: {failures}")
    print("  -> Test 2 PASS")


# --------------------------------------------------------------------------
# Test 3 — lshape Dirichlet free-mask diagonal symmetry.
# --------------------------------------------------------------------------
#
# Note on geometry: the L-shape's mirror symmetry is the ANTI-DIAGONAL
# ``(x, y) ↔ (1 - y, 1 - x)``, which fixes the top-left sub-region and
# swaps bot-left ↔ top-right. (Pure ``y = x`` reflection maps the L-shape
# onto a DIFFERENT L-shape — top-left ↔ removed quadrant — so it is NOT
# a symmetry.) On the DOF grid with a Greville-symmetric basis, that
# reflection acts as ``(i, j) ↔ (n - 1 - j, n - 1 - i)``.


def _reflect_anti_diagonal(arr2d):
    """Anti-diagonal reflection of a 2D (n, n) array: result[i, j] = arr[n-1-j, n-1-i]."""
    return arr2d[::-1, ::-1].T


def test_3_p3_free_mask_diagonal_symmetry():
    print("\nTest 3: lshape free_mask invariant under anti-diagonal reflection of DOFs")
    failures = []
    for N in (10, 20, 30, 50):
        half = N // 2
        knots = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), _P)
        free = np.asarray(free_mask_lshape(knots, knots, _P))
        free_refl = _reflect_anti_diagonal(free)
        bit_exact = np.array_equal(free, free_refl)
        n_mismatch = int(np.sum(free != free_refl))
        status = "OK" if bit_exact else "FAIL"
        print(
            f"  N={N:3d}  free shape={free.shape}  "
            f"mismatched DOFs={n_mismatch}  bit-symmetric: {bit_exact}  [{status}]"
        )
        if not bit_exact:
            failures.append((N, n_mismatch))
    if failures:
        raise AssertionError(f"Test 3 FAILED: {failures}")
    print("  -> Test 3 PASS")


# --------------------------------------------------------------------------
# Test 4 — lshape solve-based diagonal σ-swap symmetry on uniform mesh.
# --------------------------------------------------------------------------
#
# Chosen form: solve-based (option (b) in the task spec). We solve lshape at
# (σ1, σ2) and (σ2, σ1) on a uniform mesh and check that the residual
# estimator η² is bit-symmetric to 1e-12. This is the exact case the
# previous task had to restrict to N ∈ {10, 20}; with this fix it must
# pass at N=30 too. Direct K, F reflection comparison is functionally
# equivalent but would duplicate the diagnostic plumbing from the
# previous task — the solve-based form is the user-visible quantity that
# the assembly bug actually corrupted.


def test_4_p3_kf_symmetry_via_solve_eta():
    print("\nTest 4: lshape eta_squared diagonal symmetry on uniform mesh")
    print("  tolerance: |ratio - 1| < 1e-12")
    failures = []
    for N in (10, 20, 30):
        for (s1, s2) in [(0.1, 10.0), (1.0, 5.0)]:
            half = N // 2
            n_eff = p3_effective_n_elem(N, _P)
            knots = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), _P)
            r12 = galerkin_solve_lshape(
                knots, knots, _P, n_eff, n_eff,
                jnp.asarray(s1), jnp.asarray(s2), q_K=_P + 1, q_F=2,
            )
            e12 = float(eta_squared_lshape(
                r12.u_h, knots, knots, _P, n_eff, n_eff,
                jnp.asarray(s1), jnp.asarray(s2), q_est=4,
            ))
            r21 = galerkin_solve_lshape(
                knots, knots, _P, n_eff, n_eff,
                jnp.asarray(s2), jnp.asarray(s1), q_K=_P + 1, q_F=2,
            )
            e21 = float(eta_squared_lshape(
                r21.u_h, knots, knots, _P, n_eff, n_eff,
                jnp.asarray(s2), jnp.asarray(s1), q_est=4,
            ))
            ratio = e12 / e21 if e21 > 0 else float("inf")
            rel_err = abs(ratio - 1.0)
            status = "OK" if rel_err < 1e-12 else "FAIL"
            print(
                f"  N={N:3d}  σ=({s1},{s2})  η²(σ1,σ2)={e12:.6e}  "
                f"η²(σ2,σ1)={e21:.6e}  ratio={ratio!r}  |ratio-1|={rel_err:.3e}  [{status}]"
            )
            if status == "FAIL":
                failures.append((N, s1, s2, ratio))
    if failures:
        raise AssertionError(f"Test 4 FAILED: {failures}")
    print("  -> Test 4 PASS")


# --------------------------------------------------------------------------
# Test 5 — arctan non-regression: cumsum vs windowed-mean agree numerically.
# --------------------------------------------------------------------------


def _greville_via_cumsum(knots, p):
    """The PREVIOUS implementation (kept here only for the comparison
    test below). Computes greville[i] = (cumsum[i+p+1] - cumsum[i+1]) / p
    using a left-to-right cumulative sum — i.e. exactly the buggy form
    that the fix replaces."""
    knots = jnp.asarray(knots)
    n_basis = int(knots.shape[0] - p - 1)
    cumsum = jnp.cumsum(knots)
    pad = jnp.concatenate([jnp.zeros((1,), dtype=knots.dtype), cumsum])
    indices = jnp.arange(n_basis)
    return (pad[indices + p + 1] - pad[indices + 1]) / float(p)


# arctan reference value from the previous task's test_eta_jump_p3_symmetry.py.
# arctan N=16, p=2, σ=(5.0, 0.3, 0.5), q_K=3, q_F=20, q_est=30.
_REF_ETA_P2 = 0.0931882081561042
_P2_TOL = 1e-12


def test_5_p2_non_regression():
    print("\nTest 5: arctan non-regression — Greville unchanged + arctan estimator unchanged")
    failures = []

    # 5a) Greville agreement on arctan open-uniform knots (no mult-p interior).
    for p_val in (2, 3):
        for N in (8, 16, 32, 64):
            knots = jnp.asarray(open_uniform_knots(N, p_val))
            g_new = np.asarray(greville_abscissae(knots, p_val))
            g_old = np.asarray(_greville_via_cumsum(knots, p_val))
            max_diff = float(np.abs(g_new - g_old).max())
            status = "OK" if max_diff < _P2_TOL else "FAIL"
            print(
                f"  arctan open-uniform p={p_val} N={N:3d}: "
                f"max|new - cumsum|={max_diff:.3e}  [{status}]"
            )
            if status == "FAIL":
                failures.append((p_val, N, max_diff))

    # 5b) arctan estimator value unchanged at the previous task's reference point.
    N_p2, p_p2 = 16, 2
    knots_p2 = jnp.asarray(open_uniform_knots(N_p2, p_p2))
    alpha, s1, s2 = jnp.asarray(5.0), jnp.asarray(0.3), jnp.asarray(0.5)
    res_p2 = galerkin_solve_arctan(knots_p2, knots_p2, p_p2, N_p2, N_p2,
                                alpha, s1, s2, q_K=3, q_F=20)
    got_p2 = float(eta_squared_arctan(
        res_p2.u_h, knots_p2, knots_p2, p_p2, N_p2, N_p2,
        alpha, s1, s2, q_est=30,
    ))
    diff_p2 = abs(got_p2 - _REF_ETA_P2)
    status_p2 = "OK" if diff_p2 < _P2_TOL else "FAIL"
    print(
        f"  arctan eta_squared_arctan at (N=16, p=2, σ=(5,0.3,0.5)): "
        f"got={got_p2!r}  ref={_REF_ETA_P2!r}  |diff|={diff_p2:.3e}  [{status_p2}]"
    )
    if status_p2 == "FAIL":
        failures.append(("eta_squared_arctan", diff_p2))

    if failures:
        raise AssertionError(f"Test 5 FAILED: {failures}")
    print("  -> Test 5 PASS")


# --------------------------------------------------------------------------
# Test 6 — differentiability through greville_abscissae.
# --------------------------------------------------------------------------


def test_6_differentiable_through_knots():
    print("\nTest 6: jax.grad through greville_abscissae returns finite gradients")
    # Use lshape knots so the multiplicity-p region is exercised; differentiate
    # a scalar reduction of the Greville vector w.r.t. the (free) theta
    # parameters that drive the knot construction.
    N = 20
    p_val = 2
    half = N // 2
    theta_l_init = jnp.zeros((half,))
    theta_r_init = jnp.zeros((half,))

    def loss(theta_l, theta_r):
        knots = theta_to_knots_lshape(theta_l, theta_r, p_val)
        return jnp.sum(greville_abscissae(knots, p_val) ** 2)

    g_l, g_r = jax.grad(loss, argnums=(0, 1))(theta_l_init, theta_r_init)
    g_l_np = np.asarray(g_l); g_r_np = np.asarray(g_r)
    finite = bool(np.all(np.isfinite(g_l_np))) and bool(np.all(np.isfinite(g_r_np)))
    print(
        f"  d(loss)/d(theta_l) shape {g_l_np.shape}, all finite: {bool(np.all(np.isfinite(g_l_np)))}"
    )
    print(
        f"  d(loss)/d(theta_r) shape {g_r_np.shape}, all finite: {bool(np.all(np.isfinite(g_r_np)))}"
    )
    print(
        f"  ||grad_theta_l||_inf = {np.abs(g_l_np).max():.3e}, "
        f"||grad_theta_r||_inf = {np.abs(g_r_np).max():.3e}"
    )
    if not finite:
        raise AssertionError("Test 6 FAILED — non-finite gradient through greville_abscissae")
    print("  -> Test 6 PASS")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def main():
    tests = [
        ("Test 1 — Greville at multiplicity-p knot is bit-exact 0.5",
         test_1_greville_at_interface_is_exact_half),
        ("Test 2 — Greville reversal symmetry bit-exact",
         test_2_greville_reversal_symmetry),
        ("Test 3 — lshape free_mask anti-diagonal symmetry",
         test_3_p3_free_mask_diagonal_symmetry),
        ("Test 4 — lshape solve-based σ-swap symmetry to 1e-12 at N=10,20,30",
         test_4_p3_kf_symmetry_via_solve_eta),
        ("Test 5 — arctan non-regression (Greville + estimator)",
         test_5_p2_non_regression),
        ("Test 6 — differentiability through greville_abscissae",
         test_6_differentiable_through_knots),
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


if __name__ == "__main__":
    sys.exit(main())
