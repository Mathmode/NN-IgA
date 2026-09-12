"""Regression tests for the v3 H¹ metric.

These tests prevent the cancellation bug (fixed in
``src/shared/h1_metric.py`` and propagated through
``diagnostic_metrics_from_state``) from silently resurfacing.

What we're guarding against
---------------------------
The legacy metric computed ``||u_h - u^*||²`` analytically by
expanding the square as ``∫ u_h^2 - 2 ∫ u_h · u^* + ∫ (u^*)^2``.
When ``u_h ≈ u^*`` these three integrals are nearly equal and the
subtraction loses 4-5 decimal digits, giving an inflated (NOT
deflated) total.  A v1-bad sample (β=1.6273, p=3, N=8) had a true
H¹_rel of ~1.9e-4 but the legacy metric reported ~6.7e-2 — a 350x
inflation.

These tests verify:

  1. On VF's production checkpoints, the metric reports
     H¹_rel ≪ 1e-3 (not the inflated ~1e-2 the legacy metric
     would have produced).
  2. The "tight perturbation" probe directly stresses the
     cancellation regime: u_h is u^* projected onto the IGA space
     plus a small perturbation in one DOF; the metric must scale
     with the perturbation, not blow up.
  3. The exact denominators are bit-for-bit consistent with the
     closed-form ``exact_h1_norm_power`` (no regression on the
     normalization).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")

# Add VF root to sys.path so the tests run as a standalone suite too.
VF_ROOT = Path(__file__).resolve().parents[1]
if str(VF_ROOT) not in sys.path:
    sys.path.insert(0, str(VF_ROOT))

import jax  # noqa: E402

from src.nonparametric.metrics import (  # noqa: E402
    diagnostic_metrics_from_state,
)
from src.nonparametric.discretization import (  # noqa: E402
    solve_state, stiffness_quadrature_order,
)
from src.nonparametric.pde import exact_h1_norm_power  # noqa: E402
from src.parametric.evaluation_v2 import _theta_positional  # noqa: E402
from src.parametric.positional_density_network import (  # noqa: E402
    params_from_flat_dict,
)


CHECKPOINT_DIR = VF_ROOT / "results" / "checkpoints"


@pytest.mark.parametrize("beta,seed", [
    (1.6273, 3),   # one of the 5 spot-check samples from investigacion_1
    (1.7897, 2),
    (1.9466, 0),
])
def test_v1_bad_samples_pass_v3_threshold(beta, seed):
    """Network outputs that the legacy metric flagged bad must now
    report H¹_rel well below the 1e-3 threshold under the v3 metric.

    These three samples were verified by hand in the audit:
    the legacy metric reported ~1e-2 to 7e-2, but the true value is
    on the order of 1e-5 to 2e-4.
    """
    ckpt_path = CHECKPOINT_DIR / f"p3_seed{seed}" / "checkpoint_final.npz"
    if not ckpt_path.exists():
        pytest.skip(f"missing checkpoint: {ckpt_path}")
    params = params_from_flat_dict(np.load(str(ckpt_path)))
    p, N, h_min = 3, 8, 1e-8
    theta = _theta_positional(params, float(beta), p=p, N=N)
    q_order = stiffness_quadrature_order(p)
    u_full, cache = solve_state(theta, p, q_order, h_min, float(beta))
    diag = diagnostic_metrics_from_state(u_full, cache, p, float(beta))
    assert float(diag.err_full_h1_rel) < 1e-3, (
        f"v3 metric reported H¹_rel = {float(diag.err_full_h1_rel):.3e} "
        f"≥ 1e-3 at (β={beta}, seed={seed}); the cancellation bug may "
        "have resurfaced or the network may have regressed."
    )


def test_no_cancellation_on_perturbed_solution():
    """Direct stress test of the cancellation regime.

    We solve the IGA system at theta = 0 (uniform mesh, modest N),
    take the resulting u_h (which is NOT u^* but a good IGA
    approximation), perturb one DOF by ε ~ 1e-6, and compare the
    H¹ error before and after.  The metric must:

      - report a finite, non-negative H¹ error.
      - scale roughly with ε (perturbation in one DOF can change
        the H¹ error by order ε).

    The legacy metric on this kind of input typically reports a
    value 100x larger than reality because of cancellation.
    """
    p, N, h_min, beta = 3, 32, 1e-8, 1.7
    theta = np.zeros(N, dtype=np.float64)
    q_order = stiffness_quadrature_order(p)
    u_full_base, cache = solve_state(theta, p, q_order, h_min, beta)

    # Baseline (no perturbation).
    diag_base = diagnostic_metrics_from_state(u_full_base, cache, p, beta)
    h1_base = float(diag_base.err_full_h1_rel)
    assert math.isfinite(h1_base) and h1_base > 0, h1_base

    # Perturb one interior DOF by epsilon.
    eps = 1.0e-6
    u_pert = u_full_base.at[len(u_full_base) // 2].add(eps)
    diag_pert = diagnostic_metrics_from_state(u_pert, cache, p, beta)
    h1_pert = float(diag_pert.err_full_h1_rel)
    assert math.isfinite(h1_pert) and h1_pert > 0, h1_pert

    # The change must be bounded.  An ε perturbation in one DOF can
    # change ||u_h - u*||_H¹ by at most ~ ε times some O(1) factor.
    # Crucially, it must NOT inflate by 100x as the legacy bug would.
    abs_change = abs(h1_pert - h1_base)
    relative_change = abs_change / max(h1_base, 1e-300)
    assert relative_change < 0.1, (  # less than 10% relative change
        f"H¹ relative changed by {relative_change:.2e} on a "
        f"{eps:.1e} perturbation — too much, suggests instability."
    )


def test_exact_denominators_match_closed_form():
    """The relative H¹ error reported by the metric must use the SAME
    closed-form denominator as the pre-fix code; that's the only
    invariant we want preserved bit-exactly across the swap."""
    for beta in (1.55, 1.65, 1.75, 1.85, 1.95):
        # Build a tiny solve so the metric returns a denominator.
        p, N, h_min = 3, 4, 1e-8
        theta = np.zeros(N, dtype=np.float64)
        q_order = stiffness_quadrature_order(p)
        u_full, cache = solve_state(theta, p, q_order, h_min, beta)
        diag = diagnostic_metrics_from_state(u_full, cache, p, beta)
        # Recover the denominator by dividing abs by rel.
        denom_from_metric = float(diag.err_full_h1_abs
                                    / diag.err_full_h1_rel)
        denom_exact = float(exact_h1_norm_power(beta))
        rel = abs(denom_from_metric - denom_exact) / denom_exact
        assert rel < 1e-12, (beta, denom_from_metric, denom_exact, rel)


def test_v3_inside_jit_is_traceable():
    """``diagnostic_metrics_from_state`` is JIT'd and called inside
    ``solve_state_and_metrics`` (also JIT'd) in the non-parametric
    pipeline.  This test verifies the v3 implementation is
    JIT-traceable (no Python-level data-dependent branches)."""
    @jax.jit
    def wrapper(u_full, cache):
        diag = diagnostic_metrics_from_state(u_full, cache, 3, 1.7)
        return diag.err_full_h1_rel

    p, N, h_min, beta = 3, 8, 1e-8, 1.7
    theta = np.zeros(N, dtype=np.float64)
    q_order = stiffness_quadrature_order(p)
    u_full, cache = solve_state(theta, p, q_order, h_min, beta)
    out = float(wrapper(u_full, cache))
    assert math.isfinite(out) and out > 0
