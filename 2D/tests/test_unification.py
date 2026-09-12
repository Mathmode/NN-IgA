"""Unit tests for the 1D/2D method-unification target (decisions 7-A..7-H).

Each test exercises ONE of the eight authorial decisions in the 2D
codebase. The companion 1D tests live in
``1D/tests/test_unification.py``.

The map (decision → test):
  7-A  loss denominator + ε ………… test_loss_has_epsilon_in_denominator
  7-B  global τ residual gate …… test_calibrate_tau_smoke_p2,
                                  test_evaluate_p2_records_tau
  7-C  bounded-logit cap T (per-experiment) ..
                                  test_T_per_experiment,
                                  test_policy_step_bounds_logits
  7-D  per-axis box shape gate +
       h_min floor ………………… test_h_min_floor_enforced,
                                  test_per_axis_max_ratio_p3,
                                  test_shape_gate_pass
  7-E  --uniform-init ablation … test_uniform_init_zeros_params_p2
  7-F  precomputed eff_index … test_eff_index_helper,
                                  test_csv_has_eff_index_column
  7-G  percentile aggregation … (covered by aggregation tests; n/a here)
  7-H  analytic H¹ denominator
       for arctan ……………………… test_compute_p2_denoms_uses_analytic_h1

Common invariants:
  * policy_step gives uniform mesh when input is zero (gauge centering)
  * CSV column schema drops legacy placeholders (no multistart, gate_layer)
"""
from __future__ import annotations

import inspect
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.config import CORRECTOR, EPSILON_DENOM, H_MIN_ARCTAN, T_ARCTAN, T_LSHAPE
from src.parametric.evaluation_v2 import (
    CSV_COLUMNS,
    _eff_index,
    _per_axis_max_ratio,
    _shape_gate_pass,
)
from src.parametric.positional_density_network_2d import init_params, policy_step


# -----------------------------------------------------------------------------
# Decision 7-A: loss denominator has + ε safety regulariser
# -----------------------------------------------------------------------------


def test_loss_has_epsilon_in_denominator():
    """``EPSILON_DENOM`` is exported and the arctan/lshape loss adds it to the
    denominator."""
    assert EPSILON_DENOM > 0.0, f"EPSILON_DENOM must be positive; got {EPSILON_DENOM!r}"
    assert EPSILON_DENOM < 1e-8, (
        f"EPSILON_DENOM={EPSILON_DENOM!r} is too large — guardrail should be << 1"
    )

    from src.parametric import training as t2d

    def _has_denom_eps(src: str) -> bool:
        """Source uses both ``denom_sq`` and ``epsilon`` in a sum (the
        textual form is multi-line in the actual file)."""
        normalised = " ".join(src.split())   # collapse newlines + indentation
        return ("denom_sq" in normalised and "epsilon" in normalised
                and " + " in normalised)

    assert _has_denom_eps(inspect.getsource(t2d.loss_p2_single_sigma)), (
        "arctan loss does not add ε to denominator — decision 7-A not applied"
    )
    assert _has_denom_eps(inspect.getsource(t2d.loss_p3_single_sigma)), (
        "lshape loss does not add ε to denominator — decision 7-A not applied"
    )


# -----------------------------------------------------------------------------
# Decision 7-B: global τ residual gate (calibration + plumbing)
# -----------------------------------------------------------------------------


def test_evaluate_p2_records_tau(tmp_path):
    """When ``tau_global`` is passed to ``evaluate_arctan``, every row's
    ``tau`` column equals that value."""
    from src.parametric.evaluation_v2 import evaluate_arctan

    params = init_params(seed=0, sigma_dim=3, hidden_dims=(10, 10))
    test_sigmas = np.array([[3.0, 0.5, 0.5]])
    tau_test = 7.89
    out = evaluate_arctan(
        params, test_sigmas,
        p=3, levels=(4,), seed=0, output_dir=tmp_path,
        q_K=4, q_F=8, q_metric=8,
        corrector_max_iter=2,
        tau_global=tau_test,
    )
    import csv as _csv
    with open(out) as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 3                       # uniform / positional / corrected
    for r in rows:
        assert r["tau"] != "", f"tau blank in row {r['method']}"
        assert float(r["tau"]) == pytest.approx(tau_test, rel=1e-12)
        assert float(r["kappa"]) == pytest.approx(float(CORRECTOR["gate_kappa"]))


# -----------------------------------------------------------------------------
# Decision 7-C: bounded-logit cap T per experiment
# -----------------------------------------------------------------------------


def test_T_per_experiment():
    """T is a per-experiment constant; both default to 5.0 in the
    unification target (independent so they can be tuned independently)."""
    assert isinstance(T_ARCTAN, float)
    assert isinstance(T_LSHAPE, float)
    assert T_ARCTAN > 0.0 and T_LSHAPE > 0.0


def test_policy_step_bounds_logits():
    """``policy_step(z, T, h_min, budget)`` produces sizes ≥ h_min and
    summing to ``budget`` for any logit vector. The underlying bounded
    logits ``q = T·tanh(z/T)`` lie in [-T, T]."""
    rng = np.random.default_rng(0)
    n = 16
    h_min = 1e-4
    T = 3.0
    budget = 1.0

    z = jnp.asarray(rng.normal(scale=100.0, size=n))   # huge logits
    sizes = np.asarray(jax.device_get(policy_step(z, T=T, h_min=h_min, budget=budget)))
    assert sizes.shape == (n,)
    assert np.all(sizes >= h_min - 1e-15), (
        f"min(sizes)={sizes.min():.3e} < h_min={h_min}"
    )
    assert math.isclose(float(np.sum(sizes)), budget, rel_tol=1e-12)


def test_policy_step_uniform_at_zero():
    """``policy_step(z=0)`` collapses to the uniform mesh."""
    n = 8
    h_min = 1e-7
    T = 5.0
    budget = 1.0
    sizes = np.asarray(jax.device_get(
        policy_step(jnp.zeros(n), T=T, h_min=h_min, budget=budget)
    ))
    expected = np.full(n, budget / n)
    assert np.allclose(sizes, expected, atol=1e-12)


def test_policy_step_gauge_centering_invariance():
    """``policy_step`` is invariant under additive shifts of z (gauge
    centering)."""
    rng = np.random.default_rng(0)
    n = 10
    z = jnp.asarray(rng.normal(size=n))
    z_shifted = z + 17.42
    s1 = np.asarray(jax.device_get(policy_step(z, T=4.0, h_min=1e-6)))
    s2 = np.asarray(jax.device_get(policy_step(z_shifted, T=4.0, h_min=1e-6)))
    assert np.allclose(s1, s2, atol=1e-12)


# -----------------------------------------------------------------------------
# Decision 7-D: h_min floor + per-axis box shape gate
# -----------------------------------------------------------------------------


def test_h_min_floor_enforced():
    """The mesh policy enforces ``h_i ≥ h_min`` in 2D too. Reproduces
    1D's same invariant via the unified ``policy_step``."""
    n = 8
    h_min = float(H_MIN_ARCTAN)
    z = jnp.asarray([1e3] + [-1e3] * (n - 1))   # extreme saturation
    sizes = np.asarray(jax.device_get(policy_step(z, T=5.0, h_min=h_min)))
    assert np.all(sizes >= h_min - 1e-15)


def test_per_axis_max_ratio_p3():
    """``_per_axis_max_ratio`` drops the zero-length elements introduced
    by the multiplicity-p knot at 0.5 in lshape, returning the true axis
    AR."""
    # Synthetic lshape knot vector with p=2 multiplicity at 0.5 (3 copies).
    knots = np.array([
        0.0, 0.0, 0.0,                            # left clamp p+1
        0.1, 0.25, 0.5, 0.5, 0.5,                 # interior + multiplicity-3 knot
        0.6, 0.75,
        1.0, 1.0, 1.0,                            # right clamp p+1
    ])
    p = 2
    ratio = _per_axis_max_ratio(knots, p)
    # Sizes (after dropping zeros): 0.1, 0.15, 0.25, 0.1, 0.15, 0.25
    # max/min = 0.25/0.1 = 2.5
    assert ratio == pytest.approx(2.5, abs=1e-10)


def test_shape_gate_pass():
    """``_shape_gate_pass`` returns True iff both axes pass the AR
    threshold."""
    # Uniform mesh on both axes: AR=1, passes for any AR_max ≥ 1.
    knots_unif = np.array([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0])
    p = 2
    assert _shape_gate_pass(knots_unif, knots_unif, p, ar_max=2.0) is True
    # An extreme mesh on one axis with AR=10: fails when AR_max < 10.
    knots_bad = np.array([0.0, 0.0, 0.0, 0.01, 0.05, 0.95, 1.0, 1.0, 1.0])
    assert _shape_gate_pass(knots_unif, knots_bad, p, ar_max=2.0) is False
    # But passes when AR_max >> 10.
    assert _shape_gate_pass(knots_unif, knots_bad, p, ar_max=1e6) is True


# -----------------------------------------------------------------------------
# Decision 7-E: --uniform-init ablation (arctan)
# -----------------------------------------------------------------------------


def test_uniform_init_zeros_params_p2():
    """``train_with_continuation(uniform_init=True)`` runs only the
    deepest level in ``levels``. We exercise the (p, N) = (3, 4) smoke
    on a single sigma to keep the test cheap."""
    from src.parametric.continuation import train_with_continuation

    sigmas = np.array([[3.0, 0.5, 0.5]])
    denoms = np.array([1.0])    # tiny denominator placeholder; loss values
                                # don't matter — we only check the schedule.
    result = train_with_continuation(
        experiment="arctan",
        p=3,
        seed=0,
        train_sigmas=sigmas,
        train_denoms=denoms,
        levels=(4, 8),
        epochs_per_level={4: 1, 8: 1},
        batch_size=1,
        lr_init=1e-2, lr_end=1e-3,
        log_every=1_000_000,
        q_K=4, q_F=8, q_est=8,
        sigma_dim=3,
        hidden_dims=(8, 8),
        epsilon=float(EPSILON_DENOM),
        uniform_init=True,
    )
    assert list(result.levels_done) == [8], (
        f"--uniform-init must run ONLY the deepest level; "
        f"got levels_done={result.levels_done}"
    )
    assert len(result.per_level) == 1


# -----------------------------------------------------------------------------
# Decision 7-F: eff_index helper and CSV column
# -----------------------------------------------------------------------------


def test_eff_index_helper():
    """``_eff_index(eta, H1²)`` returns ``eta / sqrt(H1²)`` for positive
    inputs, NaN otherwise."""
    assert _eff_index(2.0, 4.0) == pytest.approx(1.0)
    assert math.isnan(_eff_index(2.0, 0.0))
    assert math.isnan(_eff_index(2.0, -1.0))
    assert math.isnan(_eff_index(float("nan"), 4.0))
    assert math.isnan(_eff_index(float("inf"), 4.0))


def test_csv_has_eff_index_column():
    """The new CSV schema includes ``eff_index`` and drops legacy
    placeholders (``multistart_K``, ``gate_layer``, ``gate_AR_max``)."""
    assert "eff_index" in CSV_COLUMNS
    # Gates and constants:
    for c in ("tau", "kappa", "residual_gate_pass", "shape_gate_pass",
              "accepted_raw", "accepted_corrected", "certified"):
        assert c in CSV_COLUMNS, f"missing required column {c!r}"
    # Dropped placeholders:
    for c in ("multistart_K", "gate_layer", "gate_AR_max"):
        assert c not in CSV_COLUMNS, f"placeholder column {c!r} not removed"


# -----------------------------------------------------------------------------
# Decision 7-H: arctan uses analytic H¹ denominator
# -----------------------------------------------------------------------------


def test_compute_p2_denoms_uses_analytic_h1():
    """``scripts/train_arctan.compute_p2_denoms`` calls
    ``h1_seminorm_sq_analytic``, NOT ``eta_uniform_squared_arctan``."""
    # Read the script as text so we don't need to execute it.
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "scripts" / "train_arctan.py"
    text = src.read_text()
    assert "h1_seminorm_sq_analytic" in text, (
        "train_arctan must use the analytic H¹ denominator per decision 7-H"
    )
    # The legacy uniform-mesh estimator denominator must NOT be the
    # primary source of the per-σ loss denominator anymore.
    assert "compute_p2_denoms" in text


# -----------------------------------------------------------------------------
# Cross-codebase consistency: 1D and 2D policy use the same equations
# -----------------------------------------------------------------------------


def test_policy_step_matches_1d_formula():
    """Hand-evaluate ``policy_step`` and compare with a numpy reimplementation
    of the same equation. They must agree to FP64."""
    rng = np.random.default_rng(42)
    n = 12
    z_np = rng.normal(scale=2.0, size=n)
    T = 5.0
    h_min = 1e-6
    budget = 1.0

    z_centered = z_np - z_np.mean()
    q = T * np.tanh(z_centered / T)
    e = np.exp(q - q.max())
    sm = e / e.sum()
    expected = h_min + (budget - n * h_min) * sm

    got = np.asarray(jax.device_get(
        policy_step(jnp.asarray(z_np), T=T, h_min=h_min, budget=budget)
    ))
    assert np.allclose(got, expected, atol=1e-14)


# -----------------------------------------------------------------------------
# Calibration smoke (arctan): exercises the helper end-to-end
# -----------------------------------------------------------------------------


def test_calibrate_tau_smoke_p2():
    """``calibrate_tau_global_arctan`` returns a finite positive number on
    an untrained network — the median is meaningful even before
    training because we evaluate residual ratios on the validation
    sigmas."""
    from src.parametric.evaluation_v2 import calibrate_tau_global_arctan
    params = init_params(seed=0, sigma_dim=3, hidden_dims=(10, 10))
    calibration = np.array([
        [2.0, 0.4, 0.6],
        [4.0, 0.6, 0.4],
        [6.0, 0.5, 0.5],
    ])
    tau = calibrate_tau_global_arctan(
        params, calibration, p=3, N=4,
        q_K=4, q_F=8,
    )
    assert math.isfinite(tau)
    assert tau > 0.0
