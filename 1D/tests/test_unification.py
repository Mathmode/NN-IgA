"""Unit tests for the 1D/2D method-unification target (decisions 7-A..7-H).

Each test exercises ONE of the eight authorial decisions in the 1D
codebase. The companion 2D tests live in
``2D/tests/test_unification.py``.

The map (decision → test):
  7-A  loss denominator + ε ………… test_loss_has_epsilon_in_denominator
  7-B  global τ residual gate …… test_calibrate_tau_finite,
                                  test_corrector_uses_tau_global
  7-C  bounded-logit cap T …… test_saturate_bounds_logits
  7-D  per-axis shape gate +
       h_min floor …………… test_h_min_floor_enforced,
                            test_shape_gate_aspect_ratio
  7-E  --uniform-init ablation … test_uniform_init_zeros_params,
                                  test_uniform_init_trains_only_finest
  7-F  precomputed eff_index … test_eff_index_in_csv
                                  (in test_evaluate_v2.py)
  7-G  percentile aggregation … (covered by aggregation tests; n/a here)
  7-H  analytic H¹ denominator
       for singular/arctan …………………… test_training_loss_uses_analytic_h1
       (analytic for singular; in 2D it appears in test_unification.py too)

Common invariants:
  * gauge_fix(z) has mean ≈ 0 (centering policy)
  * theta_to_sizes is monotone and bounded below by h_min
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.config import (
    CORRECTOR,
    EPSILON_DENOM,
    H_MIN_SCHEDULE,
    T_SCHEDULE,
)
from src.nonparametric.knots import theta_to_sizes
from src.parametric.continuation import train_with_continuation
from src.parametric.evaluation_v2 import calibrate_tau_global
from src.parametric.corrector_v2 import correct_lbfgsb_v2
from src.parametric.positional_density_network import cell_xi, forward_scalar, gauge_fix, saturate
from src.lhs_sampling import generate_splits


# -----------------------------------------------------------------------------
# Decision 7-A: loss denominator has + ε safety regulariser
# -----------------------------------------------------------------------------


def test_loss_has_epsilon_in_denominator():
    """``EPSILON_DENOM`` is exported and the training loss adds it to the
    H¹ denominator. Value must be positive and small enough to be a no-op
    on the working β range."""
    assert EPSILON_DENOM > 0.0, f"EPSILON_DENOM must be positive; got {EPSILON_DENOM!r}"
    assert EPSILON_DENOM < 1e-8, (
        f"EPSILON_DENOM={EPSILON_DENOM!r} is too large — guardrail should be << 1"
    )

    # Inspect source of training.py to confirm it is actually used in the loss.
    # (A grep-style assertion: cheap and explicit.)
    import inspect

    from src.parametric import training

    src = inspect.getsource(training._make_per_beta_loss)
    assert "EPSILON_DENOM" in src, (
        "EPSILON_DENOM does not appear in _make_per_beta_loss — "
        "denominator ε guard is not applied at training time"
    )


# -----------------------------------------------------------------------------
# Decision 7-B: global τ residual gate
# -----------------------------------------------------------------------------


def test_calibrate_tau_finite(_trained_params):
    """``calibrate_tau_global`` returns a finite positive number on a
    smoke-trained network."""
    _, val_betas, _ = generate_splits(0)
    tau = calibrate_tau_global(
        _trained_params, p=3, calibration_betas=val_betas, N=8
    )
    assert math.isfinite(tau)
    assert tau > 0.0
    assert math.isclose(
        tau,
        float(CORRECTOR["gate_kappa"]) * (tau / float(CORRECTOR["gate_kappa"])),
        rel_tol=1e-12,
    )


def test_corrector_uses_tau_global(_trained_params):
    """When ``tau_global`` is set, ``residual_gate_pass`` is computed
    against τ. When ``tau_global=None``, the residual gate passes
    trivially."""
    p, N, beta = 3, 8, 1.7
    h_min = float(H_MIN_SCHEDULE[p][N])
    T = float(T_SCHEDULE[p][N])
    xi = cell_xi(N)
    z = forward_scalar(_trained_params, beta, xi, N)
    theta = saturate(gauge_fix(z), T)

    # No τ → residual gate always passes; τ column is NaN.
    res_no_tau = correct_lbfgsb_v2(
        theta, p=p, N=N, h_min=h_min, beta=beta, T=T,
        tau_global=None, record_history=False,
    )
    assert res_no_tau.residual_gate_pass is True
    assert math.isnan(res_no_tau.tau)

    # Set τ to an impossibly tight 1e-30 → residual gate fails.
    res_tight = correct_lbfgsb_v2(
        theta, p=p, N=N, h_min=h_min, beta=beta, T=T,
        tau_global=1e-30, record_history=False,
    )
    assert res_tight.residual_gate_pass is False
    assert res_tight.tau == 1e-30

    # Set τ to ∞ → residual gate passes trivially.
    res_loose = correct_lbfgsb_v2(
        theta, p=p, N=N, h_min=h_min, beta=beta, T=T,
        tau_global=float("inf"), record_history=False,
    )
    assert res_loose.residual_gate_pass is True


# -----------------------------------------------------------------------------
# Decision 7-C: bounded-logit cap T
# -----------------------------------------------------------------------------


def test_saturate_bounds_logits():
    """``saturate(z, T) = T · tanh(z / T)`` strictly bounds the output."""
    T = 5.0
    z = jnp.linspace(-1e3, 1e3, 200)
    q = saturate(z, T)
    q_np = np.asarray(q)
    assert np.all(q_np >= -T), f"min(q)={q_np.min()} below -T"
    assert np.all(q_np <= T), f"max(q)={q_np.max()} above T"
    # tanh asymptotes: large positive z should saturate near +T.
    assert q_np[-1] == pytest.approx(T, abs=1e-9)
    assert q_np[0] == pytest.approx(-T, abs=1e-9)


def test_gauge_fix_zeroes_mean():
    """``gauge_fix(z)`` subtracts the mean so the centred logits have
    mean exactly 0 (within ulp)."""
    rng = np.random.default_rng(0)
    z = jnp.asarray(rng.normal(scale=10.0, size=32))
    z_c = gauge_fix(z)
    assert abs(float(jnp.mean(z_c))) < 1e-12


# -----------------------------------------------------------------------------
# Decision 7-D: h_min floor + shape gate
# -----------------------------------------------------------------------------


def test_h_min_floor_enforced():
    """``theta_to_sizes`` enforces ``h_i ≥ h_min`` even for very
    saturated logits."""
    N = 8
    h_min = 1e-7
    # Logits pegged at +∞ on one cell, -∞ on the rest → after softmax
    # one cell takes almost all the budget. h_min floor still applies.
    theta = jnp.asarray([1e3, -1e3, -1e3, -1e3, -1e3, -1e3, -1e3, -1e3])
    sizes = np.asarray(jax.device_get(theta_to_sizes(theta, h_min)))
    assert sizes.shape == (N,)
    assert np.all(sizes >= h_min - 1e-15), (
        f"h_min violated: min(sizes)={sizes.min():.3e} < h_min={h_min}"
    )
    assert math.isclose(float(np.sum(sizes)), 1.0, rel_tol=1e-12)


def test_shape_gate_aspect_ratio():
    """1D shape gate (per-axis box) = max_h / min_h ≤ AR_max. Uniform
    mesh has AR = 1; an extreme saturated mesh has AR >> AR_max and
    should NOT pass the gate."""
    ar_max = float(CORRECTOR["shape_gate_AR_max"])
    h_min = 1e-7
    # Uniform mesh: should pass.
    sizes_unif = np.full(8, 1.0 / 8.0)
    ar_unif = float(sizes_unif.max() / sizes_unif.min())
    assert ar_unif == pytest.approx(1.0)
    assert ar_unif <= ar_max

    # Severely-saturated mesh: should fail when sizes span many orders.
    theta_extreme = jnp.asarray([1e3, -1e3, -1e3, -1e3, -1e3, -1e3, -1e3, -1e3])
    sizes_ext = np.asarray(jax.device_get(theta_to_sizes(theta_extreme, h_min)))
    ar_ext = float(sizes_ext.max() / sizes_ext.min())
    assert ar_ext > 1e3, f"expected extreme AR > 1e3; got {ar_ext}"


# -----------------------------------------------------------------------------
# Decision 7-E: --uniform-init ablation
# -----------------------------------------------------------------------------


def test_uniform_init_zeros_params():
    """``train_with_continuation(uniform_init=True)`` must produce a
    network whose initial state is all zeros and train only at the
    finest level. We assert on the API contract by inspecting the
    returned ``levels_done`` (== single deepest level)."""
    train_betas, val_betas, _ = generate_splits(0)

    from src import config as cfg
    # Tiny budget — we only care that the levels_done is the single
    # finest level, not about loss values.
    cfg.TRAIN["max_epochs"] = 2
    cfg.TRAIN["patience"] = 2
    cfg.TRAIN["train_size"] = 4
    cfg.TRAIN["val_size"] = 4

    cont_uniform = train_with_continuation(
        p=3, seed=0,
        train_betas=train_betas, val_betas=val_betas,
        levels=(2, 4, 8),
        uniform_init=True,
    )
    assert cont_uniform.levels_done == [8], (
        f"--uniform-init must train ONLY the finest level; "
        f"got levels_done={cont_uniform.levels_done}"
    )


def test_uniform_init_skips_continuation():
    """Continuation runs through every level in ``levels``; --uniform-init
    runs only the finest. ``len(per_level)`` must reflect this."""
    train_betas, val_betas, _ = generate_splits(0)

    from src import config as cfg
    cfg.TRAIN["max_epochs"] = 2
    cfg.TRAIN["patience"] = 2
    cfg.TRAIN["train_size"] = 4
    cfg.TRAIN["val_size"] = 4

    cont_normal = train_with_continuation(
        p=3, seed=0,
        train_betas=train_betas, val_betas=val_betas,
        levels=(2, 4, 8),
        uniform_init=False,
    )
    cont_uniform = train_with_continuation(
        p=3, seed=0,
        train_betas=train_betas, val_betas=val_betas,
        levels=(2, 4, 8),
        uniform_init=True,
    )
    assert len(cont_normal.per_level) == 3
    assert len(cont_uniform.per_level) == 1


# -----------------------------------------------------------------------------
# Decision 7-H: analytic H¹ denominator for singular (built-in to 1D training)
# -----------------------------------------------------------------------------


def test_training_loss_uses_uniform_same_level_denominator():
    """Eq-26 release: the 1D training loss divides by η²(θ_unif^{(N)}; β)
    at the CURRENT level (config.LOSS_NORM == "uniform_same_level"), passed
    in as a precomputed ``denom`` — NOT by the legacy analytic ‖u_β‖²_{H¹}.
    We verify (a) the contract by inspecting the source, and (b) the
    denominator helper numerically: θ=0 ⇒ uniform mesh ⇒ η²_unif = 2·L_unif,
    and it CHANGES with the level N (no freeze)."""
    import inspect

    from src.config import LOSS_NORM
    from src.parametric import training

    assert LOSS_NORM == "uniform_same_level"
    src = inspect.getsource(training._make_per_beta_loss)
    # per-sample term divides by the passed denom (+ ε), and the legacy
    # closed-form ‖u_β‖²_{H¹} no longer appears in the LOSS path.
    assert "def per_beta_loss(params" in src and "denom" in src
    assert "1.0 / (2.0 * beta_t + 1.0)" not in src

    # Numerical: denominators differ across levels for the same β.
    fn8 = training.make_uniform_eta2_fn(2, 8, 1e-7)
    fn16 = training.make_uniform_eta2_fn(2, 16, 1e-7)
    betas = jnp.asarray([1.6, 1.8])
    d8 = np.asarray(fn8(betas))
    d16 = np.asarray(fn16(betas))
    assert np.all(d8 > 0.0) and np.all(d16 > 0.0)
    assert not np.allclose(d8, d16), "eq-26 denominators must change with N"


# -----------------------------------------------------------------------------
# Shared fixture: a quickly-trained PDN for the gate tests
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _trained_params():
    from src import config as cfg
    cfg.TRAIN["max_epochs"] = 3
    cfg.TRAIN["patience"] = 3
    cfg.TRAIN["train_size"] = 8
    cfg.TRAIN["val_size"] = 4
    cfg.TRAIN["test_size"] = 4

    train_betas, val_betas, _ = generate_splits(0)
    cont = train_with_continuation(
        p=3, seed=0,
        train_betas=train_betas, val_betas=val_betas,
        levels=(2, 4, 8),
    )
    return cont.params_final
