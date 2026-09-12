"""Tests for evaluate_v2 + corrector_v2 (enriched CSV schema, unification).

Multi-start was removed per the unification target — these tests cover
only the single-warm-start corrector and the new gate/eff_index columns.
The 9 tests:
  1. Header has the expected 49 columns (25 v1 + 24 v2)
  2. Bool-like columns parse as 0/1
  3. eta_volume² + eta_neumann² ≈ eta² (numerical decomposition)
  4. logit_saturation fractions in [0, 1]
  5. lbfgsb_nit ≤ lbfgsb_nfev (scipy invariant)
  6. history_loss parses as comma-separated floats
  7. runtime > 0 (sane wall-clock)
  8. eff_index column present, finite, matches eta / sqrt(H1_abs)
  9. corrector_budget override caps lbfgsb_nit
 10. tau_global plumbing: corrected row records tau == tau_global
 11. calibrate_tau_global returns a finite positive number on a smoke run
"""
from __future__ import annotations

import csv
import math

import pytest

from src import config
from src.parametric.continuation import train_with_continuation
from src.parametric.evaluation import (
    SUMMARY_FIELDNAMES as SUMMARY_FIELDNAMES_V1,
)
from src.parametric.evaluation_v2 import (
    SUMMARY_V2_FIELDNAMES,
    V2_NEW_FIELDS,
    calibrate_tau_global,
    evaluate_test_split_v2,
)
from src.lhs_sampling import generate_splits


# -----------------------------------------------------------------------------
# Shared fixture — small training + small evaluation grid
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_params_small():
    """Train a small PDN for tests. Cached at module scope to share across tests."""
    config.TRAIN["max_epochs"] = 5
    config.TRAIN["patience"] = 5
    config.TRAIN["train_size"] = 8
    config.TRAIN["val_size"] = 4
    config.TRAIN["test_size"] = 4

    train_betas, val_betas, _ = generate_splits(0)
    cont = train_with_continuation(
        p=3, seed=0,
        train_betas=train_betas, val_betas=val_betas,
        levels=(2, 4, 8),
    )
    return cont.params_final


# -----------------------------------------------------------------------------
# 1. Schema check
# -----------------------------------------------------------------------------


def test_v2_csv_has_all_required_columns(tmp_path, trained_params_small):
    """51 columns total = 25 v1 + 26 v2 new fields (multistart_K/init dropped,
    eff_index added per decision 7-F, H1_semi_abs/H1_semi_rel added so the
    effectivity index has the correct H¹-seminorm denominator)."""
    _, _, test_betas = generate_splits(0)
    csv_path = tmp_path / "summary_v2.csv"

    evaluate_test_split_v2(
        trained_params_small, p=3, seed=0,
        test_betas=test_betas, eval_levels=(8,),
        incremental_csv_path=csv_path,
        clear_caches_between_levels=False,
        verbose=False,
    )

    with open(csv_path) as f:
        header = next(csv.reader(f))

    assert tuple(header) == tuple(SUMMARY_V2_FIELDNAMES), (
        f"Header mismatch.\nGot: {header}\nExpected: {list(SUMMARY_V2_FIELDNAMES)}"
    )
    assert len(SUMMARY_FIELDNAMES_V1) == 25, "v1 schema changed unexpectedly"
    assert len(V2_NEW_FIELDS) == 26, (
        f"V2_NEW_FIELDS count is {len(V2_NEW_FIELDS)}, expected 26"
    )
    assert len(SUMMARY_V2_FIELDNAMES) == 51
    # Multistart columns must be ABSENT per the unification target.
    assert "multistart_K" not in SUMMARY_V2_FIELDNAMES
    assert "multistart_init" not in SUMMARY_V2_FIELDNAMES
    # eff_index column must be present per decision 7-F.
    assert "eff_index" in SUMMARY_V2_FIELDNAMES
    # Seminorm columns must be present.
    assert "H1_semi_abs" in SUMMARY_V2_FIELDNAMES
    assert "H1_semi_rel" in SUMMARY_V2_FIELDNAMES


# -----------------------------------------------------------------------------
# 2. Bool columns
# -----------------------------------------------------------------------------


def test_v2_bool_columns_are_bool(tmp_path, trained_params_small):
    _, _, test_betas = generate_splits(0)
    csv_path = tmp_path / "s.csv"
    evaluate_test_split_v2(
        trained_params_small, p=3, seed=0,
        test_betas=test_betas, eval_levels=(8,),
        incremental_csv_path=csv_path,
        clear_caches_between_levels=False, verbose=False,
    )
    bool_cols = ["accepted_raw", "accepted_corrected",
                 "residual_gate_pass", "shape_gate_pass",
                 "certified", "corrector_success", "hmin_active"]
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for c in bool_cols:
                v = row[c]
                assert v in ("0", "1", "True", "False"), (
                    f"Column {c} has non-bool value {v!r}"
                )


# -----------------------------------------------------------------------------
# 3. eta decomposition
# -----------------------------------------------------------------------------


def test_v2_eta_decomposition_sums(tmp_path, trained_params_small):
    """eta² ≈ eta_volume² + eta_neumann² (bit-exact since both come from
    the same loss_terms call)."""
    _, _, test_betas = generate_splits(0)
    csv_path = tmp_path / "s.csv"
    evaluate_test_split_v2(
        trained_params_small, p=3, seed=0,
        test_betas=test_betas, eval_levels=(8,),
        incremental_csv_path=csv_path,
        clear_caches_between_levels=False, verbose=False,
    )
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        eta = float(row["eta"])
        vol = float(row["eta_volume"])
        neu = float(row["eta_neumann"])
        decomp = math.sqrt(vol * vol + neu * neu)
        assert abs(decomp - eta) < 1e-10 + 1e-10 * max(eta, decomp), (
            f"Decomp mismatch: eta={eta:.6e}  sqrt(vol²+neum²)={decomp:.6e}  "
            f"(p={row['p']}, N={row['N']}, beta={row['beta']}, method={row['method']})"
        )


# -----------------------------------------------------------------------------
# 4. Saturation fractions in [0, 1]
# -----------------------------------------------------------------------------


def test_v2_logit_saturation_in_unit_interval(tmp_path, trained_params_small):
    _, _, test_betas = generate_splits(0)
    csv_path = tmp_path / "s.csv"
    evaluate_test_split_v2(
        trained_params_small, p=3, seed=0,
        test_betas=test_betas, eval_levels=(8,),
        incremental_csv_path=csv_path,
        clear_caches_between_levels=False, verbose=False,
    )
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        L = float(row["logit_saturation_lower"])
        U = float(row["logit_saturation_upper"])
        assert 0.0 <= L <= 1.0, f"saturation_lower={L} out of [0,1]"
        assert 0.0 <= U <= 1.0, f"saturation_upper={U} out of [0,1]"
        assert L + U <= 1.0 + 1e-12, f"L+U={L+U} > 1"


# -----------------------------------------------------------------------------
# 5. LBFGS-B counter invariants
# -----------------------------------------------------------------------------


def test_v2_lbfgsb_counters_consistent(tmp_path, trained_params_small):
    """lbfgsb_nit ≤ lbfgsb_nfev (scipy invariant)."""
    _, _, test_betas = generate_splits(0)
    csv_path = tmp_path / "s.csv"
    evaluate_test_split_v2(
        trained_params_small, p=3, seed=0,
        test_betas=test_betas, eval_levels=(8,),
        incremental_csv_path=csv_path,
        clear_caches_between_levels=False, verbose=False,
    )
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    n_corr = 0
    for row in rows:
        if row["method"] != "positional_corrected":
            continue
        n_corr += 1
        nit = int(row["lbfgsb_nit"])
        nfev = int(row["lbfgsb_nfev"])
        ngev = int(row["lbfgsb_ngev"])
        assert nit >= 0
        assert nfev >= nit, f"nfev={nfev} < nit={nit}"
        assert ngev >= 0
    assert n_corr > 0, "No positional_corrected rows found"


# -----------------------------------------------------------------------------
# 6. history_loss parseable
# -----------------------------------------------------------------------------


def test_v2_history_loss_parseable(tmp_path, trained_params_small):
    _, _, test_betas = generate_splits(0)
    csv_path = tmp_path / "s.csv"
    evaluate_test_split_v2(
        trained_params_small, p=3, seed=0,
        test_betas=test_betas, eval_levels=(8,),
        incremental_csv_path=csv_path,
        clear_caches_between_levels=False, verbose=False,
        record_history=True,
    )
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    found_nonempty = False
    for row in rows:
        hl = row["history_loss"]
        if not hl:
            continue
        parsed = [float(x) for x in hl.split(",")]
        assert all(isinstance(v, float) for v in parsed)
        if len(parsed) > 0:
            found_nonempty = True
            assert len(parsed) <= 200, "history_loss exceeds 200-entry cap"
    assert found_nonempty, "No corrected sample produced a non-empty history_loss"


# -----------------------------------------------------------------------------
# 7. runtime positive
# -----------------------------------------------------------------------------


def test_v2_runtime_positive(tmp_path, trained_params_small):
    _, _, test_betas = generate_splits(0)
    csv_path = tmp_path / "s.csv"
    evaluate_test_split_v2(
        trained_params_small, p=3, seed=0,
        test_betas=test_betas, eval_levels=(8,),
        incremental_csv_path=csv_path,
        clear_caches_between_levels=False, verbose=False,
    )
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        r = float(row["runtime"])
        assert r > 0.0, f"runtime={r} not positive in row {row['method']}"


# -----------------------------------------------------------------------------
# 8. eff_index column (decision 7-F)
# -----------------------------------------------------------------------------


def test_v2_eff_index_uses_h1_seminorm(tmp_path, trained_params_small):
    """``eff_index = eta / H1_semi_abs``; must be finite > 0 wherever
    both inputs are finite > 0, and exactly equal to the recomputed
    value. (The residual estimator controls the energy norm — the H¹
    seminorm — so the seminorm is the correct denominator. The legacy
    v1 ``I_eff`` column is aliased to the same value.)"""
    _, _, test_betas = generate_splits(0)
    csv_path = tmp_path / "s.csv"
    evaluate_test_split_v2(
        trained_params_small, p=3, seed=0,
        test_betas=test_betas, eval_levels=(8,),
        incremental_csv_path=csv_path,
        clear_caches_between_levels=False, verbose=False,
    )
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    n_checked = 0
    for row in rows:
        eta = float(row["eta"])
        h1_semi = float(row["H1_semi_abs"])
        if not (eta > 0.0 and h1_semi > 0.0):
            continue
        n_checked += 1
        expected = eta / h1_semi
        got = float(row["eff_index"])
        legacy = float(row["I_eff"])
        assert math.isfinite(got), f"eff_index not finite: {got}"
        assert got > 0.0, f"eff_index not positive: {got}"
        rel = abs(got - expected) / max(expected, 1e-30)
        assert rel < 1e-12, (
            f"eff_index mismatch: stored={got!r}  recomputed={expected!r}  "
            f"rel={rel:.3e}"
        )
        # Legacy column must be aliased to the same value.
        assert legacy == got, (
            f"I_eff legacy column {legacy!r} != eff_index {got!r}; "
            "they must be aliased to the same corrected formula."
        )
    assert n_checked > 0, "no rows had positive eta and H1_semi_abs to check"


def test_v2_h1_semi_le_h1_full(tmp_path, trained_params_small):
    """The H¹ seminorm is a component of the full H¹ norm
    (full² = L² + semi²), so the absolute seminorm error must never
    exceed the absolute full-H¹ error. Verified on every smoke row."""
    _, _, test_betas = generate_splits(0)
    csv_path = tmp_path / "s.csv"
    evaluate_test_split_v2(
        trained_params_small, p=3, seed=0,
        test_betas=test_betas, eval_levels=(8,),
        incremental_csv_path=csv_path,
        clear_caches_between_levels=False, verbose=False,
    )
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    for row in rows:
        h1_full = float(row["H1_abs"])
        h1_semi = float(row["H1_semi_abs"])
        assert h1_semi >= 0.0
        # Allow a tiny ulp-level slack for float reconstruction (the full
        # norm is sqrt(l2² + semi²); rounding can make semi marginally
        # larger than full at noise floor).
        assert h1_semi <= h1_full * (1.0 + 1e-12), (
            f"H1_semi_abs ({h1_semi}) > H1_abs ({h1_full})  "
            f"in row p={row['p']} N={row['N']} beta={row['beta']} "
            f"method={row['method']}"
        )


def test_v2_no_column_uses_broken_eff_formulas(trained_params_small):
    """Source-text guard: the broken formulas ``eta / h1_abs``
    (full-norm denominator) and ``eta / sqrt(h1_abs)`` (sqrt of an
    already-rooted quantity) must NOT appear anywhere in evaluation_v2
    as an effectivity-index assignment."""
    import inspect
    from src.parametric import evaluation_v2 as ev2

    src = inspect.getsource(ev2)
    # The legacy v1 formula `eta / h1_abs` is forbidden as an
    # effectivity assignment.
    assert "eta / h1_abs" not in src, (
        "Legacy broken formula `eta / h1_abs` still present in "
        "evaluation_v2 — the full H¹ norm is the wrong denominator."
    )
    # The spurious-sqrt formula is forbidden.
    assert "eta / math.sqrt(h1_abs)" not in src, (
        "Spurious-sqrt formula `eta / math.sqrt(h1_abs)` still "
        "present in evaluation_v2 — h1_abs is already the H¹ norm, "
        "not its square."
    )


# -----------------------------------------------------------------------------
# 9. corrector_budget override caps lbfgsb_nit
# -----------------------------------------------------------------------------


def test_eval_v2_corrector_budget_override(trained_params_small):
    """When `corrector_budget` is passed through, the underlying L-BFGS-B
    respects that cap. The default-budget (=80) path produces a value equal
    to the explicit budget=80 case (bit-exact backward compat)."""
    from src.parametric.evaluation_v2 import evaluate_one_v2

    rows_default = evaluate_one_v2(
        trained_params_small, p=2, N=4, beta=1.55, seed=0,
        record_history=False,
    )
    rows_explicit80 = evaluate_one_v2(
        trained_params_small, p=2, N=4, beta=1.55, seed=0,
        corrector_budget=80, record_history=False,
    )
    corr_default = [r for r in rows_default
                     if r["method"] == "positional_corrected"][0]
    corr_explicit = [r for r in rows_explicit80
                      if r["method"] == "positional_corrected"][0]
    assert int(corr_default["lbfgsb_nit"]) == int(corr_explicit["lbfgsb_nit"]), (
        corr_default["lbfgsb_nit"], corr_explicit["lbfgsb_nit"])
    for col in ("H1_rel", "eta", "eta_volume"):
        if col in corr_default and col in corr_explicit:
            assert float(corr_default[col]) == float(corr_explicit[col]), col

    # Explicit small budgets must cap lbfgsb_nit.
    for budget in (2, 5, 10):
        rows = evaluate_one_v2(
            trained_params_small, p=2, N=4, beta=1.55, seed=0,
            corrector_budget=budget, record_history=False,
        )
        corr = [r for r in rows
                 if r["method"] == "positional_corrected"][0]
        nit = int(corr["lbfgsb_nit"])
        assert 0 <= nit <= budget, (
            f"budget={budget}: lbfgsb_nit={nit} violated the cap.")


# -----------------------------------------------------------------------------
# 10. tau_global plumbing (decision 7-B)
# -----------------------------------------------------------------------------


def test_v2_tau_global_recorded(tmp_path, trained_params_small):
    """When ``tau_global`` is passed, the corrected row records ``tau ==
    tau_global`` and the residual gate is evaluated against it."""
    _, _, test_betas = generate_splits(0)
    csv_path = tmp_path / "s.csv"
    tau_test = 12.34
    evaluate_test_split_v2(
        trained_params_small, p=3, seed=0,
        test_betas=test_betas, eval_levels=(8,),
        tau_global=tau_test,
        incremental_csv_path=csv_path,
        clear_caches_between_levels=False, verbose=False,
    )
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    n_corr = 0
    for row in rows:
        if row["method"] != "positional_corrected":
            continue
        n_corr += 1
        assert row["tau"] != "", "tau column must not be blank when tau_global is set"
        assert abs(float(row["tau"]) - tau_test) < 1e-12
        # residual_gate_pass must be 0/1 (depending on whether eta ≤ tau)
        assert row["residual_gate_pass"] in ("0", "1")
    assert n_corr > 0


# -----------------------------------------------------------------------------
# 11. calibrate_tau_global returns a sensible value (decision 7-B)
# -----------------------------------------------------------------------------


def test_calibrate_tau_global_finite(trained_params_small):
    """On a smoke training, the calibrated tau must be finite > 0."""
    _, val_betas, _ = generate_splits(0)
    tau = calibrate_tau_global(
        trained_params_small, p=3,
        calibration_betas=val_betas, N=8,
    )
    assert math.isfinite(tau), f"calibrated tau is not finite: {tau!r}"
    assert tau > 0.0, f"calibrated tau is not positive: {tau!r}"
    # Sanity bound: τ = κ · median(η_pos / η_unif). Both ratios are typically
    # in (0, 10] on small smoke runs, and κ = 1.5; so τ should be in (0, 100).
    assert tau < 100.0, f"calibrated tau {tau!r} looks pathological"
