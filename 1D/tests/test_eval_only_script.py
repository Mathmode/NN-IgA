"""End-to-end test: standalone eval_singular reproduces the in-pass eval.

``train_singular.py`` does TRAIN + held-out test EVAL in a single pass;
``eval_singular.py`` re-runs that same evaluation from the saved checkpoint.
Both are corrector-free by default and share the ``evaluate_test_split_v2`` code
path, so with matching split/level settings the two summary CSVs must be
numerically identical (up to wall-clock columns).

Marked ``slow`` (it spawns two training+eval subprocesses). To run:
    pytest tests/test_eval_only_script.py -v -m slow --override-ini="addopts="
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Wall-clock / trajectory columns that legitimately differ run-to-run and are
# excluded from the numeric comparison.
_VOLATILE_COLS = {"runtime", "history_loss"}


@pytest.mark.slow
def test_standalone_eval_matches_in_pass(tmp_path):
    """train_singular (--smoke) then eval_singular on the same checkpoint with
    matching env overrides; the two summary CSVs must agree numerically."""
    pd = pytest.importorskip("pandas")

    out_main = tmp_path / "main_out"
    out_eval = tmp_path / "eval_out"

    base_env = os.environ.copy()
    base_env.setdefault("JAX_ENABLE_X64", "true")
    base_env.setdefault("JAX_PLATFORM_NAME", "cpu")
    base_env.setdefault("PYTHONUNBUFFERED", "1")

    # 1. Train + in-pass eval via train_singular --smoke (corrector-free default).
    subprocess.check_call(
        [
            sys.executable, "-u",
            str(ROOT / "scripts" / "train_singular.py"),
            "--p", "2", "--seed", "0",
            "--output-dir", str(out_main),
            "--device", "cpu",
            "--smoke",
        ],
        cwd=str(ROOT),
        env=base_env,
    )
    summary_main = out_main / "summary_p2_seed0.csv"
    assert summary_main.exists(), "train_singular did not produce a summary CSV"

    # 2. Standalone re-eval from the saved checkpoint. eval_singular has no
    #    --smoke flag (no notion of training size), so the smoke-equivalent
    #    levels/splits are forwarded via the P1D_* env vars it reads on import.
    eval_env = base_env.copy()
    eval_env["P1D_EVAL_LEVELS"] = "2,4,8,16,32"
    eval_env["P1D_TEST_SIZE"] = "16"
    eval_env["P1D_VAL_SIZE"] = "8"
    eval_env["P1D_TRAIN_SIZE"] = "32"
    subprocess.check_call(
        [
            sys.executable, "-u",
            str(ROOT / "scripts" / "eval_singular.py"),
            "--p", "2", "--seed", "0",
            "--checkpoint-dir", str(out_main / "checkpoints"),
            "--output-dir", str(out_eval),
        ],
        cwd=str(ROOT),
        env=eval_env,
    )
    summary_eval = out_eval / "summary_p2_seed0.csv"
    assert summary_eval.exists(), "eval_singular did not produce a summary CSV"

    # 3. Same code path + same final checkpoint -> numerically identical summary.
    #    Drop wall-clock columns, sort by the identifying columns, and compare
    #    the remaining columns with a tight tolerance.
    df_main = pd.read_csv(summary_main)
    df_eval = pd.read_csv(summary_eval)
    assert df_main.shape == df_eval.shape, (
        f"shape mismatch: main {df_main.shape} vs eval {df_eval.shape}"
    )
    ignore = {c for c in df_main.columns if c in _VOLATILE_COLS or "time" in c.lower()}
    cols = [c for c in df_main.columns if c not in ignore]
    sort_keys = [c for c in ("p", "seed", "N", "beta", "method") if c in cols]
    df_main = df_main[cols].sort_values(sort_keys).reset_index(drop=True)
    df_eval = df_eval[cols].sort_values(sort_keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(df_main, df_eval, atol=1e-9, rtol=1e-9)
