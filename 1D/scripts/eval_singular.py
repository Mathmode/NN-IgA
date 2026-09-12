#!/usr/bin/env python
"""Re-evaluate a saved ``singular`` checkpoint on the test split.

Writes the per-seed ``summary_p{p}_seed{seed}.csv`` (v2 schema) and an eval
metadata JSON. Corrector-free by default, matching the shipped tables.

Usage:
  python 1D/scripts/eval_singular.py --protocol val --p 3 --seed 0 \\
      --checkpoint-dir data_results/singular/p3/checkpoints \\
      --output-dir     data_results/singular/p3/eval_val
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Env setup before importing JAX.
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# Make <repo>/ (common package) and <repo>/1D (src package) importable.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = ROOT.parent
for _p in (str(PROJECT), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402
import numpy as np  # noqa: E402

from src import config  # noqa: E402
from src.parametric.continuation import load_checkpoint  # noqa: E402
from src.parametric.evaluation_v2 import (  # noqa: E402
    calibrate_tau_global,
    evaluate_test_split_v2,
)
from src.lhs_sampling import generate_splits, generate_splits_val  # noqa: E402
from common.csv_utils import summary_is_complete  # noqa: E402


# Optional env-var overrides for tests / debugging.
_env_levels = os.environ.get("P1D_EVAL_LEVELS")
if _env_levels:
    config.EVAL_LEVELS = tuple(int(x) for x in _env_levels.split(","))
for _key in ("test_size", "val_size", "train_size"):
    _val = os.environ.get(f"P1D_{_key.upper()}")
    if _val is not None:
        config.TRAIN[_key] = int(_val)


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval-only v2 from saved checkpoint.")
    parser.add_argument("--p", type=int, required=True, choices=[2, 3])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--checkpoint-dir", type=Path, required=True,
        help="Parent dir; expects p{p}_seed{seed}/checkpoint_final.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--no-record-history", action="store_true",
        help="Skip recording loss trajectories (saves ~5%% time).",
    )
    parser.add_argument(
        "--corrector-budget", type=int, default=0,
        help="L-BFGS-B corrector budget for a positional_corrected column. "
             "Default 0: corrector-free (uniform + positional rows only).",
    )
    parser.add_argument(
        "--tau", type=float, default=None,
        help="Explicit residual-gate threshold (default: calibrated on the "
             "validation betas; only used when the corrector runs).",
    )
    parser.add_argument(
        "--no-tau", action="store_true",
        help="Disable the residual gate entirely.",
    )
    parser.add_argument(
        "--calib-level", type=int, default=None,
        help="Level for the tau calibration (default: deepest EVAL level).",
    )
    parser.add_argument(
        "--protocol", choices=["legacy", "val"], default="legacy",
        help="'val' (paper): the fixed 70/15/15 held-out test subset, identical "
             "across seeds. 'legacy' (default): per-seed LHS test split.",
    )
    parser.add_argument("--device", default="cpu",
                        help="Accepted but always CPU.")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-evaluate even if a complete summary CSV for this seed already "
             "exists (default: skip complete seeds).",
    )
    args = parser.parse_args()

    print(
        f"=== eval_singular  p={args.p}  seed={args.seed} ===",
        flush=True,
    )
    t0 = time.perf_counter()

    # Skip seeds whose summary already covers every EVAL level (unless --force),
    # so re-running the eval does not recompute finished seeds.
    summary_path = args.output_dir / f"summary_p{args.p}_seed{args.seed}.csv"
    if not args.force and summary_is_complete(summary_path, config.EVAL_LEVELS):
        print(f"[skip] {summary_path} already complete for N="
              f"{list(config.EVAL_LEVELS)}; pass --force to re-evaluate", flush=True)
        return

    # 1. Locate checkpoint
    ckpt_path = args.checkpoint_dir / f"p{args.p}_seed{args.seed}" / "checkpoint_final.npz"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    params = load_checkpoint(ckpt_path)
    print(f"[t={time.perf_counter() - t0:.1f}s] loaded {ckpt_path}", flush=True)

    # 2. Generate splits. --protocol val: the fixed 70/15/15 test subset shared
    # across seeds (matches how the checkpoints were trained). --protocol legacy:
    # the old per-seed LHS (audit D9 -- a different parameter set per seed).
    if args.protocol == "val":
        train_betas, val_betas, test_betas = generate_splits_val()
    else:
        train_betas, val_betas, test_betas = generate_splits(int(args.seed))
    print(
        f"[t={time.perf_counter() - t0:.1f}s] splits done (protocol={args.protocol})."
        f" |train|={train_betas.size}  |val|={val_betas.size}  |test|={test_betas.size}",
        flush=True,
    )
    if args.protocol == "val":
        # Disjointness guard (remediation T5): the evaluated betas must be exactly
        # the held-out test subset -- no overlap with train or val.
        lo = 1e-9 * (float(test_betas.max()) - float(test_betas.min()) + 1.0)
        tset = np.sort(test_betas)
        for name, other in (("train", train_betas), ("val", val_betas)):
            d = np.abs(test_betas[:, None] - np.asarray(other)[None, :])
            n_overlap = int(np.sum(np.any(d < lo, axis=1)))
            assert n_overlap == 0, (
                f"{n_overlap}/{test_betas.size} test betas overlap the {name} set "
                f"(protocol contamination -- see audit D9)."
            )

    # The residual gate only feeds the corrector; corrector-free runs skip the
    # tau calibration entirely.
    run_corrector = int(args.corrector_budget) > 0
    corrector_budget = int(args.corrector_budget) if run_corrector else None

    # 3a. Residual-gate threshold tau (corrector only).
    tau_global: float | None
    if not run_corrector:
        tau_global = None
        print(
            f"[t={time.perf_counter() - t0:.1f}s] corrector-free: residual gate "
            f"disabled (tau=None)",
            flush=True,
        )
    elif args.no_tau:
        tau_global = None
        print(
            f"[t={time.perf_counter() - t0:.1f}s] --no-tau: residual gate "
            f"disabled",
            flush=True,
        )
    elif args.tau is not None:
        tau_global = float(args.tau)
        if not math.isfinite(tau_global) or tau_global <= 0.0:
            raise ValueError(f"--tau must be finite > 0; got {tau_global!r}")
        print(
            f"[t={time.perf_counter() - t0:.1f}s] explicit --tau = {tau_global}",
            flush=True,
        )
    else:
        calib_N = int(args.calib_level) if args.calib_level is not None \
            else int(config.EVAL_LEVELS[-1])
        print(
            f"[t={time.perf_counter() - t0:.1f}s] calibrate_tau_global N={calib_N} "
            f"on {val_betas.size} val betas",
            flush=True,
        )
        tau_global = calibrate_tau_global(
            params, p=int(args.p),
            calibration_betas=val_betas, N=calib_N,
        )
        print(
            f"[t={time.perf_counter() - t0:.1f}s] tau_global = {tau_global!r}",
            flush=True,
        )

    # 3b. Test-split evaluation with incremental write.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"summary_p{args.p}_seed{args.seed}.csv"
    if summary_path.exists():
        summary_path.unlink()

    print(
        f"[t={time.perf_counter() - t0:.1f}s] evaluate_test_split_v2 start"
        f" (incremental writes to {summary_path.name})",
        flush=True,
    )
    rows = evaluate_test_split_v2(
        params,
        p=int(args.p),
        seed=int(args.seed),
        test_betas=test_betas,
        eval_levels=config.EVAL_LEVELS,
        tau_global=tau_global,
        incremental_csv_path=summary_path,
        clear_caches_between_levels=True,
        verbose=True,
        record_history=not bool(args.no_record_history),
        run_corrector=run_corrector,
        corrector_budget=corrector_budget,
    )
    print(
        f"[t={time.perf_counter() - t0:.1f}s] test split done."
        f" {len(rows)} rows total written to {summary_path.name}",
        flush=True,
    )

    # 4. Final cache flush + metadata stub.
    jax.clear_caches()
    gc.collect()

    elapsed = time.perf_counter() - t0
    meta_path = args.output_dir / f"metadata_eval_p{args.p}_seed{args.seed}.json"
    meta = {
        "p": int(args.p),
        "seed": int(args.seed),
        "mode": "eval_singular",
        "protocol": str(args.protocol),
        "n_test_betas": int(test_betas.size),
        "test_set_seed_independent": bool(args.protocol == "val"),
        "remediation_task": "T5 (audit D9)" if args.protocol == "val" else "legacy",
        "tau_global": (None if tau_global is None else float(tau_global)),
        "no_tau": bool(args.no_tau),
        "checkpoint": str(ckpt_path),
        "git_hash": _git_hash(),
        "platform": platform.platform(),
        "python": sys.version,
        "device": args.device,
        "elapsed_sec": float(elapsed),
        "eval_levels": list(config.EVAL_LEVELS),
        "n_summary_rows": int(len(rows)),
        "record_history": not bool(args.no_record_history),
        "corrector_budget_override": (int(args.corrector_budget)
                                       if args.corrector_budget is not None
                                       else None),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"=== eval_singular DONE  total={elapsed:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
