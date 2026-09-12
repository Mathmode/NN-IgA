"""Train the parametric ``singular`` experiment for one (p, seed).

Trains the positional density network with coarse-to-fine continuation, saves
per-level and final checkpoints, evaluates on the test split, and writes the
summary/history/metadata files under ``data_results/singular/p<p>/``.

Usage:
    python 1D/scripts/train_singular.py --protocol val --p 3 --seed 0
    python 1D/scripts/train_singular.py --p 3 --seed 0 --smoke   # ~20 min check
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Make <repo>/ (common package) and <repo>/1D (src package) importable when the
# script is run standalone.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = ROOT.parent
for _p in (str(PROJECT), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _setup_environment(args) -> None:
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    if args.device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    elif args.device == "gpu":
        os.environ.setdefault("JAX_PLATFORMS", "gpu,cpu")


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    import csv as _csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


HISTORY_FIELDS = [
    "p", "seed", "level_N", "epoch",
    "train_loss", "val_loss", "h1_rel_val_median", "lr", "time_sec",
    "optimizer_restarts", "is_best",
]


SMOKE_OVERRIDES = {
    "CONTINUATION_LEVELS": (2, 4, 8, 16, 32),
    "EVAL_LEVELS": (2, 4, 8, 16, 32),
    "TRAIN": {
        "train_size": 32,
        "val_size": 8,
        "test_size": 16,
        "max_epochs": 30,
        "patience": 10,
        "max_optimizer_restarts": 0,
        "batch_size": 16,
    },
    "VAL": {
        "p1_total": 48,
        "max_epochs": 30,
        "min_epochs": 6,
        "patience": 6,
    },
}


def _apply_smoke_overrides() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Shrink the config for a fast end-to-end run.

    Dicts (TRAIN/VAL) are mutated in place, which propagates to their consumers;
    the level tuples cannot be rebound for other modules, so they are returned
    and passed explicitly to continuation/evaluation.
    """
    from src import config as cfg
    for k, v in SMOKE_OVERRIDES["TRAIN"].items():
        cfg.TRAIN[k] = v  # type: ignore[index]
    for k, v in SMOKE_OVERRIDES["VAL"].items():
        cfg.VAL[k] = v  # type: ignore[index]
    return SMOKE_OVERRIDES["CONTINUATION_LEVELS"], SMOKE_OVERRIDES["EVAL_LEVELS"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parametric singular main entry point.")
    parser.add_argument("--p", type=int, required=True, choices=[2, 3])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output root (default: data_results/singular/p<p>/; the "
             "--init-mode uniform variant writes to .../singular_theta0/).",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "gpu"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init-mode", choices=["warmstart", "uniform"], default="warmstart",
        help="'warmstart' (default): inherit weights across continuation levels. "
             "'uniform': re-initialise at every level (ablation).",
    )
    parser.add_argument(
        "--epochs-per-level", type=int, default=None,
        help="Override the per-level epoch cap (default: config value).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Reduced levels/splits for a fast end-to-end check (~20 min).",
    )
    parser.add_argument(
        "--protocol", choices=["fixed", "val"], default="fixed",
        help="'val' (paper): fixed 70/15/15 split shared across seeds, validation "
             "early stopping, best-val checkpoint. 'fixed' (default): legacy recipe.",
    )
    parser.add_argument(
        "--uniform-init", action="store_true",
        help="Ablation: zero-init the network and train only at the finest level "
             "(no continuation).",
    )
    parser.add_argument(
        "--evaluation", choices=("v1", "v2"), default="v2",
        help="Evaluation CSV schema; 'v2' (default) is the enriched one.",
    )
    parser.add_argument(
        "--corrector-max-iter", type=int, default=0,
        help="L-BFGS-B corrector budget for a positional_corrected column. "
             "Default 0: corrector-free (uniform + positional rows only).",
    )
    args = parser.parse_args()

    _setup_environment(args)

    # Import after env setup so JAX picks up x64 + device.
    import gc

    import jax

    from src.config import ANCHOR_N, CONTINUATION_LEVELS, EVAL_LEVELS, VIZ, SPLIT_SEED
    from src.parametric.continuation import (
        save_checkpoint,
        train_with_continuation,
    )
    from src.parametric.evaluation import (
        evaluate_test_split,
        evaluate_viz_set,
    )
    from src.parametric.evaluation_v2 import (
        SUMMARY_V2_FIELDNAMES,
        calibrate_tau_global,
        evaluate_test_split_v2,
    )
    from src.lhs_sampling import generate_splits, generate_splits_val

    # Apply smoke overrides after imports so config mutations are visible.
    if args.smoke:
        continuation_levels, eval_levels = _apply_smoke_overrides()
        print("[smoke] CONTINUATION_LEVELS =", continuation_levels, flush=True)
        print("[smoke] EVAL_LEVELS         =", eval_levels, flush=True)
    else:
        continuation_levels = CONTINUATION_LEVELS
        eval_levels = EVAL_LEVELS

    PROBLEM = "singular"
    if args.output_dir is not None:
        out_root = Path(args.output_dir).resolve()
    else:
        base = PROJECT / "data_results" / PROBLEM / f"p{int(args.p)}"
        out_root = (base / f"{PROBLEM}_theta0"
                    if args.init_mode == "uniform" else base)
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_root / "checkpoints" / f"p{args.p}_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"=== main_parametric_1D_singular  p={args.p}  seed={args.seed}"
          f"  smoke={bool(args.smoke)} ===", flush=True)
    print(f"Working dir: {ROOT}", flush=True)
    print(f"Output dir : {out_root}", flush=True)

    # 1. Splits: 'val' = fixed SPLIT_SEED 70/15/15 shared across seeds;
    #    'fixed' = legacy per-seed LHS draws.
    if args.protocol == "val":
        train_betas, val_betas, test_betas = generate_splits_val()
    else:
        train_betas, val_betas, test_betas = generate_splits(int(args.seed))
    print(
        f"[t={time.perf_counter() - t0:.1f}s] Splits ({args.protocol}): |train|={train_betas.size}"
        f"  |val|={val_betas.size}  |test|={test_betas.size}"
        f"  range=[{train_betas.min():.3f}, {train_betas.max():.3f}]",
        flush=True,
    )

    # 2. Coarse-to-fine continuation training.
    cont = train_with_continuation(
        p=int(args.p),
        seed=int(args.seed),
        train_betas=train_betas,
        val_betas=val_betas,
        levels=continuation_levels,
        checkpoint_dir=ckpt_dir,
        resume=bool(args.resume),
        uniform_init=bool(args.uniform_init),
        init_mode=str(args.init_mode),
        epochs_per_level=args.epochs_per_level,
        protocol=str(args.protocol),
    )
    print(
        f"[t={time.perf_counter() - t0:.1f}s] Training elapsed: {cont.elapsed_sec:.1f}s"
        f"; levels={cont.levels_done}",
        flush=True,
    )

    # 3. Final checkpoint.
    save_checkpoint(cont.params_final, ckpt_dir / "checkpoint_final.npz")
    print(
        f"[t={time.perf_counter() - t0:.1f}s] saved checkpoint_final.npz",
        flush=True,
    )

    # Free the training JIT cache before evaluation re-traces with new shapes.
    jax.clear_caches()
    gc.collect()
    print(
        f"[t={time.perf_counter() - t0:.1f}s] cleared post-training caches",
        flush=True,
    )

    run_corrector = int(args.corrector_max_iter) > 0
    corrector_budget = int(args.corrector_max_iter) if run_corrector else None
    print(
        f"[t={time.perf_counter() - t0:.1f}s] corrector: "
        f"{'ON (budget=%d)' % corrector_budget if run_corrector else 'OFF (corrector-free)'}",
        flush=True,
    )

    # 4a. Residual-gate threshold tau (feeds the corrector only).
    tau_global = None
    if args.evaluation == "v2" and run_corrector:
        calib_N = int(continuation_levels[-1])
        print(
            f"[t={time.perf_counter() - t0:.1f}s] calibrate_tau_global N={calib_N} "
            f"on {val_betas.size} val betas",
            flush=True,
        )
        tau_global = calibrate_tau_global(
            cont.params_final,
            p=int(args.p),
            calibration_betas=val_betas,
            N=calib_N,
        )
        print(
            f"[t={time.perf_counter() - t0:.1f}s] tau_global = {tau_global!r}",
            flush=True,
        )

    # 4b. Test-split evaluation (incremental CSV write).
    summary_path = out_root / f"summary_p{args.p}_seed{args.seed}.csv"
    print(
        f"[t={time.perf_counter() - t0:.1f}s] evaluate_test_split ({args.evaluation}) start "
        f"(incremental writes to {summary_path.name})",
        flush=True,
    )
    if args.evaluation == "v2":
        summary_rows = evaluate_test_split_v2(
            cont.params_final,
            p=int(args.p),
            seed=int(args.seed),
            test_betas=test_betas,
            eval_levels=eval_levels,
            tau_global=tau_global,
            incremental_csv_path=summary_path,
            clear_caches_between_levels=True,
            verbose=True,
            run_corrector=run_corrector,
            corrector_budget=corrector_budget,
        )
    else:
        summary_rows = evaluate_test_split(
            cont.params_final,
            p=int(args.p),
            seed=int(args.seed),
            test_betas=test_betas,
            eval_levels=eval_levels,
            incremental_csv_path=summary_path,
            clear_caches_between_levels=True,
            verbose=True,
            run_corrector=run_corrector,
        )
    print(
        f"[t={time.perf_counter() - t0:.1f}s] wrote {summary_path}"
        f"  ({len(summary_rows)} rows)",
        flush=True,
    )

    # 5. Viz set (viz seed only).
    if int(args.seed) == int(VIZ["viz_seed"]):  # type: ignore[index]
        print(
            f"[t={time.perf_counter() - t0:.1f}s] evaluate_viz_set start",
            flush=True,
        )
        viz_paths = evaluate_viz_set(
            cont.params_final,
            p=int(args.p),
            output_dir=out_root,
            eval_levels=eval_levels,
            clear_caches_between_levels=True,
            verbose=True,
            run_corrector=run_corrector,
        )
        for name, path in viz_paths.items():
            print(f"  wrote {path}", flush=True)

    # 6. Per-epoch training history.
    history_rows = []
    for row in cont.history:
        history_rows.append({"p": int(args.p), "seed": int(args.seed), **row})
    history_path = out_root / f"history_p{args.p}_seed{args.seed}.csv"
    _write_csv(history_path, history_rows, HISTORY_FIELDS)
    print(
        f"[t={time.perf_counter() - t0:.1f}s] wrote {history_path}"
        f"  ({len(history_rows)} rows)",
        flush=True,
    )

    # 6b. Unified history schema (val protocol), shared with the 2D experiments.
    if args.protocol == "val":
        uni_dir = out_root / "history"
        uni_dir.mkdir(parents=True, exist_ok=True)
        uni_path = uni_dir / f"history_p{args.p}_seed{args.seed}.csv"
        with uni_path.open("w") as f:
            f.write("iteration,train_loss,val_eta2\n")
            for i, row in enumerate(cont.history, start=1):
                f.write(f"{i},{float(row['train_loss']):.6e},{float(row['val_loss']):.6e}\n")
        print(
            f"[t={time.perf_counter() - t0:.1f}s] wrote {uni_path}"
            f"  ({len(cont.history)} epochs)",
            flush=True,
        )

    # 7. Run metadata.
    elapsed = time.perf_counter() - t0
    meta = {
        "p": int(args.p),
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "protocol": str(args.protocol),
        "split_seed": (int(SPLIT_SEED) if args.protocol == "val" else int(args.seed)),
        "uniform_init": bool(args.uniform_init),
        "evaluation": str(args.evaluation),
        "corrector_max_iter": int(args.corrector_max_iter),
        "run_corrector": bool(run_corrector),
        "tau_global": (None if tau_global is None else float(tau_global)),
        "git_hash": _git_hash(),
        "platform": platform.platform(),
        "python": sys.version,
        "device": args.device or "auto",
        "elapsed_sec": float(elapsed),
        "training_elapsed_sec": float(cont.elapsed_sec),
        "levels_done": list(cont.levels_done),
        "eval_levels": list(eval_levels),
        "continuation_levels": list(continuation_levels),
    }
    (out_root / f"metadata_p{args.p}_seed{args.seed}.json").write_text(
        json.dumps(meta, indent=2)
    )
    print(f"=== DONE  total={elapsed:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
