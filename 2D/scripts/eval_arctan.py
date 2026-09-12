"""Re-evaluate a saved ``arctan`` checkpoint on the test set.

Writes the per-seed ``summary_p2_seed{seed}.csv`` plus an eval metadata JSON.
Use --protocol val for the paper's held-out test subset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TWO_D_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _TWO_D_ROOT.parent
for p in (str(_PROJECT_ROOT), str(_TWO_D_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import json

import numpy as np

from src.config import ARCTAN, SPLIT_SEED, TRAIN_VAL
from src.parametric.continuation import load_checkpoint
from src.parametric.evaluation_v2 import evaluate_arctan
from common.parameter_sampling import build_arctan_grid
from common.csv_utils import summary_is_complete


def _assert_disjoint_from_train(grid, test_sigmas):
    """Guard: the evaluated tuples must have zero overlap with train/val (the
    legacy per-seed 70/30 split overlapped the training set -- audit D3)."""
    def _keyset(a):
        return {tuple(np.round(r, 10)) for r in np.asarray(a)}
    test_k = _keyset(test_sigmas)
    train_k = _keyset(grid["train"])
    val_k = _keyset(grid.get("val", np.empty((0, test_sigmas.shape[1]))))
    n_in_train = len(test_k & train_k)
    n_in_val = len(test_k & val_k)
    assert n_in_train == 0, (
        f"{n_in_train}/{len(test_k)} evaluated tuples are in the TRAINING set "
        f"(protocol contamination -- see audit D3)."
    )
    assert n_in_val == 0, f"{n_in_val}/{len(test_k)} evaluated tuples are in the VAL set."
    return n_in_train, n_in_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--N-levels", type=int, nargs="+", default=None)
    ap.add_argument("--n-test-sigmas", type=int, default=None)
    ap.add_argument("--corrector-max-iter", type=int, default=30)
    ap.add_argument("--protocol", choices=["fixed", "val"], default="fixed",
                    help="'val' (paper): the fixed 70/15/15 held-out test set, identical "
                         "across seeds. 'fixed' (default): legacy per-seed 70/30 split "
                         "(overlaps the training set -- audit D3).")
    ap.add_argument("--force", action="store_true",
                    help="Re-evaluate even if a complete summary CSV for this "
                         "seed already exists (default: skip complete seeds).")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_arctan] seed={args.seed} smoke={args.smoke} protocol={args.protocol} -> {out_dir}")

    # 'val': fixed SPLIT_SEED 70/15/15 shared across seeds; 'fixed': legacy
    # per-seed 70/30 (overlaps training -- audit D3).
    val_protocol = (args.protocol == "val")
    grid = build_arctan_grid(
        n_alpha=int(ARCTAN["n_alpha"]),
        n_s1=int(ARCTAN["n_s1"]),
        n_s2=int(ARCTAN["n_s2"]),
        alpha_min=float(ARCTAN["alpha_min"]),
        alpha_max=float(ARCTAN["alpha_max"]),
        s_min=float(ARCTAN["s_min"]),
        s_max=float(ARCTAN["s_max"]),
        train_frac=(float(TRAIN_VAL["train_frac"]) if val_protocol else float(ARCTAN["train_frac"])),
        seed=(int(SPLIT_SEED) if val_protocol else int(args.seed)),
        val_frac=(float(TRAIN_VAL["val_frac"]) if val_protocol else None),
    )
    test_sigmas = grid["test"]
    if val_protocol:
        n_in_train, n_in_val = _assert_disjoint_from_train(grid, test_sigmas)
        print(f"  protocol=val: {test_sigmas.shape[0]} held-out test tuples "
              f"(overlap with train={n_in_train}, val={n_in_val}) -- seed-independent")
    if args.smoke:
        n_test = int(args.n_test_sigmas or 4)
        rng = np.random.default_rng(args.seed)
        test_sigmas = test_sigmas[rng.choice(test_sigmas.shape[0], n_test, replace=False)]
    elif args.n_test_sigmas is not None:
        test_sigmas = test_sigmas[: int(args.n_test_sigmas)]

    print(f"  test sigmas: {test_sigmas.shape}")

    levels = tuple(args.N_levels) if args.N_levels else (
        (4,) if args.smoke else tuple(ARCTAN["levels"])
    )
    print(f"  levels: {levels}")

    summary_path = out_dir / f"summary_p2_seed{int(args.seed)}.csv"
    if not args.force and summary_is_complete(summary_path, levels):
        print(f"[skip] {summary_path} already complete for N={list(levels)}; "
              f"pass --force to re-evaluate")
        return 0

    params = load_checkpoint(Path(args.checkpoint))

    out_csv = evaluate_arctan(
        params, test_sigmas,
        p=int(args.p),
        levels=levels,
        seed=int(args.seed),
        output_dir=out_dir,
        q_K=int(args.p) + 1,
        q_F=40 if args.smoke else int(ARCTAN["quad_forcing"]),
        q_metric=40 if args.smoke else int(ARCTAN["quad_metric"]),
        corrector_max_iter=int(args.corrector_max_iter),
        checkpoint_path=str(args.checkpoint),
    )
    # Provenance sidecar: records which protocol produced this CSV.
    meta_path = out_dir / f"metadata_eval_arctan_p{int(args.p)}_seed{int(args.seed)}.json"
    with open(meta_path, "w") as fh:
        json.dump({
            "experiment": "arctan",
            "mode": "eval_arctan",
            "protocol": str(args.protocol),
            "split_seed": int(SPLIT_SEED) if val_protocol else int(args.seed),
            "n_test_tuples": int(test_sigmas.shape[0]),
            "test_set_seed_independent": bool(val_protocol),
            "p": int(args.p),
            "seed": int(args.seed),
            "levels": list(levels),
            "checkpoint": str(args.checkpoint),
            "remediation_task": "T4 (audit D3)",
        }, fh, indent=2)
    print(f"=== eval_arctan DONE -> {out_csv} (meta: {meta_path.name}) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
