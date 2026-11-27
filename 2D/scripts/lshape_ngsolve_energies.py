"""Independent NGSolve hp reference energies F(u) = int_Omega u for Experiment 4.

The exact energy F(u) is the only reference quantity the energy-identity error
needs (see ``lshape_energy_errors.py``). For every test parameter pair in the
L-shape evaluation CSVs it solves the conforming three-subdomain hp-FEM problem
(geometric refinement toward the re-entrant corner) at two polynomial orders and
writes both energies.

Usage:
    python 2D/scripts/lshape_ngsolve_energies.py \
        --out references_v2/lshape_energy_ngsolve.csv --levels 16 --orders 12 14
    python 2D/scripts/lshape_ngsolve_energies.py --grid all --origin --outer-hpref 0.7 \
        --levels 20 --orders 16 18 --level-check --out references_v3/lshape_energy_ngsolve.csv
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "2D")]

import numpy as np
import pandas as pd

from src.nonparametric.lshape.ngsolve_energy import (
    INTERFACE,
    OUTER,
    boundary_names,
    build_lshape_mesh,
    solve_energy,
)


def grid_sigmas():
    from common.parameter_sampling import build_lshape_grid
    from src.config import LSHAPE, SPLIT_SEED, TRAIN_VAL

    g = build_lshape_grid(n_sigma1=int(LSHAPE["n_sigma1"]), n_sigma2=int(LSHAPE["n_sigma2"]),
                          exp_min=float(LSHAPE["sigma_exp_min"]), exp_max=float(LSHAPE["sigma_exp_max"]),
                          train_frac=float(TRAIN_VAL["train_frac"]), seed=int(SPLIT_SEED),
                          val_frac=float(TRAIN_VAL["val_frac"]))
    frames = [pd.DataFrame(np.asarray(g[k]), columns=["sigma1", "sigma2"]).assign(split=k)
              for k in ("train", "val", "test")]
    return pd.concat(frames).sort_values(["sigma1", "sigma2"]).reset_index(drop=True)


def test_sigmas(eval_glob):
    frames = [pd.read_csv(f, usecols=["sigma1", "sigma2"]) for f in sorted(REPO.glob(eval_glob))]
    if not frames:
        raise SystemExit(f"no evaluation CSVs match {eval_glob}")
    return pd.concat(frames).drop_duplicates().sort_values(["sigma1", "sigma2"]).reset_index(drop=True)


def setup_meshes(args):
    mesh = build_lshape_mesh(args.levels, args.factor, origin=args.origin, outer_hpref=args.outer_hpref)
    if set(boundary_names(mesh)) != {OUTER, INTERFACE}:
        raise ValueError(f"unexpected boundary names {boundary_names(mesh)}")
    coarse = None
    if args.level_check:
        coarse = build_lshape_mesh(args.levels - 1, args.factor, origin=args.origin, outer_hpref=args.outer_hpref)
    return mesh, coarse


def solve_row(rec, meshes, args):
    mesh, coarse = meshes
    s1, s2 = rec["sigma1"], rec["sigma2"]
    row = rec.copy()
    for q in args.orders:
        row[f"F_order{q}"], row[f"ndof_order{q}"] = solve_energy(mesh, s1, s2, q)
    lo, hi = (row[f"F_order{q}"] for q in (min(args.orders), max(args.orders)))
    if min(lo, hi) <= 0 or not np.isfinite([lo, hi]).all():
        raise ValueError(f"invalid energies for {rec}")
    row["rel_diff_orders"] = abs(hi - lo) / hi
    if coarse is not None:
        q = max(args.orders)
        value, _ = solve_energy(coarse, s1, s2, q)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"invalid level check for {rec}")
        row[f"F_order{q}_levels{args.levels - 1}"] = value
        row["rel_diff_levels"] = abs(hi - value) / hi
    return row


def solve_chunk(task):
    records, args = task
    meshes = setup_meshes(args)
    return [solve_row(rec, meshes, args) for rec in records]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--eval-glob", default="data_results/lshape/p*/eval/*.csv")
    ap.add_argument("--levels", type=int, default=16)
    ap.add_argument("--factor", type=float, default=0.15)
    ap.add_argument("--orders", type=int, nargs="+", default=[12, 14])
    ap.add_argument("--max-sigmas", type=int, default=None)
    ap.add_argument("--grid", choices=["eval", "all"], default="eval")
    ap.add_argument("--origin", action="store_true")
    ap.add_argument("--outer-hpref", type=float, default=None)
    ap.add_argument("--level-check", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be positive")

    sig = test_sigmas(args.eval_glob) if args.grid == "eval" else grid_sigmas()
    if args.max_sigmas:
        sig = sig.head(int(args.max_sigmas))
    version = "v3" if args.grid == "all" else "v2"
    out = REPO / (args.out or f"references_{version}/lshape_energy_ngsolve.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.csv")
    records = sig.to_dict("records")
    start = time.perf_counter()
    chunks = [records[i::args.workers] for i in range(min(args.workers, len(records)))]
    tasks = [(chunk, args) for chunk in chunks]
    if args.workers == 1:
        rows = solve_chunk(tasks[0])
    else:
        with multiprocessing.get_context("spawn").Pool(len(tasks)) as pool:
            rows = [row for chunk in pool.imap_unordered(solve_chunk, tasks) for row in chunk]
    df = pd.DataFrame(rows).sort_values(["sigma1", "sigma2"])
    df.to_csv(partial, index=False)
    partial.replace(out)
    print(f"elapsed {time.perf_counter() - start:.3f}s", flush=True)
    extra = f", levels {df.rel_diff_levels.max():.2e}" if "rel_diff_levels" in df else ""
    print(f"wrote {len(df)} rows -> {out}; max rel diff between orders {df.rel_diff_orders.max():.2e}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
