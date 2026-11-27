"""Computational-cost table (revision v2, item 3.4).

All timings of the table come from one directory (--root, default
data_results_v2/icelake), written by Hyperion jobs on the same CPU model:
  T_train   wall time of one training run, median over the seeds of
            <root>/timing/<exp>/p<p>/ (slurm_v2/timing_runs.sbatch), with min, max
            and peak memory
  T_online  network forward pass + one Galerkin solve, median over the test
            parameters of <root>/mesh_diagnostics/conditioning_online_*.csv
  T_alg1    Algorithm 1 on Theta_ad with the continuation ladder up to the comparison
            level, median over the test parameters (<root>/loss_ablation,
            <root>/alg1_2d)
  n*        break-even number of queries T_train / (T_alg1 - T_online)
  cpu       CPU models seen in the three sources (a single model when nothing is mixed)
T_train_paper_s is the median of the original paper runs (data_results/, mixed nodes),
reported only for comparison.

Usage:
    python scripts/cost_table.py --root data_results_v2/icelake --out data_results_v2/cost_table.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
D1, D2 = REPO / "data_results", REPO / "data_results_v2"
COMPARE_N = {"singular": 64, "helmholtz": 128, "arctan": 32, "lshape": 32, "advdiff": 32, "adr": 64}


def _json(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(str(pattern), recursive=True))]


def _seconds(m):
    for k in ("training_elapsed_sec", "elapsed_sec"):
        if k in m:
            return float(m[k])
    return np.nan


def _cpu(m):
    return (m.get("hardware") or {}).get("cpu_model", "")


def paper_times():
    specs = {
        "singular": D1 / "singular" / "p{p}" / "metadata_p{p}_seed*.json",
        "helmholtz": D1 / "helmholtz" / "p{p}" / "meta_train_p5c_p{p}_seed*.json",
        "arctan": D1 / "arctan" / "p{p}" / "meta_train_arctan_seed*.json",
        "lshape": D1 / "lshape" / "p{p}" / "meta_train_lshape_seed*.json",
        "adr": D2 / "adr" / "p{p}" / "metadata_p{p}_seed*.json",
    }
    rows = []
    for exp, pat in specs.items():
        for p in (2, 3):
            t = [x for x in (_seconds(m) for m in _json(str(pat).format(p=p))) if np.isfinite(x)]
            rows.append(dict(experiment=exp, p=p, T_train_paper_s=float(np.median(t)) if t else np.nan,
                             n_paper=len(t)))
    return pd.DataFrame(rows)


def train_times(root):
    rows = []
    for exp in ["singular", "helmholtz", "arctan", "lshape", "advdiff", "adr"]:
        for p in (2, 3):
            ms = [m for m in _json(root / "timing" / exp / f"p{p}" / "*.json") if "peak_rss_mb" in m]
            t = np.array([_seconds(m) for m in ms])
            rss = np.array([float(m["peak_rss_mb"]) for m in ms])
            rows.append(dict(
                experiment=exp, p=p, n_runs=len(ms),
                T_train_s=float(np.median(t)) if len(t) else np.nan,
                T_train_min_s=float(t.min()) if len(t) else np.nan,
                T_train_max_s=float(t.max()) if len(t) else np.nan,
                peak_rss_mb=float(np.median(rss)) if len(rss) else np.nan,
                peak_rss_max_mb=float(rss.max()) if len(rss) else np.nan,
                cpu_train=";".join(sorted({_cpu(m) for m in ms})),
            ))
    return pd.DataFrame(rows)


def online_times(root):
    frames = [pd.read_csv(f) for f in glob.glob(str(root / "mesh_diagnostics" / "conditioning_online_*.csv"))]
    if not frames:
        return pd.DataFrame(columns=["experiment", "p", "T_online_s", "N_online"])
    d = pd.concat(frames)
    d = d[d.method == "adapted"]
    pick = []
    for exp, g in d.groupby("experiment"):
        N = COMPARE_N[exp] if COMPARE_N[exp] in set(g.N) else g.N.max()
        pick.append(g[g.N == N].assign(N_online=N))
    d = pd.concat(pick)
    return d.groupby(["experiment", "p"]).agg(T_online_s=("online_s", "median"), N_online=("N_online", "first")).reset_index()


def alg1_times(root):
    rows = []
    files = (glob.glob(str(root / "loss_ablation" / "exp1_alg1_ladder_chunk*.csv"))
             or glob.glob(str(root / "loss_ablation" / "exp1_alg1_ladder.csv")))
    if files:
        d = pd.concat(pd.read_csv(x) for x in files)
        d = d[(d.method == "alg1-eta") & (d.N == COMPARE_N["singular"])]
        rows += [dict(experiment="singular", p=p, T_alg1_s=float(g.time_s.median())) for p, g in d.groupby("p")]
    for exp in ("arctan", "lshape"):
        files = glob.glob(str(root / "alg1_2d" / f"{exp}_p*_chunk*.csv"))
        if files:
            d = pd.concat(pd.read_csv(x) for x in files)
            d = d[(d.method == "alg1-eta") & (d.N == COMPARE_N[exp])]
            rows += [dict(experiment=exp, p=p, T_alg1_s=float(g.time_s.median())) for p, g in d.groupby("p")]
    return pd.DataFrame(rows, columns=["experiment", "p", "T_alg1_s"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data_results_v2/icelake")
    ap.add_argument("--out", default="data_results_v2/cost_table.csv")
    args = ap.parse_args()
    root = REPO / args.root
    df = (train_times(root).merge(online_times(root), how="left").merge(alg1_times(root), how="left")
          .merge(paper_times(), how="left"))
    df["N_compare"] = df.experiment.map(COMPARE_N)
    df["break_even_n"] = df.T_train_s / (df.T_alg1_s - df.T_online_s)
    df.to_csv(REPO / args.out, index=False)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
