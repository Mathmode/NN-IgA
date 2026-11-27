"""Summary of the 2D comparison network / Algorithm 1 (revision v2, Sec. 5.8).

Collects data_results_v2/alg1_2d/<exp>_p<p>_chunk*.csv (Hyperion job
slurm_v2/alg1_2d.sbatch; falls back to the local *_local.csv runs) and writes the
median error, median time and the ratio network / Algorithm 1 per (experiment,
p, N, method).

Usage:
    python scripts/summarize_alg1_2d.py --out data_results_v2/alg1_2d/summary.csv
    python scripts/summarize_alg1_2d.py --in-dir data_results_v2/alg1_2d_bounded --out data_results_v2/alg1_2d_bounded/summary.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
D = REPO / "data_results_v2" / "alg1_2d"


def load(exp, p, d=D):
    files = glob.glob(str(d / f"{exp}_p{p}_chunk*.csv")) or glob.glob(str(d / f"{exp}_p{p}_local.csv"))
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True) if files else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=str(D.relative_to(REPO)))
    ap.add_argument("--out", default="data_results_v2/alg1_2d/summary.csv")
    args = ap.parse_args()
    rows = []
    for exp in ("arctan", "lshape"):
        for p in (2, 3):
            d = load(exp, p, REPO / args.in_dir)
            if d is None:
                continue
            g = d.groupby(["N", "method"]).agg(err=("err", "median"), time_s=("time_s", "median"),
                                               n=("nu", "nunique"), n_valid=("err", "count")).reset_index()
            rows.append(g.assign(experiment=exp, p=p))
    df = pd.concat(rows, ignore_index=True)
    wide = df.pivot_table(index=["experiment", "p", "N"], columns="method", values="err")
    for net, alg in (("net-eta", "alg1-eta"), ("net-ritz", "alg1-ritz")):
        if net in wide and alg in wide:
            wide[f"{net}/{alg}"] = wide[net] / wide[alg]
    df.to_csv(REPO / args.out, index=False)
    pd.set_option("display.width", 220)
    print(wide.to_string(float_format=lambda v: f"{v:.3g}"))
    print(df.pivot_table(index=["experiment", "p", "N"], columns="method", values="time_s")
          .to_string(float_format=lambda v: f"{v:.3g}"))
    bad = df[df.n_valid < df.n]
    if len(bad):
        print("groups with non-finite errors:")
        print(bad.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
