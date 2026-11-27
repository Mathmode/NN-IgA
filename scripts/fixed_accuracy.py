"""Fixed-accuracy comparison (revision v2, item 3.4).

For every experiment and degree, the target is the median error of the uniform
mesh at the finest evaluated level N_max. The number of elements per direction
the adapted mesh needs to reach it is found by log-log interpolation of its
median error curve, and the DOF ratio is ((N_max + p) / (N_a + p))^d.
Medians follow scripts/make_tables.py (pooled over seeds and test parameters);
Experiment 4 uses the corrected energy-identity errors
(data_results_v2/lshape/energy_error_rows.csv), Experiment 6 the new runs.

Usage:
    python scripts/fixed_accuracy.py --out data_results_v2/fixed_accuracy.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import make_tables as MT

DIM = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 1}
NAMES = {1: "singular", 2: "helmholtz", 3: "arctan", 4: "lshape (energy identity)", 5: "advdiff", 6: "adr"}


def medians(k, p):
    if k == 4:
        d = pd.read_csv(REPO / "data_results_v2" / "lshape" / "energy_error_rows.csv")
        d = d[d.p == p].assign(err=lambda x: x.err_E)
    elif k == 6:
        files = glob.glob(str(REPO / "data_results_v2" / "adr" / f"p{p}" / "summary_p*_seed*.csv"))
        if not files:
            return None
        d = pd.concat(pd.read_csv(f) for f in files)
        d = d.assign(err=d.err_E, method=d.method.replace({"adapted": "positional"}))
    else:
        d = MT.SPECS[k]["rows"](p)
    return d.groupby(["N", "method"]).err.median().unstack("method").sort_index()


def needed_level(levels, adapted, target):
    la, ln = np.log(adapted), np.log(levels)
    i = next((i for i in range(len(adapted)) if adapted[i] <= target), None)
    if i is None:
        return np.nan, ">"
    if i == 0:
        return float(levels[0]), "<="
    t = (np.log(target) - la[i - 1]) / (la[i] - la[i - 1])
    return float(np.exp(ln[i - 1] + t * (ln[i] - ln[i - 1]))), "~"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_results_v2/fixed_accuracy.csv")
    args = ap.parse_args()
    rows = []
    for k in (1, 2, 3, 4, 5, 6):
        for p in (2, 3):
            m = medians(k, p)
            if m is None:
                continue
            N = m.index.to_numpy(dtype=float)
            target = float(m["uniform"].iloc[-1])
            Na, tag = needed_level(N, m["positional"].to_numpy(), target)
            rows.append(dict(experiment=k, name=NAMES[k], p=p, N_max=int(N[-1]), target=target,
                             N_adapted=Na, bound=tag, dof_ratio=((N[-1] + p) / (Na + p)) ** DIM[k]))
    df = pd.DataFrame(rows)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
