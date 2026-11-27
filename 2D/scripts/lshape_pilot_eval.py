from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from alg1_lshape_Tscan import REPO, reference_energy, realized_hmin, split_instances
from src.nonparametric.alg1_ladder_2d import mesh_quality
from src.nonparametric.lshape.variational_energy import energy_and_load
from src.parametric.continuation import load_checkpoint
from src.parametric.evaluation_v2 import _eval_p3_network


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--p", type=int, choices=(2, 3), required=True)
    ap.add_argument("--variant", choices=("V0", "V1", "V2", "V3"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    params = load_checkpoint(args.checkpoint)
    references = pd.read_csv(REPO / "references_v3/lshape_energy_ngsolve.csv", float_precision="round_trip")
    references = references.loc[references.split == "val"]
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    rows = []
    for n in ((4, 8) if args.smoke else (32, 64)):
        for s in split_instances("val", 1 if args.smoke else len(references)):
            f = reference_energy(references, s, "val")
            sol = _eval_p3_network(params, s, p=args.p, n_elem=n, q_K=args.p + 1, q_F=4)
            x, y = sol["knots_x"], sol["knots_y"]
            strain, load = energy_and_load(sol["u_h"], x, y, args.p, *s)
            gap = float((f + strain - 2 * load) / f)
            gamma, _ = mesh_quality(x, y, args.p)
            if not np.isfinite(gap) or gap < 0 or not np.isfinite(gamma):
                raise ValueError(f"invalid validation observation for p={args.p}, N={n}, sigma={s}: {gap}, {gamma}")
            rows.append(dict(variant=args.variant, p=args.p, N=n, split="val", sigma1=s[0], sigma2=s[1],
                             F_ref=f, strain=float(strain), load=float(load), ritz=sol["ritz"], gap=gap,
                             err=np.sqrt(gap), gamma_loc=gamma, h_min=realized_hmin(x, y, n), checkpoint_sha256=digest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    print(frame.groupby("N")[["err", "gamma_loc"]].median().to_string())


if __name__ == "__main__":
    main()
