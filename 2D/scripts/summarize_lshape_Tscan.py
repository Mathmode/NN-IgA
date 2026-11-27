from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.alg1_lshape_Tscan import REPO, reference_energy, split_instances

LEVELS = (4, 8, 16, 32, 64, 128)
BOUNDS = (5, 8, 10, 15)
RESOLUTION = 1e-11


def read_scan(root, energies, split):
    expected = split_instances(split, 16)
    frames = []
    for p in (2, 3):
        for bound in BOUNDS:
            for i, sigma in enumerate(expected):
                path = root / f"{split}_p{p}_T{bound}_i{i}.csv"
                data = pd.read_csv(path, float_precision="round_trip")
                index = pd.MultiIndex.from_product([["uniform", "alg1-eta"], LEVELS], names=["method", "N"])
                if split == "test" and not (data.cpu.str.contains("Platinum 8362 ", regex=False) & (data.cpus == 4)).all():
                    raise ValueError(f"unexpected test sweep hardware in {path}")
                actual = data.set_index(["method", "N"]).index
                if not actual.is_unique or not actual.sort_values().equals(index.sort_values()):
                    raise ValueError(f"incomplete ladder in {path}")
                if not (data.split.eq(split) & data.p.eq(p) & data["T"].eq(bound)).all():
                    raise ValueError(f"incorrect selection keys in {path}")
                if not np.allclose(data[["sigma1", "sigma2"]], sigma, atol=1e-12, rtol=0):
                    raise ValueError(f"incorrect {split} pair in {path}")
                F = reference_energy(energies, sigma, split)
                if not np.allclose(data.F_ref, F, atol=0, rtol=2e-15):
                    raise ValueError(f"reference mismatch in {path}")
                finite = data[["gap", "ritz", "ritz_assembled", "F_load", "strain_quadrature",
                               "load_quadrature", "eta", "h_min", "gamma_loc", "time_s"]]
                if not np.isfinite(finite).all().all() or not (data.h_min > 0).all():
                    raise ValueError(f"invalid scan row in {path}")
                np.testing.assert_allclose(data.gap, (F + 2 * data.ritz) / F, atol=5e-15, rtol=0)
                np.testing.assert_allclose(data.gap, (F + data.strain_quadrature - 2 * data.load_quadrature) / F,
                                           atol=5e-15, rtol=0)
                frames.append(data[data.method == "alg1-eta"])
    return pd.concat(frames, ignore_index=True)


def read_validation(root, energies):
    return read_scan(root, energies, "val")


def scan_tables(data):
    rows = []
    for (p, N, bound), group in data.groupby(["p", "N", "T"]):
        gap = group.gap.to_numpy()
        resolved = gap > RESOLUTION
        lower = np.sqrt(np.where(resolved, gap, 0))
        upper = np.sqrt(np.where(resolved, gap, RESOLUTION))
        invalid = gap < -RESOLUTION
        upper[invalid] = np.inf
        lo, hi = np.median(lower), np.median(upper)
        work_gap = (group.F_ref - group.F_load) / group.F_ref
        work_median_gap = float(np.median(work_gap))
        work_median = float(np.median(np.sqrt(work_gap.clip(lower=0)))) if work_median_gap > RESOLUTION else np.nan
        consistency = ((group.F_load + 2 * group.ritz) / group.F_ref).abs()
        stops = group.stop.value_counts()
        assembled_gap = (group.F_ref + 2 * group.ritz_assembled) / group.F_ref
        rows.append(dict(p=p, N=N, T=bound, n=len(group), median_err=lo if lo == hi else np.nan,
                         median_lo=lo, median_hi=hi, n_unresolved=int((~resolved).sum()),
                         min_gap=gap.min(), max_ritz_load_diff=consistency.max(), median_err_from_load=work_median,
                         max_assembled_gap_diff=float(np.max(np.abs(assembled_gap - gap))),
                         n_negative_assembled=int((assembled_gap < 0).sum()),
                         n_zero_iter=int((group.nit == 0).sum()), n_gradient=int(stops.get("gradient", 0)),
                         n_line_search=int(stops.get("line_search", 0)), n_maxiter=int(stops.get("maxiter", 0)),
                         n_below_minus_tolerance=int((gap < -RESOLUTION).sum()),
                         h_min=group.h_min.min(), median_gamma_loc=group.gamma_loc.median()))
    medians = pd.DataFrame(rows)
    values = medians.pivot(index=["p", "T"], columns="N", values="median_err")
    rates = np.log2(values[32] / values[64]).rename("order_32_64").reset_index()
    return medians, rates


def validation_tables(data):
    medians, rates = scan_tables(data)
    schedule = []
    for (p, N), sub in medians.groupby(["p", "N"]):
        sub = sub.sort_values("T")
        if sub.median_err.isna().any():
            schedule.append(dict(p=p, N=N, T=np.nan, h_min_floor=np.nan))
            continue
        best_larger = sub.median_err.iloc[::-1].cummin().iloc[::-1]
        selected = sub.loc[(1 - best_larger / sub.median_err) < 0.02].iloc[0]
        schedule.append(dict(p=p, N=N, T=int(selected["T"]), h_min_floor=max(1e-12, selected.h_min / 10)))
    return medians, rates, pd.DataFrame(schedule)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO / "data_results_v3/lshape_alg1_Tscan")
    parser.add_argument("--split", choices=("val","test"), default="val")
    args = parser.parse_args()
    energies = pd.read_csv(REPO / "references_v3/lshape_energy_ngsolve.csv", float_precision="round_trip")
    data = read_scan(args.root, energies, args.split)
    if args.split == "test":
        medians, rates = scan_tables(data)
        medians.to_csv(args.root / "test_medians.csv", index=False)
        rates.to_csv(args.root / "test_orders.csv", index=False)
        return
    medians, rates, schedule = validation_tables(data)
    for name, frame in [("validation_medians", medians), ("validation_orders", rates), ("validation_schedule", schedule)]:
        frame.to_csv(args.root / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
