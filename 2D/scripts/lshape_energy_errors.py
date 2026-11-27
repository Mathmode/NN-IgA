"""Reference-free L-shape errors (Experiment 4) via the energy identity.

The discrete space is conforming and the problem is symmetric with f = 1, so

    ||u - u_h||_E^2 / ||u||_E^2 = 1 - F(u_h) / F(u),   F(v) = int_Omega v,

with F(u_h) = F^T U = -2 J(u_h) taken from the stored ``ritz_energy`` column
(exact) and F(u) from the independent NGSolve hp reference
(``lshape_ngsolve_energies.py``). Writes per-row errors, the corrected Table 4
(Table R5 in the response letter) and the effective orders.

Usage:
    python 2D/scripts/lshape_energy_errors.py \
        --energies references_v2/lshape_energy_ngsolve.csv --out-dir data_results_v2/lshape
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "2D")]

import numpy as np
import pandas as pd

from src.nonparametric.lshape.energy_error import (
    attach_energy_errors,
    improvement_factors,
    median_table,
    order_summary,
)


def load_rows(eval_glob):
    files = sorted(REPO.glob(eval_glob))
    if not files:
        raise SystemExit(f"no evaluation CSVs match {eval_glob}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def align_parameter_pairs(rows, energies):
    keys = ["sigma1", "sigma2"]
    pairs = rows[keys].drop_duplicates()
    target = pairs.to_numpy()
    source = energies[keys].to_numpy()
    close = (np.abs(target[:, None, :] - source[None, :, :]) < 1e-12).all(axis=2)
    if not (close.sum(axis=1) == 1).all():
        raise ValueError("missing or ambiguous energy reference for an evaluation pair")
    matched = energies.iloc[close.argmax(axis=1)].copy()
    matched[keys] = target
    return matched


def bracket_energies(energies, best, other):
    spread = (energies[best] - energies[other]).abs()
    out = energies.copy()
    out["F_lo"] = energies[best] - spread
    out["F_hi"] = energies[best] + spread
    return out


def checks(rows, tol):
    cons = ((rows["F_h"] - rows["sigma_h1_seminorm_sq_fem"]).abs() / rows["F_h"]).max()
    return {
        "rows": int(len(rows)),
        "missing_reference": int(rows["F_ref"].isna().sum()),
        "max_rel_galerkin_consistency": float(cons),
        "rows_with_F_h_above_F": int((rows["gap"] < -tol).sum()),
        "min_gap": float(rows["gap"].min()),
    }


def corrected_table(rows):
    err = median_table(rows, "err_E")
    sub = median_table(rows, "h1_rel_error_iga_ref")
    ieff = median_table(rows, "I_eff_E")
    ieff_sub = median_table(rows.assign(ieff=rows["eta"] / rows["h1_seminorm_abs_iga_ref"]), "ieff")
    return pd.DataFrame({
        "uniform_submitted": sub["uniform"],
        "adapted_submitted": sub["positional"],
        "uniform_corrected": err["uniform"],
        "adapted_corrected": err["positional"],
        "factor_corrected": improvement_factors(err),
        "factor_submitted": improvement_factors(sub),
        "I_eff_uniform_corrected": ieff["uniform"],
        "I_eff_adapted_corrected": ieff["positional"],
        "I_eff_uniform_submitted": ieff_sub["uniform"],
        "I_eff_adapted_submitted": ieff_sub["positional"],
    })


def factor_range(rows, energies):
    out = {}
    for col in ("F_lo", "F_hi"):
        r = attach_energy_errors(rows.drop(columns=["F_ref", "F_h", "gap", "err_E", "I_eff_E"]), energies, col)
        out[col] = improvement_factors(median_table(r, "err_E"))
    return pd.DataFrame({"factor_with_F_minus_spread": out["F_lo"], "factor_with_F_plus_spread": out["F_hi"]})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-glob", default="data_results/lshape/p*/eval/*.csv")
    ap.add_argument("--reference-version", choices=["v2", "v3"], default="v2")
    ap.add_argument("--energies")
    ap.add_argument("--best")
    ap.add_argument("--other")
    ap.add_argument("--out-dir")
    ap.add_argument("--tol", type=float, default=2e-8)
    args = ap.parse_args()

    version = args.reference_version
    path = args.energies or f"references_{version}/lshape_energy_ngsolve.csv"
    energies = pd.read_csv(REPO / path, float_precision="round_trip" if version == "v3" else None)
    orders = sorted((int(c[7:]) for c in energies if c.startswith("F_order") and c[7:].isdigit()))
    best = args.best or f"F_order{orders[-1]}"
    other = args.other or f"F_order{orders[-2]}"
    energies = bracket_energies(energies, best, other)
    raw = load_rows(args.eval_glob)
    energies = align_parameter_pairs(raw, energies)
    rows = attach_energy_errors(raw, energies, best)
    rows = rows.merge(energies[["sigma1", "sigma2", "F_lo", "F_hi"]], on=["sigma1", "sigma2"], how="left")
    info = checks(rows, args.tol)

    table = corrected_table(rows)
    table = table.join(factor_range(rows.drop(columns=["F_lo", "F_hi"]), energies))
    err = median_table(rows, "err_E")
    orders = pd.concat([order_summary(err, m) for m in ("uniform", "positional")], ignore_index=True)

    out = REPO / (args.out_dir or f"data_results_{version}/lshape")
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "energy_error_rows.csv", index=False)
    table.to_csv(out / "table4_corrected.csv")
    orders.to_csv(out / "effective_orders.csv", index=False)
    pd.Series(info).to_csv(out / "checks.csv", header=["value"])

    pd.set_option("display.width", 220)
    for k, v in info.items():
        print(f"{k}: {v}")
    print(table.map(lambda v: f"{v:.3g}").to_string())
    print(orders.to_string(index=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
