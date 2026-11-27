from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["sigma1", "sigma2"]


def discrete_energy(ritz_energy):
    return -2.0 * np.asarray(ritz_energy, dtype=np.float64)


def relative_energy_gap(F_h, F):
    return (np.asarray(F) - np.asarray(F_h)) / np.asarray(F)


def relative_energy_error(F_h, F):
    return np.sqrt(np.clip(relative_energy_gap(F_h, F), 0.0, None))


def absolute_energy_error(F_h, F):
    return np.sqrt(np.clip(np.asarray(F) - np.asarray(F_h), 0.0, None))


def attach_energy_errors(rows, energies, column):
    out = rows.merge(energies[KEYS + [column]], on=KEYS, how="left", validate="many_to_one")
    out["F_h"] = discrete_energy(out["ritz_energy"])
    out["F_ref"] = out[column]
    out["gap"] = relative_energy_gap(out["F_h"], out["F_ref"])
    out["err_E"] = relative_energy_error(out["F_h"], out["F_ref"])
    out["I_eff_E"] = out["eta"] / absolute_energy_error(out["F_h"], out["F_ref"])
    return out


def median_table(rows, value):
    return rows.groupby(["p", "N", "method"])[value].median().unstack("method")


def improvement_factors(table, reference="uniform", adapted="positional"):
    return table[reference] / table[adapted]


def endpoint_order(errors, levels):
    e = np.asarray(errors, dtype=np.float64)
    n = np.asarray(levels, dtype=np.float64)
    return float(np.log(e[0] / e[-1]) / np.log(n[-1] / n[0]))


def local_orders(errors, levels):
    e = np.asarray(errors, dtype=np.float64)
    n = np.asarray(levels, dtype=np.float64)
    return np.log(e[:-1] / e[1:]) / np.log(n[1:] / n[:-1])


def fitted_order(errors, levels):
    slope, _ = np.polyfit(np.log(np.asarray(levels, float)), np.log(np.asarray(errors, float)), 1)
    return float(-slope)


def order_summary(table, method, fit_from=16):
    rows = []
    for p, sub in table[method].groupby(level="p"):
        n = sub.index.get_level_values("N").to_numpy()
        e = sub.to_numpy()
        keep = n >= fit_from
        rows.append({
            "p": int(p),
            "method": method,
            "endpoint_order": endpoint_order(e, n),
            "fitted_order_from_%d" % fit_from: fitted_order(e[keep], n[keep]),
            "local_orders": " ".join(f"{v:.2f}" for v in local_orders(e, n)),
        })
    return pd.DataFrame(rows)


__all__ = [
    "discrete_energy",
    "relative_energy_gap",
    "relative_energy_error",
    "absolute_energy_error",
    "attach_energy_errors",
    "median_table",
    "improvement_factors",
    "endpoint_order",
    "local_orders",
    "fitted_order",
    "order_summary",
]
