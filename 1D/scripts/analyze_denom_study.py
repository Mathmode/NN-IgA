#!/usr/bin/env python3
"""Tabulate metrics for the helmholtz DENOMINATOR STUDY (uniform / exact / robust).

This script computes and tabulates METRICS ONLY -- it does NOT emit verdicts, fix
success thresholds, or decide anything. The decision is made elsewhere from these
numbers. Per denominator and degree (aggregated as the median over seeds):

  Training stability
    - %rises : fraction of epochs whose train_loss exceeds the previous epoch's,
               per level and global (concatenated).
    - epochs/level : number of epochs actually used at each level (early-stopping
               means this differs between denominators -- reported alongside every
               metric so it is never forgotten).
  Descent
    - decades : log10(first/last) of train_loss, per level and global.
  Error dispersion
    - from the EVAL summary if present (per-contrast h1_rel_error per N): the band
      width (p75-p25)/median, plus corr(h1_rel_error, contrast c) per N;
    - else from the history h1_rel_val_median (per-epoch val error median): its
      median and (p75-p25)/median per level (a proxy until eval is run).

Data layout (one tree per denominator; produced by submit_denom_study.sbatch):
    <data-root>/helmholtz_denom_<denom>/p<deg>/history/history_p<deg>_seed<k>.csv
    <data-root>/helmholtz_denom_<denom>/p<deg>/eval/summary_*_seed<k>.csv   (optional)

METHODOLOGICAL CAVEAT (printed at the top of the report): because the early-stopping
monitor (val_eta2) is kept, (a) each denominator may stop at a DIFFERENT epoch, so
%rises / decades are not strictly comparable at equal epochs; and (b) val_eta2 is a
DIFFERENT quantity under each denominator (a different object is monitored under
exact vs uniform/robust). A fixed-epoch second batch would be needed for a strict
equal-epochs comparison; that is intentionally NOT done here.

Usage:
    python scripts/analyze_denom_study.py
    python scripts/analyze_denom_study.py --data-root <path> --p 2 3 --seeds 0 1 2 3
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path

import numpy as np

HIST_COLS = ("level_N", "iteration", "train_loss", "val_eta2", "h1_rel_val_median")


# ----------------------------------------------------------------------------- IO
def _read_csv_rows(path: Path):
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    return list(reader), (reader.fieldnames or [])


def load_history(path: Path):
    rows, cols = _read_csv_rows(path)
    missing = [c for c in HIST_COLS if c not in cols]
    assert not missing, f"{path}: missing history column(s) {missing}; found {cols}"
    assert rows, f"{path}: no rows"
    return rows


def _col(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return np.asarray(out, dtype=float)


# ------------------------------------------------------------------------ metrics
def pct_rises(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.mean(x[1:] > x[:-1]))


def decades(x: np.ndarray) -> float:
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 2:
        return float("nan")
    return float(np.log10(x[0] / x[-1]))


def rel_iqr(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    med = float(np.median(x))
    return float((np.percentile(x, 75) - np.percentile(x, 25)) / med) if med != 0 else float("nan")


def history_metrics_one_seed(rows):
    """Per-level + global metrics for one history CSV."""
    levelN = _col(rows, "level_N").astype(int)
    tl = _col(rows, "train_loss")
    h1 = _col(rows, "h1_rel_val_median")
    out = {"global": {"epochs": int(tl.size), "pct_rises": pct_rises(tl), "decades": decades(tl)}, "levels": {}}
    for N in sorted(set(levelN.tolist())):
        m = levelN == N
        out["levels"][N] = {
            "epochs": int(m.sum()),
            "pct_rises": pct_rises(tl[m]),
            "decades": decades(tl[m]),
            "h1_med": float(np.median(h1[m][np.isfinite(h1[m])])) if np.isfinite(h1[m]).any() else float("nan"),
            "h1_relIQR": rel_iqr(h1[m]),
        }
    return out


def eval_metrics_one_seed(path: Path):
    """Per-N band width + error-contrast correlation from an eval summary, if it has
    the expected columns (c, h1_rel_error, N). Returns {} if not usable."""
    rows, cols = _read_csv_rows(path)
    if not ({"c", "h1_rel_error", "N"} <= set(cols)):
        return {}
    N = _col(rows, "N").astype(int)
    c = _col(rows, "c")
    err = _col(rows, "h1_rel_error")
    res = {}
    for n in sorted(set(N.tolist())):
        m = (N == n) & np.isfinite(err) & np.isfinite(c)
        if m.sum() < 2:
            continue
        e, cc = err[m], c[m]
        corr = float(np.corrcoef(e, cc)[0, 1]) if e.size >= 2 and np.std(e) > 0 and np.std(cc) > 0 else float("nan")
        res[n] = {"band_relIQR": rel_iqr(e), "err_med": float(np.median(e)),
                  "corr_err_c": corr, "n_contrasts": int(m.sum())}
    return res


# ------------------------------------------------------------------- aggregation
def med_over_seeds(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def gather(data_root: Path, denom: str, p: int, seeds):
    base = data_root / f"helmholtz_denom_{denom}" / f"p{p}"
    hist_seeds, eval_seeds = [], []
    for s in seeds:
        hp = base / "history" / f"history_p{p}_seed{s}.csv"
        if hp.exists():
            hist_seeds.append(history_metrics_one_seed(load_history(hp)))
        ev = sorted(glob.glob(str(base / "eval" / f"summary_*seed{s}.csv")))
        if ev:
            em = eval_metrics_one_seed(Path(ev[0]))
            if em:
                eval_seeds.append(em)
    return hist_seeds, eval_seeds


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  --  "
    return f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Metrics for the helmholtz denominator study (no verdicts).")
    here = Path(__file__).resolve().parent
    default_root = here.parent.parent / "data_results"
    ap.add_argument("--data-root", type=Path, default=default_root)
    ap.add_argument("--denoms", nargs="+", default=["uniform", "exact", "robust"])
    ap.add_argument("--p", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()

    print("=== helmholtz denominator study -- metrics (NO verdicts, NO thresholds) ===")
    print(f"    data-root: {args.data_root}")
    print("    CAVEAT (early-stopping kept): each denominator may stop at a DIFFERENT")
    print("    epoch (epochs/level reported beside every metric), and val_eta2 is a")
    print("    DIFFERENT quantity under each denominator -- so %rises and decades are")
    print("    NOT strictly comparable at equal epochs. A fixed-epoch second batch")
    print("    would be needed for that; it is intentionally not done here.")
    print()

    for p in args.p:
        print(f"================ degree p={p}  (median over seeds {args.seeds}) ================")
        hdr = (f"{'denom':>8} | {'epochs_tot':>10} | {'%rises_glob':>11} | {'decades_glob':>12} | "
               f"{'h1med_lastN':>11} | {'h1relIQR_lastN':>14} | {'eval band/corr (finest N)':>26}")
        print(hdr)
        print("-" * len(hdr))
        per_denom_levels = {}
        for denom in args.denoms:
            hs, es = gather(args.data_root, denom, p, args.seeds)
            if not hs:
                print(f"{denom:>8} | {'MISSING (no history found)':>40}")
                per_denom_levels[denom] = (None, None)
                continue
            # global, median over seeds
            epochs_tot = med_over_seeds([h["global"]["epochs"] for h in hs])
            rises_g = med_over_seeds([h["global"]["pct_rises"] for h in hs])
            dec_g = med_over_seeds([h["global"]["decades"] for h in hs])
            # last level present (finest)
            all_levels = sorted(set().union(*[set(h["levels"]) for h in hs]))
            lastN = all_levels[-1]
            h1med_last = med_over_seeds([h["levels"].get(lastN, {}).get("h1_med") for h in hs])
            h1iqr_last = med_over_seeds([h["levels"].get(lastN, {}).get("h1_relIQR") for h in hs])
            # eval (finest N) if present
            ev_txt = "-- (no eval yet)"
            if es:
                ev_levels = sorted(set().union(*[set(e) for e in es]))
                evN = ev_levels[-1]
                band = med_over_seeds([e.get(evN, {}).get("band_relIQR") for e in es])
                corr = med_over_seeds([e.get(evN, {}).get("corr_err_c") for e in es])
                ev_txt = f"N={evN}: band={fmt(band)} corr={fmt(corr,2)}"
            print(f"{denom:>8} | {epochs_tot:>10.0f} | {fmt(rises_g):>11} | {fmt(dec_g):>12} | "
                  f"{fmt(h1med_last):>11} | {fmt(h1iqr_last):>14} | {ev_txt:>26}")
            per_denom_levels[denom] = (hs, es)

        # per-level detail
        print()
        print(f"  per-level detail p={p} (median over seeds): N | epochs | %rises | decades | h1_med | h1_relIQR"
              " [| eval band | corr]")
        for denom in args.denoms:
            hs, es = per_denom_levels.get(denom, (None, None))
            if not hs:
                continue
            print(f"  [{denom}]")
            all_levels = sorted(set().union(*[set(h["levels"]) for h in hs]))
            for N in all_levels:
                ep = med_over_seeds([h["levels"].get(N, {}).get("epochs") for h in hs])
                rs = med_over_seeds([h["levels"].get(N, {}).get("pct_rises") for h in hs])
                dc = med_over_seeds([h["levels"].get(N, {}).get("decades") for h in hs])
                hm = med_over_seeds([h["levels"].get(N, {}).get("h1_med") for h in hs])
                hq = med_over_seeds([h["levels"].get(N, {}).get("h1_relIQR") for h in hs])
                ev_txt = ""
                if es:
                    band = med_over_seeds([e.get(N, {}).get("band_relIQR") for e in es])
                    corr = med_over_seeds([e.get(N, {}).get("corr_err_c") for e in es])
                    if np.isfinite(band) or np.isfinite(corr):
                        ev_txt = f" | band={fmt(band)} | corr={fmt(corr,2)}"
                print(f"     N={N:<4} | {ep:>6.0f} | {fmt(rs):>6} | {fmt(dc):>7} | {fmt(hm):>7} | {fmt(hq):>7}{ev_txt}")
        print()

    print("Reminder: these are descriptive metrics. No denominator is declared better")
    print("here; read the numbers together with the epochs/level and the caveat above.")


if __name__ == "__main__":
    main()
