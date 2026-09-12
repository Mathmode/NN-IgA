#!/usr/bin/env python3
"""Inter-seed REPRODUCIBILITY table (reported separately from the convergence figures,
whose shaded band is the spread over the physical parameter family). For each experiment
and degree it reports the relative inter-seed inter-quartile range of the POSITIONAL
error, (p75-p25)/median taken over the 4 seeds (each seed summarized by its median over
the parameter test set), as the typical (median over the mesh levels N) and the maximum
over N. Uniform Galerkin is essentially seed-independent (~0) and is noted in the caption.

Writes TiKz_overleaf/reproducibility_interseed.{tex,csv}. Run: python3 make_repro_table.py
"""
import csv, glob, os
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
TO = os.path.join(ROOT, "TiKz_overleaf")
# experiment label -> (data slug, error column)
EXPS = [("Singular", "singular", "H1_rel"),
        ("Helmholtz", "helmholtz", "h1_rel_error"),
        ("Arctan", "arctan", "H1_seminorm_rel"),
        ("L-shape", "lshape", "h1_rel_error_iga_ref"),
        ("Adv-diff", "advdiff", "H1_semi_rel")]


def rel_iqr_over_N(slug, err, P, method):
    """For each N: inter-seed (p75-p25)/median over the 4 per-seed medians. Returns the
    list of relative IQRs across N (the levels present in the eval)."""
    by_N = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(ROOT, "data_results", slug, f"p{P}", "eval", "summary_*seed*.csv"))):
        for r in csv.DictReader(open(f)):
            try:
                if r["method"] == method and r.get(err, "") != "":
                    e = float(r[err])
                    if np.isfinite(e):
                        by_N[int(round(float(r["N"])))][r.get("seed", "0")].append(e)
            except (KeyError, ValueError):
                continue
    out = []
    for N in sorted(by_N):
        per_seed = [np.median(v) for v in by_N[N].values() if v]
        if len(per_seed) >= 2:
            med = float(np.median(per_seed))
            if med:
                out.append((np.percentile(per_seed, 75) - np.percentile(per_seed, 25)) / med)
    return out


rows = []
for label, slug, err in EXPS:
    rec = {"exp": label}
    for P in (2, 3):
        w = rel_iqr_over_N(slug, err, P, "positional")
        rec[f"p{P}_med"] = float(np.median(w)) if w else float("nan")
        rec[f"p{P}_max"] = float(np.max(w)) if w else float("nan")
    rows.append(rec)

# CSV
with open(os.path.join(TO, "reproducibility_interseed.csv"), "w", newline="") as fh:
    wcsv = csv.writer(fh)
    wcsv.writerow(["experiment", "p2_interseed_iqr_median", "p2_interseed_iqr_max",
                   "p3_interseed_iqr_median", "p3_interseed_iqr_max"])
    for r in rows:
        wcsv.writerow([r["exp"], f"{r['p2_med']:.4f}", f"{r['p2_max']:.4f}",
                       f"{r['p3_med']:.4f}", f"{r['p3_max']:.4f}"])

# LaTeX (booktabs), percentages
def pct(x):
    return "--" if (x != x) else f"{100*x:.1f}\\%"

lines = [
 r"% Inter-seed reproducibility of the positional (r-adaptive) error.",
 r"% Relative inter-quartile range (p75-p25)/median over the 4 random seeds, each seed",
 r"% summarized by its median over the parameter test set; 'med'/'max' are over mesh",
 r"% levels N. Uniform Galerkin is seed-independent (inter-seed IQR ~ 0).",
 r"\begin{tabular}{lcccc}",
 r"\toprule",
 r" & \multicolumn{2}{c}{$p=2$} & \multicolumn{2}{c}{$p=3$} \\",
 r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
 r"Experiment & median & max & median & max \\",
 r"\midrule",
]
for r in rows:
    lines.append(f"{r['exp']} & {pct(r['p2_med'])} & {pct(r['p2_max'])} & "
                 f"{pct(r['p3_med'])} & {pct(r['p3_max'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TO, "reproducibility_interseed.tex"), "w").write("\n".join(lines) + "\n")

print("inter-seed reproducibility (positional, relative IQR over seeds):")
print(f"  {'exp':10} {'p2 med':>8} {'p2 max':>8} {'p3 med':>8} {'p3 max':>8}")
for r in rows:
    print(f"  {r['exp']:10} {pct(r['p2_med']):>8} {pct(r['p2_max']):>8} {pct(r['p3_med']):>8} {pct(r['p3_max']):>8}")
print(f"\nwrote {TO}/reproducibility_interseed.{{tex,csv}}")
