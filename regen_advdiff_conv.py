#!/usr/bin/env python3
"""Regenerate ONLY the advdiff (exp5, P5) convergence figures' embedded band CSV +
reference slope + shared y-axis, from the CURRENT data_results/advdiff eval summaries.

Band recipe (identical to the original _data_source/regen.py builder):
  for each (p, N, method): group the eval rows by the family params (logeps, b),
  take the MEDIAN over seeds within each param, then med/p25/p75 OVER the params.
Reference slope: C = positional err_med(N_min) * N_min**p  (k = p; N^-2 for p2, N^-3 p3).
Shared y-axis: decades bracketing the median curves of BOTH degrees.
Error column: H1_semi_rel. Methods plotted: positional + uniform.

Only the advdiff convergence .tex are touched (embedded filecontents block + the O(N^-k)
line + ymin/ymax). Everything else in those files is preserved. Run: python regen_advdiff_conv.py
"""
import csv, glob, math, os, re
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "TiKz_overleaf", "exp5_advdiff")
ERR = "H1_semi_rel"
NS = [4, 8, 16, 32, 64]
METHODS = ["positional", "uniform"]
BLK = re.compile(r"(\\begin\{filecontents\*\}(?:\[[^\]]*\])?\{)([^}]+)(\}\s*\n)(.*?)(\\end\{filecontents\*\})", re.S)


def band(P, N, method):
    """med/p25/p75 over params of (median over seeds) — the parameter-spread band."""
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ROOT, "data_results", "advdiff", f"p{P}", "eval", "summary_*seed*.csv"))):
        for r in csv.DictReader(open(f)):
            try:
                if int(round(float(r["N"]))) == N and r["method"] == method and r.get(ERR, "") != "":
                    v = float(r[ERR])
                    if np.isfinite(v):
                        groups[(r["logeps"], r["b"])].append(v)
            except (KeyError, ValueError):
                continue
    per = [float(np.median(v)) for v in groups.values() if v]     # median over seeds, per param
    return (float(np.median(per)), float(np.percentile(per, 25)), float(np.percentile(per, 75)),
            len(per), max(len(v) for v in groups.values()))


# ---- build the CSV rows (both degrees) + collect medians for slope/axis ----
rows = ["p,N,method,err_med,err_p25,err_p75"]
med = {}
nparam = nseed = 0
for P in (2, 3):
    for method in METHODS:
        for N in NS:
            m, lo, hi, npar, nsd = band(P, N, method)
            rows.append(f"{P},{N},{method},{m:.10e},{lo:.10e},{hi:.10e}")
            med[(P, method, N)] = m
            nparam, nseed = npar, nsd
csv_text = "\n".join(rows) + "\n"

slopeC = {P: med[(P, "positional", NS[0])] * (NS[0] ** P) for P in (2, 3)}   # C = pos med(Nmin)*Nmin^p
allmed = list(med.values())
ymin = 10.0 ** math.floor(math.log10(min(allmed)))
ymax = 10.0 ** math.ceil(math.log10(max(allmed)))

print(f"advdiff convergence: band over {nparam} params x {nseed} seeds (median-over-seeds then spread-over-params)")
for P in (2, 3):
    pos = [f"{med[(P, 'positional', N)]:.2e}" for N in NS]
    uni = [f"{med[(P, 'uniform', N)]:.2e}" for N in NS]
    print(f"  p{P}: positional med {pos}  uniform med {uni}")
    print(f"       ref slope C={slopeC[P]:.6e} * N^(-{P})")
print(f"  shared y-axis: [{ymin:.0e}, {ymax:.0e}]")

# ---- write into each convergence .tex (block + slope + ymin/ymax) ----
for P in (2, 3):
    path = os.path.join(FIG, f"P5_convergence_p{P}.tex")
    tex = open(path).read()
    bm = BLK.search(tex)
    tex = tex[:bm.start()] + bm.group(1) + bm.group(2) + bm.group(3) + csv_text + bm.group(5) + tex[bm.end():]
    # reference slope: {C*x^(-k)} -> keep k=P, update C
    tex = re.sub(r"\{\s*[\d.eE+-]+\s*\*\s*x\^\(-\d\)",
                 lambda m: "{%.6e*x^(-%d)" % (slopeC[P], P), tex, count=1)
    # shared y-axis
    tex = re.sub(r"(?m)^(\s*)ymin=[^\n]*", lambda m: f"{m.group(1)}ymin={ymin:.0e},", tex, count=1)
    tex = re.sub(r"(?m)^(\s*)ymax=[^\n]*", lambda m: f"{m.group(1)}ymax={ymax:.0e},", tex, count=1)
    open(path, "w").write(tex)
    print(f"  wrote {path}")
