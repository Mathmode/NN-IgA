#!/usr/bin/env python3
"""Regenerate Table 5 (advdiff, exp5) — TiKz_overleaf/exp5_advdiff/tables/P5_table.tex —
from the CURRENT data_results/advdiff eval summaries, fixing audit finding D4:

  * SOURCE: data_results/advdiff (not the superseded advdiff_OLD_jun15_mac).
  * I_eff FILTER: the effectivity index is restricted to the diffusion-dominated
    sub-range logeps in [-1.75, -1.5] (paper §5.5), unlike the previous unfiltered table.
  * Aggregation: for each (p, N, method) the median over the parameter family of the
    per-parameter median over the four seeds — identical to the P5_convergence figure, so
    the H1 columns of the table equal the figure (verified). H1 uses the FULL parameter
    set; only I_eff is sub-range filtered.

u_h = uniform mesh, u_theta = positional (network) mesh. Run: python regen_advdiff_table.py
"""
import glob, math, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "TiKz_overleaf", "exp5_advdiff", "tables", "P5_table.tex")
NS = [4, 8, 16, 32, 64]
LOGEPS_LO, LOGEPS_HI = -1.75, -1.5


def load():
    fr = []
    for p in (2, 3):
        for f in sorted(glob.glob(os.path.join(ROOT, "data_results", "advdiff", f"p{p}", "eval", "summary_*seed*.csv"))):
            df = pd.read_csv(f); df["dirp"] = p; fr.append(df)
    return pd.concat(fr, ignore_index=True)


def recipeC(sub, col):
    """median over params of (median over seeds) — matches the convergence figure."""
    return float(sub.groupby(["logeps", "b"])[col].median().median())


def fmt_h1(v):
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10.0 ** e
    if m >= 9.995:            # rounding guard (e.g. 9.999->10.0)
        m /= 10.0; e += 1
    return f"${m:.2f}{{\\times}}10^{{{e}}}$"


def fmt_ieff(v):
    return f"${v:.2f}$"


def main():
    d = load()
    df = d[(d.logeps >= LOGEPS_LO) & (d.logeps <= LOGEPS_HI)]     # filtered subset for I_eff
    n_full = d.groupby("dirp").apply(lambda x: x[["logeps", "b"]].drop_duplicates().shape[0], include_groups=False)
    n_filt = df.groupby("dirp").apply(lambda x: x[["logeps", "b"]].drop_duplicates().shape[0], include_groups=False)

    cells = {}
    for p in (2, 3):
        for N in NS:
            row = {}
            for meth, lab in (("uniform", "uh"), ("positional", "ut")):
                base = d[(d.dirp == p) & (d.N == N) & (d.method == meth)]
                filt = df[(df.dirp == p) & (df.N == N) & (df.method == meth)]
                row[f"H1_{lab}"] = recipeC(base, "H1_semi_rel")       # H1: FULL set
                row[f"Ie_{lab}"] = recipeC(filt, "eff_index")         # I_eff: FILTERED set
            cells[(p, N)] = row

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (r"\caption{\textbf{Experiment 5.} Median relative $H^1$ error "
         r"$|u^\ast-u_{(\cdot)}|_{H^1}/|u^\ast|_{H^1}$ and median effectivity index "
         r"$I_{\mathrm{eff}}=\eta/|u^\ast-u_{(\cdot)}|_{H^1}$ per level $N$, for the uniform "
         r"($u_h$) and adapted ($u_\theta$) meshes, at $p=2$ and $p=3$. Medians are taken "
         r"over the parameter test set (four seeds); $I_{\mathrm{eff}}$ is reported on the "
         r"diffusion-dominated sub-range $\ell_\varepsilon\in[-1.75,-1.5]$.}"),
        r"\label{tab:advdiff_summary}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{c cccc @{\hspace{1.0em}} cccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{$p=2$} & \multicolumn{4}{c}{$p=3$} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"& \multicolumn{2}{c}{$H^1_{\mathrm{rel}}$} & \multicolumn{2}{c}{$I_{\mathrm{eff}}$} & \multicolumn{2}{c}{$H^1_{\mathrm{rel}}$} & \multicolumn{2}{c}{$I_{\mathrm{eff}}$} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r"$N$ & $u_h$ & $u_\theta$ & $u_h$ & $u_\theta$ & $u_h$ & $u_\theta$ & $u_h$ & $u_\theta$ \\",
        r"\midrule",
    ]
    for N in NS:
        c2, c3 = cells[(2, N)], cells[(3, N)]
        lines.append(
            f"{N} & {fmt_h1(c2['H1_uh'])} & {fmt_h1(c2['H1_ut'])} & {fmt_ieff(c2['Ie_uh'])} & {fmt_ieff(c2['Ie_ut'])} & "
            f"{fmt_h1(c3['H1_uh'])} & {fmt_h1(c3['H1_ut'])} & {fmt_ieff(c3['Ie_uh'])} & {fmt_ieff(c3['Ie_ut'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""]

    open(OUT, "w").write("\n".join(lines))
    print(f"params full/filtered: full={dict(n_full)} filtered[-1.75,-1.5]={dict(n_filt)}")
    print(f"wrote {OUT}")
    for N in NS:
        c2, c3 = cells[(2, N)], cells[(3, N)]
        print(f"  N{N:<3} p2 H1[{c2['H1_uh']:.2e},{c2['H1_ut']:.2e}] Ieff[{c2['Ie_uh']:.2f},{c2['Ie_ut']:.2f}] | "
              f"p3 H1[{c3['H1_uh']:.2e},{c3['H1_ut']:.2e}] Ieff[{c3['Ie_uh']:.2f},{c3['Ie_ut']:.2f}]")


if __name__ == "__main__":
    main()
