"""pgfplots sources for Figure R1 (resonance scan) and the new Fig. 6 (N = 34).

Reads the outputs of ``helmholtz_resonance_study.py`` and writes standalone
TikZ files plus their CSV data to ``TiKz_overleaf_v2/exp2_helmholtz/``, in the
style of ``TiKz_overleaf/exp2_helmholtz`` (uniform blue squares, adapted orange).

Usage:
    python 1D/scripts/helmholtz_v2_figures.py --data data_results_v2/helmholtz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "1D")]

import numpy as np
import pandas as pd

PREAMBLE = r"""\documentclass[11pt,tikz,border=2pt]{standalone}
\usepackage{pgfplots}
\usepackage{amsmath}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\pgfplotsset{
table/col sep=comma,
every axis/.append style={
tick label style={font=\footnotesize},
label style={font=\footnotesize},
title style={font=\footnotesize},
legend style={font=\tiny,draw=none,fill=white,fill opacity=0.8,text opacity=1,cells={anchor=west}},
line width=0.9pt,
grid=none
},
/series/uniform/.style={blue,solid,line width=1.0pt},
/series/adapted/.style={orange!90!black,solid,line width=1.0pt},
/series/exact/.style={black,densely dashed,line width=0.8pt}
}
"""


def scan_panels(data, fig):
    scan = pd.read_csv(data / "scan.csv")
    r1 = pd.read_csv(data / "table_R1.csv")
    for (p, N), d in scan.groupby(["p", "N"]):
        d.pivot_table(index="c", columns="method", values="err").reset_index().to_csv(
            fig / f"R1_scan_p{p}_N{N}.csv", index=False, float_format="%.6e")
    plots = []
    for p in (2, 3):
        for N in (32, 64):
            ch = r1[(r1.p == p) & (r1.N == N)].c_h_uniform.to_numpy()
            vlines = "\n".join(
                [rf"\draw[gray,dashed,line width=0.6pt] (axis cs:{c},1e-6) -- (axis cs:{c},1e3);" for c in (4.8, 5.2)]
                + [rf"\draw[blue,dotted,line width=0.8pt] (axis cs:{c:.6f},1e-6) -- (axis cs:{c:.6f},1e3);" for c in ch if 4.5 <= c <= 5.5]
                + [r"\draw[black!60,line width=0.4pt] (axis cs:5.0005,1e-6) -- (axis cs:5.0005,1e3);"])
            legend = ("\\addlegendentry{uniform}\n" if (p, N) == (2, 32) else "")
            legend_a = ("\\addlegendentry{adapted}\n" if (p, N) == (2, 32) else "")
            plots.append(rf"""\nextgroupplot[title={{$p={p}$, $N={N}$}}]
\addplot[/series/uniform] table[x=c,y=uniform]{{R1_scan_p{p}_N{N}.csv}};
{legend}\addplot[/series/adapted] table[x=c,y=adapted]{{R1_scan_p{p}_N{N}.csv}};
{legend_a}{vlines}""")
    body = "\n".join(plots)
    tex = PREAMBLE + rf"""
\begin{{document}}
\begin{{tikzpicture}}
\begin{{groupplot}}[
group style={{group size=2 by 2,horizontal sep=1.6cm,vertical sep=1.4cm,xlabels at=edge bottom,ylabels at=edge left}},
width=0.47\textwidth,height=0.34\textwidth,scale only axis,
xmin=4.5,xmax=5.5,ymode=log,ymin=1e-4,ymax=1e3,
xlabel={{contrast $c$}},ylabel={{$|u^\ast-u_{{(\cdot)}}|_{{H^1}}/|u^\ast|_{{H^1}}$}},
legend pos=north west,clip=true]
{body}
\end{{groupplot}}
\end{{tikzpicture}}
\end{{document}}
"""
    (fig / "R1_resonance_scan.tex").write_text(tex, encoding="utf-8")


def knots_marks(p, N, method, c, fig):
    import jax.numpy as jnp
    import src.helmholtz.mesh_helmholtz_global as G
    from src.config import HELMHOLTZ
    from src.helmholtz.network_helmholtz import knots_from_network_global, params_from_flat_dict
    if method == "uniform":
        kn = np.asarray(G.uniform_two_patch_knots(N, p))
    else:
        ck = REPO / "data_results" / "helmholtz" / f"p{p}" / "checkpoints" / f"p{p}_seed0" / "checkpoint_final.npz"
        kn = np.asarray(knots_from_network_global(params_from_flat_dict(np.load(ck)), jnp.asarray(c), N, p,
                                                  T=float(HELMHOLTZ["T_cap"]), h_min=float(HELMHOLTZ["h_min"]),
                                                  use_hmax=True))
    pd.DataFrame({"x": np.unique(kn)}).to_csv(fig / f"P2_c5_N{N}_knots_{method}_p{p}.csv", index=False, float_format="%.8f")


def solution_panels(data, fig, N, c):
    errs = pd.read_csv(data / "fig6_errors.csv")
    for p in (2, 3):
        prof = pd.read_csv(data / f"fig6_profile_p{p}_N{N}.csv")
        prof.to_csv(fig / f"P2_c5_N{N}_profile_p{p}.csv", index=False, float_format="%.6e")
        ylim = 1.15 * float(np.max(np.abs(prof[["exact", "uniform", "adapted"]].to_numpy())))
        for method, tag in (("uniform", "uniform"), ("adapted", "NN")):
            knots_marks(p, N, method, c, fig)
            e = errs[(errs.p == p) & (errs.N == N) & (errs.method == method)].err.iloc[0]
            name = r"u_h" if method == "uniform" else r"u_\theta"
            tex = PREAMBLE + rf"""
\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[scale only axis,width=0.62\textwidth,height=0.32\textwidth,
xmin=0,xmax=1,ymin={-ylim:.4g},ymax={ylim:.4g},xlabel={{$x$}},ylabel={{$u$}},
legend style={{at={{(0.5,1.03)}},anchor=south}},legend columns=2]
\addplot[/series/exact] table[x=x,y=exact]{{P2_c5_N{N}_profile_p{p}.csv}};
\addlegendentry{{$u^\ast$}}
\addplot[/series/{method}] table[x=x,y={method}]{{P2_c5_N{N}_profile_p{p}.csv}};
\addlegendentry{{${name}$, rel.\ $H^1$ error {e:.2g}}}
\addplot[only marks,mark=|,mark size=3pt,black!70,forget plot] table[x=x,y expr={-0.93 * ylim:.4g}]{{P2_c5_N{N}_knots_{method}_p{p}.csv}};
\draw[gray,line width=0.4pt] (axis cs:0.5,{-ylim:.4g}) -- (axis cs:0.5,{ylim:.4g});
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
            (fig / f"P2_c5_N{N}_{tag}_p{p}.tex").write_text(tex, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_results_v2/helmholtz")
    ap.add_argument("--fig-dir", default="TiKz_overleaf_v2/exp2_helmholtz")
    ap.add_argument("--N", type=int, default=34)
    ap.add_argument("--c", type=float, default=5.0005)
    args = ap.parse_args()
    data, fig = REPO / args.data, REPO / args.fig_dir
    fig.mkdir(parents=True, exist_ok=True)
    scan_panels(data, fig)
    solution_panels(data, fig, args.N, args.c)
    print(f"wrote {sorted(p.name for p in fig.glob('*.tex'))} -> {fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
