"""pgfplots convergence figures of Experiment 6 (1D advection-diffusion-reaction).

Reads data_results_v2/adr/p*/summary_p*_seed*.csv and writes
TiKz_overleaf_v2/exp6_adr/P6_convergence_p{2,3}.tex (median and interquartile
band of the relative energy error over seeds and test parameters).

Usage:
    python 1D/scripts/adr_v2_figures.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

TEMPLATE = r"""\documentclass[11pt,tikz,border=2pt]{standalone}
\usepackage{pgfplots}
\usepackage{amsmath}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{fillbetween}
\pgfplotsset{
  discard if not/.style 2 args={x filter/.append code={
    \edef\tempa{\thisrow{#1}}\edef\tempb{#2}\ifx\tempa\tempb\else\def\pgfmathresult{NaN}\fi}},
  filter discard warning=false, unbounded coords=discard,
  table/col sep=comma,
  every axis/.append style={tick label style={font=\footnotesize}, label style={font=\footnotesize},
    legend style={font=\tiny,draw=none,fill=white,fill opacity=0.8,text opacity=1,cells={anchor=west},
      inner sep=2pt,row sep=-1pt,nodes={scale=0.9,transform shape}}, line width=0.9pt, grid=none},
  /series/.is family,
  /series/uniform/.style={blue,mark=square,solid,line width=1.1pt,mark size=2.2pt},
  /series/adapted/.style={orange!90!black,mark=o,solid,line width=1.1pt,mark size=2.6pt},
  /band/.is family,
  /band/uniform/.style={draw=none,fill=blue,fill opacity=0.14},
  /band/adapted/.style={draw=none,fill=orange!90!black,fill opacity=0.14}
}
\begin{filecontents*}[overwrite]{FILE}
DATA
\end{filecontents*}
\begin{document}
\begin{tikzpicture}
\begin{axis}[width=0.65\textwidth,height=0.5\textwidth,xlabel={$N$},
  ylabel={$\|u^\ast-u_{(\cdot)}\|_{\mathcal E}/\|u^\ast\|_{\mathcal E}$},
  xmode=log,log basis x=2,xtick={8,16,32,64,128},xticklabels={8,16,32,64,128},
  xmin=7,xmax=146,ymode=log,ymin=YMIN,ymax=1,legend pos=south west]
BANDS
\addplot[/series/uniform,discard if not={method}{uniform}] table[x=N,y=err_med]{FILE};
\addlegendentry{uniform $u_h$}
\addplot[/series/adapted,discard if not={method}{adapted}] table[x=N,y=err_med]{FILE};
\addlegendentry{adapted $u_\theta$}
\addplot[gray,dashed,line width=1.0pt,domain=7:146,samples=2]{CONST*x^(-P)};
\addlegendentry{$\mathcal{O}(N^{-P})$}
\end{axis}
\end{tikzpicture}
\end{document}
"""

BAND = r"""\addplot[name path=M_lo,draw=none,forget plot,discard if not={method}{M}] table[x=N,y=err_p25]{FILE};
\addplot[name path=M_hi,draw=none,forget plot,discard if not={method}{M}] table[x=N,y=err_p75]{FILE};
\addplot[/band/M,forget plot] fill between[of=M_lo and M_hi];
"""


def main():
    d = pd.concat(pd.read_csv(f) for f in glob.glob(str(REPO / "data_results_v2" / "adr" / "p*" / "summary_p*_seed*.csv")))
    out = REPO / "TiKz_overleaf_v2" / "exp6_adr"
    out.mkdir(parents=True, exist_ok=True)
    for p in (2, 3):
        g = d[d.p == p].groupby(["N", "method"]).err_E
        b = pd.DataFrame({"err_med": g.median(), "err_p25": g.quantile(0.25), "err_p75": g.quantile(0.75)}).reset_index()
        name = f"P6_p{p}.csv"
        data = "N,method,err_med,err_p25,err_p75\n" + "\n".join(
            f"{r.N},{r.method},{r.err_med:.10e},{r.err_p25:.10e},{r.err_p75:.10e}" for r in b.itertuples())
        const = float(b[(b.method == "adapted") & (b.N == 32)].err_med.iloc[0]) * 32.0 ** p
        ymin = 10.0 ** (int(f"{b.err_p25.min():e}".split("e")[1]) - 1)
        bands = "".join(BAND.replace("M", m) for m in ("uniform", "adapted"))
        tex = (TEMPLATE.replace("BANDS", bands).replace("DATA", data).replace("FILE", name)
               .replace("CONST", f"{const:.6e}").replace("YMIN", f"{ymin:g}").replace("P}", f"{p}}}").replace("^(-P)", f"^(-{p})"))
        (out / f"P6_convergence_p{p}.tex").write_text(tex, encoding="utf-8")
    print(f"wrote {sorted(x.name for x in out.glob('*.tex'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
