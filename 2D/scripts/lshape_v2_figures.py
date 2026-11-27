"""pgfplots convergence figures of Experiment 4 with the corrected (energy-identity) error.

Reads data_results_v2/lshape/energy_error_rows.csv and writes
TiKz_overleaf_v2/exp4_lshape/P4_convergence_p{2,3}.tex in the style of
TiKz_overleaf/exp4_lshape (median and interquartile band over seeds and test pairs).

Usage:
    python 2D/scripts/lshape_v2_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "TiKz_overleaf" / "exp4_lshape" / "P4_convergence_p2.tex"


def band_rows(rows):
    g = rows.groupby(["p", "N", "method"]).err_E
    out = pd.DataFrame({"err_med": g.median(), "err_p25": g.quantile(0.25), "err_p75": g.quantile(0.75)}).reset_index()
    return out.sort_values(["p", "method", "N"])


def figure(template, p, band):
    sub = band[band.p == p]
    name = f"P4E_p{p}.csv"
    lines = ["p,N,method,err_med,err_p25,err_p75"] + [
        f"{r.p},{r.N},{r.method},{r.err_med:.10e},{r.err_p25:.10e},{r.err_p75:.10e}" for r in sub.itertuples()]
    head, rest = template.split("\\begin{filecontents*}", 1)
    _, tail = rest.split("\\end{filecontents*}", 1)
    tex = head + "\\begin{filecontents*}[overwrite]{" + name + "}\n" + "\n".join(lines) + "\n\\end{filecontents*}" + tail
    tex = tex.replace("\\newcommand{\\Sfile}{p3_p2.csv}", "\\newcommand{\\Sfile}{" + name + "}")
    tex = tex.replace("discard if not={p}{2}", "discard if not={p}{" + str(p) + "}")
    tex = tex.replace("ylabel={$|u^\\ast-u_{(\\cdot)}|_{H^{1}}/|u^\\ast|_{H^{1}}$}",
                      "ylabel={$\\|u-u_{(\\cdot)}\\|_{\\mathcal E}/\\|u\\|_{\\mathcal E}$}")
    tex = tex.replace("\\addlegendentry{$|u^\\ast-u_h|_{H^1}$}", "\\addlegendentry{uniform $u_h$}")
    tex = tex.replace("\\addlegendentry{$|u^\\ast-u_\\theta|_{H^1}$}", "\\addlegendentry{adapted $u_\\theta$}")
    a = sub[(sub.method == "positional") & (sub.N == 16)].err_med.iloc[0] * 16.0 ** p
    u = sub[(sub.method == "uniform") & (sub.N == 64)].err_med.iloc[0] * 64.0 ** (2.0 / 3.0)
    ref_old = tex[tex.index("\\addplot [gray,dashed"):tex.index("\\end{axis}")]
    ref_new = (f"\\addplot [gray,dashed,line width=1.0pt,domain=3.6:70.4,samples=2]\n  {{{a:.6e}*x^(-{p})}};\n"
               f"\\addlegendentry{{$\\mathcal{{O}}(N^{{-{p}}})$}}\n\n"
               f"\\addplot [gray,dotted,line width=1.0pt,domain=3.6:70.4,samples=2]\n  {{{u:.6e}*x^(-2/3)}};\n"
               f"\\addlegendentry{{$\\mathcal{{O}}(N^{{-2/3}})$}}\n\n")
    tex = tex.replace("xtick={4,8,16,32,64},", "xtick={4,8,16,32,64},\n  xticklabels={4,8,16,32,64},")
    return tex.replace(ref_old, ref_new).replace("ymin=1e-4", "ymin=5e-5")


def main():
    rows = pd.read_csv(REPO / "data_results_v2" / "lshape" / "energy_error_rows.csv")
    band = band_rows(rows)
    out = REPO / "TiKz_overleaf_v2" / "exp4_lshape"
    out.mkdir(parents=True, exist_ok=True)
    template = TEMPLATE.read_text(encoding="utf-8")
    for p in (2, 3):
        (out / f"P4_convergence_p{p}.tex").write_text(figure(template, p, band), encoding="utf-8")
    band.to_csv(out / "P4E_band.csv", index=False)
    print(band.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
