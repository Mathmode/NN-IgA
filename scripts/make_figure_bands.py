"""Regenerate the convergence-figure shaded bands as POOLED seed x test-parameter
medians / 25-75 IQR (remediation T7, audit D2).

The shipped bands were seed-0 parameter-spread percentiles (audit D2). Decision 1:
the median and the shaded band are pooled over the four seeds AND the held-out test
parameters. This script rewrites, in place, the ``filecontents*`` band block of each
``TiKz_overleaf/exp{k}_*/P{k}_convergence_p{2,3}.tex`` (header
``p,N,method,err_med,err_p25,err_p75``): every row's three numbers are recomputed
from the same eval CSVs / transforms as the tables (scripts/make_tables), preserving
row order and the standalone band-CSV filename. It also re-anchors the reference-
slope constant ``C`` in ``{C*x^(-k)}`` to the pooled uniform median at the largest N.

Deterministic; ``--check`` re-derives without writing. Run after make_tables.

Usage:
    python scripts/make_figure_bands.py            # rewrite all present figures
    python scripts/make_figure_bands.py --check
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import make_tables as MT

TIKZ = REPO / "TiKz_overleaf"

# experiment -> (folder, convergence-tex basename); reuses MT.SPECS loaders.
FILES = {
    1: ("exp1_singular/05_convergence", "P1_convergence_p{P}.tex"),
    2: ("exp2_helmholtz", "P2_convergence_p{P}.tex"),
    3: ("exp3_arctan", "P3_convergence_p{P}.tex"),
    4: ("exp4_lshape", "P4_convergence_p{P}.tex"),
    5: ("exp5_advdiff", "P5_convergence_p{P}.tex"),
}

_BAND = re.compile(
    r"(\\begin\{filecontents\*\}(?:\[[^\]]*\])?\{)([^}]+)(\}\s*\n)(.*?)(\\end\{filecontents\*\})",
    re.S,
)


def _err_rows(k, p):
    """Cache per (k,p): the loaded per-row dataframe with an 'err' column."""
    return MT.SPECS[k]["rows"](p)


def _band(df, N, method):
    d = df[(df.N == N) & (df.method == method)]["err"]
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return None
    return (float(np.median(d)), float(np.percentile(d, 25)), float(np.percentile(d, 75)))


def _rewrite_block(k, body_lines, cache):
    header = body_lines[0]
    out = [header]
    changed = 0
    for ln in body_lines[1:]:
        if not ln.strip():
            continue
        c = ln.split(",")
        if len(c) < 3 or not c[0].strip().isdigit():
            out.append(ln)
            continue
        p, N, method = int(c[0]), int(c[1]), c[2].strip()
        if p not in cache:
            cache[p] = _err_rows(k, p)
        b = _band(cache[p], N, method)
        if b is None:
            out.append(ln)
            continue
        out.append(f"{p},{N},{method},{b[0]:.10e},{b[1]:.10e},{b[2]:.10e}")
        changed += 1
    return out, changed


def _reanchor_slope(tex, cache, k):
    """Set C in each {C*x^(-m)} to pooled uniform-median(Nmax)*Nmax^m."""
    def repl(match):
        C_str, m_str = match.group(1), match.group(2)
        m = int(m_str)
        # anchor to the degree whose data we have; use p=3 if present else p=2.
        p = 3 if 3 in cache and cache[3] is not None else 2
        df = cache[p]
        Ns = sorted(df[df.method == "uniform"]["N"].unique())
        Nmax = int(Ns[-1])
        med = _band(df, Nmax, "uniform")
        if med is None:
            return match.group(0)
        C = med[0] * (Nmax ** m)
        return match.group(0).replace(C_str, f"{C:.6e}")
    return re.sub(r"\{\s*([\d.eE+-]+)\s*\*\s*x\^\(?-(\d)\)?", repl, tex)


def process(k, P, *, check):
    folder, base = FILES[k]
    path = TIKZ / folder / base.format(P=P)
    if not path.exists():
        return None
    tex = path.read_text()
    bm = _BAND.search(tex)
    if bm is None:
        return None
    csvname = bm.group(2)
    body = bm.group(4).strip().splitlines()
    cache = {}
    new_body, changed = _rewrite_block(k, body, cache)
    new_block = bm.group(1) + csvname + bm.group(3) + "\n".join(new_body) + "\n" + bm.group(5)
    new_tex = tex[:bm.start()] + new_block + tex[bm.end():]
    new_tex = _reanchor_slope(new_tex, cache, k)
    if check:
        return ("would-change" if new_tex != tex else "identical", changed, csvname)
    path.write_text(new_tex)
    # standalone band CSV alongside (some figures \input it directly)
    (path.parent / csvname).write_text("\n".join(new_body) + "\n")
    return ("written", changed, csvname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", type=int, nargs="+", default=None)
    args = ap.parse_args()
    ks = args.only or sorted(FILES)
    for k in ks:
        for P in (2, 3):
            try:
                res = process(k, P, check=args.check)
            except FileNotFoundError as e:
                print(f"[skip] Exp {k} p{P}: {e}")
                continue
            if res is None:
                print(f"[skip] Exp {k} p{P}: no convergence tex / band block")
            else:
                status, changed, csvname = res
                print(f"Exp {k} p{P}: {status} ({changed} band rows) -> {csvname}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
