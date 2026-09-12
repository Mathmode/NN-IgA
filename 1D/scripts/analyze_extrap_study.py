#!/usr/bin/env python3
"""Compare the three arms of the contrast-EXTRAPOLATION study on the SHARED high-c
test set (c >= c_split). METRICS ONLY -- no verdicts, no thresholds.

  A  interpolation : production network, re-evaluated on the high-c set
                     (data_results/helmholtz_extrap_evalA/.../eval/, method=positional)
  B  extrapolation : network trained flat on low-c only
                     (data_results/helmholtz_extrap/.../eval/, method=positional)
  C  uniform floor : Galerkin on the uniform mesh (method=uniform in either eval)

The two evals share the SAME high-c contrast file, so A/B/C are compared on identical
contrasts at identical N levels (the script asserts the contrast sets coincide -- the
md5-equivalent check). Reports, per degree and N (median over seeds):
  - median H1 rel error (+ p25/p75) of A, B, C;
  - ratios  B/A (extrapolation penalty vs interpolation),
            C/B and C/A (how far A and B beat the uniform floor);
  - per-contrast-bin medians, to see whether B degrades with distance from c_split.

Resonance note: a few high-c contrasts sit near discrete resonances where ALL arms
saturate (H1 ~ 1.0). Medians are robust to these; the count is reported.

Usage:
    python scripts/analyze_extrap_study.py
    python scripts/analyze_extrap_study.py --c-split 3.0 --p 2 3 --seeds 0 1 2 3
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

EVAL_COLS = {"p", "seed", "N", "c", "method", "h1_rel_error"}


def load_summary(path: Path):
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or not (EVAL_COLS <= set(rows[0].keys())):
        return None
    return rows


def collect(data_root: Path, subdir: str, p: int, seeds, method: str):
    """Return {(N, c_rounded): [h1 over seeds]} for the requested method."""
    acc = defaultdict(list)
    found = 0
    for s in seeds:
        path = data_root / subdir / f"p{p}" / "eval" / f"summary_p5c_seed{s}.csv"
        rows = load_summary(path)
        if rows is None:
            continue
        found += 1
        for r in rows:
            if r["method"] != method:
                continue
            try:
                N = int(float(r["N"])); c = round(float(r["c"]), 4); h = float(r["h1_rel_error"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(h):
                acc[(N, c)].append(h)
    return acc, found


def med(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  --  "
    return f"{x:.{nd}e}" if (abs(x) < 1e-2 or abs(x) >= 1e3) else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Contrast-extrapolation A/B/C comparison (no verdicts).")
    here = Path(__file__).resolve().parent
    ap.add_argument("--data-root", type=Path, default=here.parent.parent / "data_results")
    ap.add_argument("--p", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--c-split", type=float, default=3.0)
    ap.add_argument("--resonance-hi", type=float, default=0.9,
                    help="H1 >= this is treated as a saturated/near-resonant contrast (reported).")
    args = ap.parse_args()

    print("=== contrast-extrapolation study: A (interp) / B (extrap) / C (uniform) ===")
    print(f"    high-c test set: c >= c_split = {args.c_split}; metrics only, NO verdicts.")
    print()

    for p in args.p:
        A, fa = collect(args.data_root, "helmholtz_extrap_evalA", p, args.seeds, "positional")
        B, fb = collect(args.data_root, "helmholtz_extrap", p, args.seeds, "positional")
        C, fc = collect(args.data_root, "helmholtz_extrap_evalA", p, args.seeds, "uniform")
        Cb, _ = collect(args.data_root, "helmholtz_extrap", p, args.seeds, "uniform")
        print(f"================ degree p={p}  (A:{fa} seeds, B:{fb} seeds eval CSVs found) ================")
        if fa == 0 or fb == 0:
            print("   MISSING eval CSVs for A and/or B -- run the cluster jobs first.\n")
            continue

        # Same-test-set check (md5-equivalent): the contrast sets of A and B must coincide.
        cA = sorted({c for (_N, c) in A}); cB = sorted({c for (_N, c) in B})
        same = (cA == cB)
        print(f"   same high-c contrast set across A and B? {'YES' if same else 'NO -- WARNING, not comparable'}"
              f"  ({len(cA)} contrasts)")
        # C from A vs C from B (uniform is network-independent -> should match).
        if Cb:
            dC = [abs(med(C[k]) - med(Cb[k])) for k in set(C) & set(Cb)]
            print(f"   uniform floor identical across the two evals? max|dC|={max(dC):.2e}"
                  if dC else "   (uniform cross-check unavailable)")

        Ns = sorted({N for (N, _c) in A})
        print(f"   {'N':>4} | {'medA':>10} {'medB':>10} {'medC':>10} | {'B/A':>6} {'C/B':>6} {'C/A':>6} | reson(A/B/C)")
        print("   " + "-" * 78)
        for N in Ns:
            cs = sorted({c for (NN, c) in A if NN == N and c >= args.c_split})
            aL = [med(A[(N, c)]) for c in cs]
            bL = [med(B[(N, c)]) for c in cs if (N, c) in B]
            cL = [med(C[(N, c)]) for c in cs if (N, c) in C]
            mA, mB, mC = med(aL), med(bL), med(cL)
            rBA = mB / mA if (mA and math.isfinite(mA) and mA > 0) else float("nan")
            rCB = mC / mB if (mB and math.isfinite(mB) and mB > 0) else float("nan")
            rCA = mC / mA if (mA and math.isfinite(mA) and mA > 0) else float("nan")
            rA = sum(x >= args.resonance_hi for x in aL); rB = sum(x >= args.resonance_hi for x in bL)
            rC = sum(x >= args.resonance_hi for x in cL)
            print(f"   {N:>4} | {fmt(mA):>10} {fmt(mB):>10} {fmt(mC):>10} | "
                  f"{fmt(rBA,2):>6} {fmt(rCB,2):>6} {fmt(rCA,2):>6} | {rA}/{rB}/{rC}")

        # Per-contrast-bin degradation at the finest N (does B worsen away from c_split?).
        Nf = Ns[-1]
        print(f"\n   per-contrast-bin medians at N={Nf} (does B degrade with distance from c_split?):")
        print(f"   {'c-bin':>10} | {'medA':>10} {'medB':>10} {'medC':>10} | {'B/A':>6} | n_c")
        edges = np.linspace(args.c_split, 6.0, 4)   # 3 bins across the high-c range
        for lo, hi in zip(edges[:-1], edges[1:]):
            cs = sorted({c for (N, c) in A if N == Nf and lo <= c < hi + (1e-9 if hi == edges[-1] else 0)})
            if not cs:
                continue
            aL = [med(A[(Nf, c)]) for c in cs]
            bL = [med(B[(Nf, c)]) for c in cs if (Nf, c) in B]
            cL = [med(C[(Nf, c)]) for c in cs if (Nf, c) in C]
            mA, mB, mC = med(aL), med(bL), med(cL)
            rBA = mB / mA if (mA and mA > 0 and math.isfinite(mA)) else float("nan")
            print(f"   [{lo:>4.2f},{hi:>4.2f}) | {fmt(mA):>10} {fmt(mB):>10} {fmt(mC):>10} | {fmt(rBA,2):>6} | {len(cs)}")
        print()

    print("Reading guide (NOT a verdict -- the numbers decide):")
    print("  * B ~ A  and both << C  -> extrapolation works; the c-parametrization is smooth.")
    print("  * B -> C (B loses to A, approaches the uniform floor) -> extrapolation fails;")
    print("    seeing high contrast (the curriculum) is necessary.")
    print("  * B between A and C -> partial generalization; the per-bin table quantifies the reach.")
    print("Caveats: medians over seeds on the shared high-c set; near-resonant contrasts")
    print("(reson counts) saturate all arms and are excluded from the medians by robustness, not by filtering.")


if __name__ == "__main__":
    main()
