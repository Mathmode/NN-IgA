#!/usr/bin/env python3
"""Regenerate the Helmholtz (Experiment 2) TikZ figures in IGA_PINN_REPOSITORY with two
changes, from the official synced data (no retraining). Reads
`data_results/helmholtz/`; edits `TiKz_overleaf/exp2_helmholtz/` in place.

  Change 1 -- convergence band over SEEDS (not contrasts). For each (N, method):
    per-seed value = median of h1_rel_error over that seed's contrasts; the band
    median/p25/p75 are then over the 4 per-seed values. The old band percentiled over
    the ~67 contrasts (wide ~1.2); this is the narrow inter-seed band. Only the numbers
    in the embedded P5_conv_band block change; the median series + reference slope are
    recomputed (medians barely move) and a legend entry labels the band as inter-seed.
    The uniform band is zero-width (uniform Galerkin is seed-independent).

  Change 2 -- loss train/val PER LEVEL N (64,96,128). Splits the seed0 history by
    level_N and writes one loss_trainval_helmholtz_p{P}_N{N}.csv per level (epoch,
    train,val; val = val_eta2) plus a loss figure per level
    (loss_helmholtz_p{P}_N{N}.tex), in the existing TiKz_overleaf loss style (based on
    the single-level `_level` template). The existing `_concat` and `_level` figures are
    left in place (names unchanged).

Non-loss / non-convergence figures (solutions, resonance, tables) are NOT touched.

Run:  python3 regen_helmholtz_figs.py
"""
import csv, glob, io, os, re
from collections import defaultdict, OrderedDict
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data_results/helmholtz")
FIG = os.path.join(ROOT, "TiKz_overleaf/exp2_helmholtz")
LEVELS = (64, 96, 128)


# --------------------------- Change 1: seed band ----------------------------
def seed_band(p, N, method):
    by_seed = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(DATA, f"p{p}/eval/summary_p5c_seed*.csv"))):
        for r in csv.DictReader(open(f)):
            if int(round(float(r["N"]))) == N and r["method"] == method:
                try:
                    e = float(r["h1_rel_error"])
                except (KeyError, ValueError):
                    continue
                if np.isfinite(e):
                    by_seed[r["seed"]].append(e)
    per_seed = [float(np.median(v)) for v in by_seed.values() if v]
    if not per_seed:
        return None
    return (float(np.median(per_seed)), float(np.percentile(per_seed, 25)),
            float(np.percentile(per_seed, 75)))


BAND_BLOCK = re.compile(
    r"(\\begin\{filecontents\*\}\[overwrite\]\{P5_conv_band_p\d\.csv\}\s*\n)(.*?)(\\end\{filecontents\*\})", re.S)


def regen_conv(P):
    path = os.path.join(FIG, f"P2_convergence_p{P}.tex")
    tex = open(path).read()
    bm = BAND_BLOCK.search(tex)
    lines = bm.group(2).strip().splitlines()
    header = lines[0]; body = [header]
    for ln in lines[1:]:
        if not ln.strip():
            continue
        c = ln.split(",")
        p, N, meth = int(c[0]), int(c[1]), c[2]
        sb = seed_band(p, N, meth)
        body.append(ln if sb is None else f"{p},{N},{meth},{sb[0]:.6e},{sb[1]:.6e},{sb[2]:.6e}")
    tex = tex[:bm.start()] + bm.group(1) + "\n".join(body) + "\n" + bm.group(3) + tex[bm.end():]
    open(os.path.join(FIG, f"P5_conv_band_p{P}.csv"), "w").write("\n".join(body) + "\n")

    # reference slope C = uniform seed-median(N_max) * N_max^k (unchanged in practice).
    sm = re.search(r"\{\s*([\d.eE+-]+)\s*\*\s*x\^\(?-(\d)\)?", tex)
    k = int(sm.group(2)); Nmax = max(int(l.split(",")[1]) for l in body[1:])
    Cnew = seed_band(P, Nmax, "uniform")[0] * (Nmax ** k)
    tex = tex[:sm.start(1)] + ("%.6e" % Cnew) + tex[sm.end(1):]

    # label the shaded band as inter-seed (once).
    if "inter-seed" not in tex:
        anchor = "\\addlegendentry{$|u^\\ast-u_\\theta|_{H^1}$}\n"
        tex = tex.replace(anchor, anchor +
                          "\n\\addlegendimage{area legend,draw=none,fill=gray,fill opacity=0.18}\n"
                          "\\addlegendentry{shaded: inter-seed $p_{25}$--$p_{75}$}\n", 1)
    open(path, "w").write(tex)
    return dict(Ns=sorted({int(l.split(",")[1]) for l in body[1:]}), C=Cnew, k=k)


# --------------------------- Change 2: per-level loss -----------------------
def level_segments(P):
    fp = os.path.join(DATA, f"p{P}/history/history_p{P}_seed0.csv")
    rows = list(csv.DictReader(io.StringIO("".join(l for l in open(fp) if not l.startswith("#")))))
    out = OrderedDict()
    for r in rows:
        N = int(r["level_N"]); out.setdefault(N, ([], []))
        out[N][0].append(float(r["train_loss"]))
        out[N][1].append(float(r["val_eta2"]))
    return out


def write_level_csv(P, N, tr, va):
    with open(os.path.join(FIG, f"loss_trainval_helmholtz_p{P}_N{N}.csv"), "w") as fh:
        fh.write("epoch,train,val\n")
        for i, (t, v) in enumerate(zip(tr, va), start=1):
            fh.write(f"{i},{t:.6e},{v:.6e}\n")


def _set(tex, key, val, fmt="%.3e"):
    return re.sub(rf"({key}=\s*)([\d.eE+-]+)(\s*,)", lambda m: m.group(1) + (fmt % val) + m.group(3), tex, count=1)


def make_loss_tex(template, P, N, tr, va):
    tex = re.sub(r"(?m)^\\newcommand\{\\csvfile\}\{[^}]*\}",
                 r"\\newcommand{\\csvfile}{" + f"loss_trainval_helmholtz_p{P}_N{N}.csv" + "}", template)
    n = len(tr); rep = max(8, round(n / 12.0)); vals = tr + va
    tex = _set(tex, "ymin", min(vals) * 0.85)
    tex = _set(tex, "ymax", max(vals) * 1.15)
    tex = re.sub(r"mark repeat=\d+", f"mark repeat={rep}", tex)
    return tex


def regen_loss(P):
    segs = level_segments(P)
    template = open(os.path.join(FIG, f"loss_helmholtz_p{P}_level.tex")).read()
    info = {}
    for N, (tr, va) in segs.items():
        write_level_csv(P, N, tr, va)
        open(os.path.join(FIG, f"loss_helmholtz_p{P}_N{N}.tex"), "w").write(make_loss_tex(template, P, N, tr, va))
        info[N] = len(tr)
    return info


if __name__ == "__main__":
    print("### CHANGE 1 -- convergence band over SEEDS (TiKz_overleaf) ###")
    for P in (2, 3):
        d = regen_conv(P)
        wp = seed_band(P, 128, "positional"); wu = seed_band(P, 128, "uniform")
        print(f"  p{P}: Ns={d['Ns']} C={d['C']:.6e} (k={d['k']}); relwidth positional N128="
              f"{(wp[2]-wp[1])/wp[0]:.3f}, uniform N128={(wu[2]-wu[1])/wu[0]:.3f} (0=seed-indep)")
    print("\n### CHANGE 2 -- loss per level N (TiKz_overleaf) ###")
    for P in (2, 3):
        info = regen_loss(P)
        print(f"  p{P}: per-level epochs {dict(info)}; figures loss_helmholtz_p{P}_N{{64,96,128}}.tex "
              f"(+ CSVs); existing _concat/_level left untouched")
