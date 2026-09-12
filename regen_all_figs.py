#!/usr/bin/env python3
"""Regenerate ALL Helmholtz-style TikZ figures of IGA_PINN_REPOSITORY/TiKz_overleaf/
with two changes, from the official synced data (no retraining):

  Change 1 -- convergence band over SEEDS (not the physical parameter). For each
    (N, method): per-seed value = median of the error over that seed's parameter
    test set; band median/p25/p75 over the 4 per-seed values -> the narrow
    inter-seed reproducibility band. Only the numbers in the embedded band block
    change (median series + reference slope recomputed; they barely move), and one
    legend entry labels the band as inter-seed. The uniform band is zero-width
    (uniform Galerkin is seed-independent).

  Change 2 -- loss train/val PER LEVEL N. Where the history carries a level_N
    column (helmholtz, arctan, advdiff) one loss figure + CSV per level is written
    (epoch,train,val; val = val_eta2). singular and lshape have a single flat
    history (no level_N -> concat == level), so they get the band change only.

Per-experiment error column (matched to the current band median within ~1%):
  singular H1_rel | helmholtz h1_rel_error | arctan H1_seminorm_rel |
  lshape h1_rel_error_iga_ref | advdiff H1_semi_rel.

Non-loss/non-convergence figures are not touched. Run: python3 regen_all_figs.py
"""
import csv, glob, io, json, os, re
from collections import defaultdict, OrderedDict
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
TO = os.path.join(ROOT, "TiKz_overleaf")

EXPS = OrderedDict([
 ("singular",  dict(folder="exp1_singular", sub="05_convergence", conv="P1_convergence_p{P}.tex",
                    slug="singular",  err="H1_rel")),
 ("helmholtz", dict(folder="exp2_helmholtz", sub="", conv="P2_convergence_p{P}.tex",
                    slug="helmholtz", err="h1_rel_error")),
 ("arctan",    dict(folder="exp3_arctan", sub="", conv="P3_convergence_p{P}.tex",
                    slug="arctan",    err="H1_seminorm_rel")),
 ("lshape",    dict(folder="exp4_lshape", sub="", conv="P4_convergence_p{P}.tex",
                    slug="lshape",    err="h1_rel_error_iga_ref")),
 ("advdiff",   dict(folder="exp5_advdiff", sub="", conv="P5_convergence_p{P}.tex",
                    slug="advdiff",   err="H1_semi_rel")),
])
BLK = re.compile(r"(\\begin\{filecontents\*\}(?:\[[^\]]*\])?\{)([^}]+)(\}\s*\n)(.*?)(\\end\{filecontents\*\})", re.S)


def figdir(exp):
    c = EXPS[exp]
    return os.path.join(TO, c["folder"], c["sub"]) if c["sub"] else os.path.join(TO, c["folder"])


def evaldir(exp, P):
    return os.path.join(ROOT, "data_results", EXPS[exp]["slug"], f"p{P}", "eval")


def histpath(exp, P):
    return os.path.join(ROOT, "data_results", EXPS[exp]["slug"], f"p{P}", "history", f"history_p{P}_seed{{s}}.csv")


# --------------------------- Change 1: seed band ----------------------------
def seed_band(exp, P, N, method):
    err = EXPS[exp]["err"]; by = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(evaldir(exp, P), "summary_*seed*.csv"))):
        for r in csv.DictReader(open(f)):
            try:
                if int(round(float(r["N"]))) == N and r["method"] == method and r.get(err, "") != "":
                    e = float(r[err])
                    if np.isfinite(e):
                        by[r.get("seed", "0")].append(e)
            except (KeyError, ValueError):
                continue
    ps = [float(np.median(v)) for v in by.values() if v]
    if not ps:
        return None
    return (float(np.median(ps)), float(np.percentile(ps, 25)), float(np.percentile(ps, 75)))


def regen_conv(exp, P):
    path = os.path.join(figdir(exp), EXPS[exp]["conv"].format(P=P))
    tex = open(path).read(); bm = BLK.search(tex); csvname = bm.group(2)
    lines = bm.group(4).strip().splitlines(); header = lines[0]; body = [header]
    for ln in lines[1:]:
        if not ln.strip():
            continue
        c = ln.split(",")
        if not c[0].strip().isdigit():
            body.append(ln); continue
        p, N, meth = int(c[0]), int(c[1]), c[2]
        sb = seed_band(exp, p, N, meth)
        body.append(ln if sb is None else f"{p},{N},{meth},{sb[0]:.10e},{sb[1]:.10e},{sb[2]:.10e}")
    tex = tex[:bm.start()] + bm.group(1) + csvname + bm.group(3) + "\n".join(body) + "\n" + bm.group(5) + tex[bm.end():]
    open(os.path.join(figdir(exp), csvname), "w").write("\n".join(body) + "\n")

    # reference slope C = uniform seed-median(N_max) * N_max^k (k from the figure).
    sm = re.search(r"\{\s*([\d.eE+-]+)\s*\*\s*x\^\(?-(\d)\)?", tex)
    k = int(sm.group(2)); Nmax = max(int(l.split(",")[1]) for l in body[1:])
    Cnew = seed_band(exp, P, Nmax, "uniform")[0] * (Nmax ** k)
    tex = tex[:sm.start(1)] + ("%.6e" % Cnew) + tex[sm.end(1):]

    # label the shaded band as inter-seed, inserted just before the reference-slope
    # \addplot (so the legend order is uniform, positional, inter-seed, O(N^-k)).
    if "inter-seed" not in tex:
        inject = ("\\addlegendimage{area legend,draw=none,fill=gray,fill opacity=0.18}\n"
                  "\\addlegendentry{shaded: inter-seed $p_{25}$--$p_{75}$}\n\n")
        # function replacement: re.sub does NOT escape-process the returned string
        # (a plain replacement string would turn the \a of \addlegendimage into a bell).
        tex = re.sub(r"(\n)(\\addplot\s*\[gray,dashed)", lambda m: m.group(1) + inject + m.group(2), tex, count=1)
    open(path, "w").write(tex)
    return dict(csvname=csvname, k=k, C=Cnew,
                Ns=sorted({int(l.split(",")[1]) for l in body[1:]}))


# --------------------------- Change 2: per-level loss -----------------------
def level_segments(exp, P):
    """seed0 history -> OrderedDict {N:(train,val)}.

    Preferred: split on the per-epoch `level_N` column (helmholtz, arctan, advdiff).
    Fallback (lshape): the history is one flat stream with NO level_N column, but the
    continuation early-stopped per level and `meta_train_*_seed0.json` records the exact
    `epochs_used` per level -- so the level boundaries are the cumulative epochs_used.
    Returns None only if neither source is available."""
    fp = histpath(exp, P).format(s=0)
    rows = list(csv.DictReader(io.StringIO("".join(l for l in open(fp) if not l.startswith("#")))))
    if not rows:
        return None
    if "level_N" in rows[0]:
        out = OrderedDict()
        for r in rows:
            N = int(r["level_N"]); out.setdefault(N, ([], []))
            out[N][0].append(float(r["train_loss"])); out[N][1].append(float(r["val_eta2"]))
        return out
    # fallback A (lshape): exact boundaries from meta_train epochs_used.
    slug = EXPS[exp]["slug"]
    mp = os.path.join(ROOT, "data_results", slug, f"p{P}", f"meta_train_{slug}_seed0.json")
    if os.path.exists(mp):
        m = json.load(open(mp)); eu = m.get("epochs_used"); lv = m.get("levels")
        if eu and lv:
            counts = [int(eu[str(N)]) for N in lv]
            if sum(counts) == len(rows):   # boundaries line up exactly
                out = OrderedDict(); i = 0
                for N, c in zip(lv, counts):
                    seg = rows[i:i + c]; i += c
                    out[int(N)] = ([float(r["train_loss"]) for r in seg], [float(r["val_eta2"]) for r in seg])
                return out
    # fallback B (singular): no level_N, no meta. The eq-26 val denominator is recomputed
    # per level, so val_eta2 jumps sharply at every level transition -> recover the
    # boundaries from those jumps and map the K segments to the K COARSEST eval N-levels
    # (training runs the coarse levels; the eval grid extrapolates beyond). INFERRED.
    if "val_eta2" not in rows[0]:
        return None
    va = np.array([float(r["val_eta2"]) for r in rows]); jr = va[1:] / np.maximum(va[:-1], 1e-30)
    bnd = [i + 1 for i in range(len(jr)) if jr[i] < 0.6 or jr[i] > 1.6]
    if not bnd:
        return None
    evalNs = sorted({int(round(float(r["N"]))) for f in glob.glob(os.path.join(evaldir(exp, P), "summary_*seed*.csv"))
                     for r in csv.DictReader(open(f))})
    edges = [0] + bnd + [len(rows)]; K = len(edges) - 1
    if not evalNs or K > len(evalNs):
        return None
    out = OrderedDict()
    for (lo, hi), N in zip(zip(edges[:-1], edges[1:]), evalNs[:K]):
        seg = rows[lo:hi]
        out[int(N)] = ([float(r["train_loss"]) for r in seg], [float(r["val_eta2"]) for r in seg])
    return out


def _set(tex, key, val, fmt="%.3e"):
    return re.sub(rf"({key}=\s*)([\d.eE+-]+)(\s*,)", lambda m: m.group(1) + (fmt % val) + m.group(3), tex, count=1)


def regen_loss(exp, P):
    segs = level_segments(exp, P)
    if segs is None:
        return None  # flat history (singular, lshape): no per-level
    # loss figures live in the experiment folder ROOT (the convergence subdir, e.g.
    # singular's 05_convergence/, holds only the convergence figure).
    slug = EXPS[exp]["slug"]; fd = os.path.join(TO, EXPS[exp]["folder"])
    template = open(os.path.join(fd, f"loss_{slug}_p{P}_level.tex")).read()
    info = {}
    for N, (tr, va) in segs.items():
        with open(os.path.join(fd, f"loss_trainval_{slug}_p{P}_N{N}.csv"), "w") as fh:
            fh.write("epoch,train,val\n")
            for i, (t, v) in enumerate(zip(tr, va), start=1):
                fh.write(f"{i},{t:.6e},{v:.6e}\n")
        n = len(tr); rep = max(8, round(n / 12.0)); vals = tr + va
        tex = re.sub(r"(?m)^\\newcommand\{\\csvfile\}\{[^}]*\}",
                     r"\\newcommand{\\csvfile}{" + f"loss_trainval_{slug}_p{P}_N{N}.csv" + "}", template)
        tex = _set(tex, "ymin", min(vals) * 0.85); tex = _set(tex, "ymax", max(vals) * 1.15)
        tex = re.sub(r"mark repeat=\d+", f"mark repeat={rep}", tex)
        open(os.path.join(fd, f"loss_{slug}_p{P}_N{N}.tex"), "w").write(tex)
        info[N] = n
    return info


if __name__ == "__main__":
    # NOTE: the convergence shaded band is INTENTIONALLY the spread over the physical
    # parameter family (the original, visible band) -- per the user's hybrid decision,
    # the inter-seed band was too thin to see (~0.5-3%) so reproducibility is reported
    # separately (make_repro_table.py) instead. regen_conv() (the seed-band rewrite) is
    # therefore NOT called here; re-running it would replace the visible band with the
    # invisible inter-seed one. Only the per-level loss is regenerated below.
    print("### convergence band: left as the (visible) parameter-spread band -- not regenerated ###")
    print("\n### loss per level N (where level_N exists) ###")
    for exp in EXPS:
        for P in (2, 3):
            info = regen_loss(exp, P)
            if info is None:
                print(f"  {exp:9} p{P}: flat history (no level_N) -> band-only, loss unchanged")
            else:
                print(f"  {exp:9} p{P}: per-level epochs {dict(info)} -> loss_{EXPS[exp]['slug']}_p{P}_N*.tex (+CSVs)")
