"""Data layer: reproducible consolidation + labelling of the r-adaptive experiments.

POST-PROCESS ONLY. Reads the outputs the (validated, untouched) training/eval
pipeline already writes, and reorganises them under a clean, labelled,
reproducible `Data/` tree. It never imports or modifies the solver / estimator /
masking / training / PDE code -- it only reads CSV/JSON files and never writes to
`data_results/`.

Experiment numbering (the canonical paper nomenclature):
    1 = singular (1D)   2 = helmholtz (1D)   3 = arctan (2D)
    4 = lshape  (2D)    5 = advdiff   (2D boundary layer)
The problem -> Experiment# map lives ONLY here; degree is read from the `p<p>/`
sub-dir AND the internal CSV `p` column (the `summary_p2_*` filename prefix is a
legacy *problem* tag, NOT the polynomial degree).

Sources (per seed, under <data-root>/<problem>/p<p>[/<problem>_theta0]/):
  * eval CSV : eval/summary_p<p>_seed<s>.csv
  * meta JSON: meta_train_<experiment>_seed<s>.json
  * history  : history/history_p<p>_seed<s>.csv

Output tree (per Experiment x degree, all seeds together):
  Data/[<campaign>/]Experiment<N>_<name>/
    p<p>/
      N<level>/ h1_convergence.csv     # H1 error + eta^2 + eff_index by (seed, method, sigma)
      convergence_summary.csv          # median-over-(seed,sigma) error + RATE per (N, method)
      hyperparameters.json             # shared config + per-seed run info (from meta)
      manifest.json                    # git-hash + command + date + SHA256 of sources + labels
    history/ history_p<p>_seed<s>.csv  # verbatim per-iteration training curves
  Data/index.csv                       # master index

`--campaign` (optional) namespaces the tree (e.g. to keep `arctan` dense vs the
`arctan_fdm` variant separate); without it the tree is flat (Experiment-keyed).

Examples:
  python scripts/consolidate_experiment.py --problem lshape --p 2 3 --all-seeds \
      --init warmstart --data-root /scratch/ecaru/repo_pro/data_results \
      --command "P_DEGREES=3 sbatch scripts/submit_eval_lshape.sbatch"
  python scripts/consolidate_experiment.py --problem arctan_fdm --p 2 3 --all-seeds \
      --campaign fdm --data-root /scratch/ecaru/repo_pro/data_results
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

EXPERIMENT_MAP = {                      # canonical problem -> (number, name)
    "singular": (1, "singular"), "helmholtz": (2, "helmholtz"),
    "arctan": (3, "arctan"), "lshape": (4, "lshape"), "advdiff": (5, "advdiff"),
}
# H1-error column to use as the convergence metric, by experiment. lshape uses the
# DIRECT sigma-weighted seminorm vs the IGA self-reference; the analytic
# experiments use the closed-form relative seminorm. Robust: first non-empty wins.
H1_COL_PRIORITY = {
    "lshape":  ["h1_rel_error_iga_ref", "h1_rel_error_ngsolve", "H1_seminorm_rel"],
    "arctan":  ["H1_seminorm_rel"], "advdiff": ["H1_seminorm_rel"],
}
_H1_FALLBACK = ["h1_rel_error_iga_ref", "H1_seminorm_rel", "h1_rel_error_ngsolve"]
SIGMA_ID_COLS = ["alpha", "s1", "s2", "sigma1", "sigma2"]    # whichever are present
PER_N_COLS = ["h1_rel_error", "eta_squared", "eta", "eta_uniform_squared", "eff_index"]
# Shared (run-constant) hyperparameters carried from the eval CSV when populated.
EVAL_HP_COLS = ["quad_K", "quad_F", "quad_metric", "sigma_dim", "anchor_N", "T_cap",
                "h_min", "experiment_version"]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def resolve_experiment(problem: str):
    """problem may carry a variant suffix (e.g. 'arctan_fdm'); map by canonical prefix."""
    for key, (num, name) in EXPERIMENT_MAP.items():
        if problem == key:
            return num, name, ""
        if problem.startswith(key + "_"):
            return num, name, problem[len(key) + 1:]
    return None, problem, ""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else f"(git error: {out.stderr.strip()[:80]})"
    except Exception as e:  # noqa: BLE001
        return f"(git unavailable: {e!r})"


def query_sacct(job_id: str) -> dict:
    fields = ["JobID", "JobName", "Elapsed", "ElapsedRaw", "AllocCPUS", "ReqMem",
              "Partition", "NodeList", "ExitCode", "State", "Start", "End"]
    try:
        out = subprocess.run(["sacct", "-j", str(job_id), "-P", "-n",
                              "--format=" + ",".join(fields)],
                             capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return {"job_id": job_id, "note": "sacct not available on this host"}
    except Exception as e:  # noqa: BLE001
        return {"job_id": job_id, "note": f"sacct failed: {e!r}"}
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    if not lines:
        return {"job_id": job_id, "note": "job not found in sacct (purged?)"}
    rows = [dict(zip(fields, ln.split("|"))) for ln in lines]
    main = next((r for r in rows if "." not in r["JobID"]), rows[0])
    main["job_id"] = job_id
    return main


def experiment_base_dir(data_root, problem, p, init, override):
    if override is not None:
        return override
    base = data_root / problem / f"p{p}"
    return base / f"{problem}_theta0" if init == "uniform" else base


def pick_h1_col(exp_name, rows):
    cols = rows[0].keys() if rows else []
    for c in H1_COL_PRIORITY.get(exp_name, []) + _H1_FALLBACK:
        if c in cols and any(_f(r.get(c)) is not None for r in rows):
            return c
    return None


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def consolidate(problem, p, seeds, init, campaign, data_root, out_root, repo_root,
                job_id, command, base_override):
    num, name, variant = resolve_experiment(problem)
    exp_label = f"Experiment{num}_{name}" if num else name
    root = (out_root / campaign) if campaign else out_root
    exp_dir, pdir, histdir = root / exp_label, root / exp_label / f"p{p}", root / exp_label / "history"
    pdir.mkdir(parents=True, exist_ok=True)
    histdir.mkdir(parents=True, exist_ok=True)

    base = experiment_base_dir(data_root, problem, p, init, base_override)
    seeds_data, sources, missing = [], {}, []
    for seed in seeds:
        eval_csv = base / "eval" / f"summary_p{p}_seed{seed}.csv"
        meta_json = next((c for c in (base / f"meta_train_{name}_seed{seed}.json",
                                      base / f"meta_train_{problem}_seed{seed}.json") if c.exists()), None)
        hist_csv = base / "history" / f"history_p{p}_seed{seed}.csv"
        rows, meta = [], {}
        if eval_csv.exists():
            with open(eval_csv, newline="") as fh:
                rows = list(csv.DictReader(fh))
            sources[f"eval_seed{seed}"] = {"path": str(eval_csv), "sha256": sha256_of(eval_csv), "n_rows": len(rows)}
        else:
            missing.append(f"eval seed{seed}: {eval_csv}")
        if meta_json:
            meta = json.load(open(meta_json))
            sources[f"meta_seed{seed}"] = {"path": str(meta_json), "sha256": sha256_of(meta_json)}
        else:
            missing.append(f"meta seed{seed}")
        if hist_csv.exists():
            shutil.copyfile(hist_csv, histdir / f"history_p{p}_seed{seed}.csv")
            with open(hist_csv) as fh:
                ndata = sum(1 for ln in fh if ln.strip() and not ln.lstrip().startswith("#")) - 1
            sources[f"history_seed{seed}"] = {"path": str(hist_csv), "sha256": sha256_of(hist_csv), "n_rows": max(ndata, 0)}
        else:
            missing.append(f"history seed{seed}: {hist_csv}")
        if rows or meta:
            seeds_data.append({"seed": seed, "rows": rows, "meta": meta})

    if not seeds_data:
        return {"exp_label": exp_label, "p": p, "out_dir": str(pdir), "missing": missing,
                "seeds_found": [], "note": "no data"}

    all_rows = [(d["seed"], r) for d in seeds_data for r in d["rows"]]
    h1col = pick_h1_col(name, [r for _, r in all_rows])
    sigma_cols = [c for c in SIGMA_ID_COLS if all_rows and c in all_rows[0][1] and
                  any((all_rows[0][1].get(c) not in ("", None)) for _, r in all_rows)]

    # ---- per-N h1_convergence.csv (all seeds, all sigma, by method) ----
    by_N = {}
    for seed, r in all_rows:
        try:
            N = int(_f(r.get("N")))
        except (TypeError, ValueError):
            continue
        by_N.setdefault(N, []).append((seed, r))
    for N, rN in sorted(by_N.items()):
        Ndir = pdir / f"N{N}"
        Ndir.mkdir(parents=True, exist_ok=True)
        hdr = ["seed", "method"] + sigma_cols + ["h1_rel_error"] + \
              [c for c in PER_N_COLS if c != "h1_rel_error"]
        with open(Ndir / "h1_convergence.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=hdr)
            w.writeheader()
            for seed, r in rN:
                row = {"seed": seed, "method": r.get("method", "")}
                for c in sigma_cols:
                    row[c] = r.get(c, "")
                row["h1_rel_error"] = r.get(h1col, "") if h1col else ""
                for c in PER_N_COLS:
                    if c != "h1_rel_error":
                        row[c] = r.get(c, "")
                w.writerow(row)

    # ---- convergence_summary.csv: median over (seed, sigma) + RATE per (N, method) ----
    methods = sorted({r.get("method", "") for _, r in all_rows})
    summary = []
    for method in methods:
        prev_N = prev_err = None
        for N in sorted(by_N):
            rows_Nm = [r for seed, r in by_N[N] if r.get("method", "") == method]
            if not rows_Nm:
                continue
            err = _median([_f(r.get(h1col)) for r in rows_Nm]) if h1col else None
            eta = _median([_f(r.get("eta_squared")) for r in rows_Nm])
            eff = _median([_f(r.get("eff_index")) for r in rows_Nm])
            rate = ""
            if prev_err and err and prev_N and N != prev_N and prev_err > 0 and err > 0:
                rate = round(math.log(prev_err / err) / math.log(N / prev_N), 4)
            summary.append({"N": N, "method": method,
                            "n_points": len(rows_Nm),
                            "h1_rel_error_median": err if err is not None else "",
                            "eta_squared_median": eta if eta is not None else "",
                            "eff_index_median": eff if eff is not None else "",
                            "rate_vs_prevN": rate})
            prev_N, prev_err = N, (err if err else prev_err)
    with open(pdir / "convergence_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["N", "method", "n_points", "h1_rel_error_median",
                                           "eta_squared_median", "eff_index_median", "rate_vs_prevN"])
        w.writeheader(); w.writerows(summary)

    # ---- hyperparameters.json (shared config + per-seed run info, from meta) ----
    m0 = seeds_data[0]["meta"]
    r0 = seeds_data[0]["rows"][0] if seeds_data[0]["rows"] else {}
    shared = {k: m0.get(k) for k in ("levels", "epochs_per_level", "lr_init", "lr_end",
                                     "batch_size", "split_seed", "protocol") if k in m0}
    shared.update({c: r0.get(c) for c in EVAL_HP_COLS if r0.get(c) not in ("", None)})
    per_seed = {}
    for d in seeds_data:
        m = d["meta"]
        per_seed[str(d["seed"])] = {k: m.get(k) for k in
            ("epochs_used", "stop_reasons", "final_loss", "elapsed_sec",
             "best_val_eta2", "best_val_epoch", "n_train", "n_val", "n_test") if k in m}
    json.dump({"_note": "shared = run-constant config (seeds share it); per_seed = "
               "per-run outcome (from meta_train JSON).", "shared": shared,
               "per_seed": per_seed}, open(pdir / "hyperparameters.json", "w"),
              indent=2, default=str)

    # ---- manifest.json (reproducibility) ----
    seeds_found = [d["seed"] for d in seeds_data]
    manifest = {
        "labels": {"experiment": exp_label, "experiment_number": num, "name": name,
                   "variant": variant or None, "campaign": campaign, "p": p,
                   "init": init, "seeds": seeds_found},
        "git_commit": git_hash(repo_root),
        "command": command or "(not provided; pass --command to record the exact launch)",
        "consolidated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "h1_metric_column": h1col,
        "resources": query_sacct(job_id) if job_id else
                     {"note": "no --job-id; SLURM resources not collected"},
        "sources": sources,
        "missing": missing,
    }
    json.dump(manifest, open(pdir / "manifest.json", "w"), indent=2, default=str)

    update_index(out_root, campaign, exp_label, num, variant, p, init, seeds_found,
                 manifest["git_commit"], manifest["consolidated_at"], pdir, missing)
    return {"exp_label": exp_label, "p": p, "out_dir": str(pdir),
            "seeds_found": seeds_found, "missing": missing, "h1col": h1col,
            "levels": sorted(by_N)}


def update_index(out_root, campaign, exp_label, num, variant, p, init, seeds,
                 ghash, date, pdir, missing):
    idx = out_root / "index.csv"
    header = ["campaign", "experiment", "experiment_number", "variant", "p", "init",
              "seeds", "git_commit", "consolidated_at", "path", "complete"]
    rows = []
    if idx.exists():
        with open(idx, newline="") as fh:
            rows = list(csv.DictReader(fh))
    key = (campaign or "", exp_label, str(p), init, variant or "")
    rows = [r for r in rows if (r.get("campaign", ""), r.get("experiment"), r.get("p"),
                                r.get("init"), r.get("variant", "")) != key]
    rows.append({"campaign": campaign or "", "experiment": exp_label,
                 "experiment_number": num or "", "variant": variant or "", "p": p,
                 "init": init, "seeds": " ".join(map(str, seeds)), "git_commit": ghash,
                 "consolidated_at": date, "path": str(pdir),
                 "complete": "yes" if not missing else f"missing:{len(missing)}"})
    rows.sort(key=lambda r: (r.get("campaign", ""), r.get("experiment", ""), str(r.get("p"))))
    out_root.mkdir(parents=True, exist_ok=True)
    with open(idx, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", required=True, help="raw data_results dir: lshape | arctan | arctan_fdm | advdiff | singular | helmholtz")
    ap.add_argument("--p", type=int, nargs="+", required=True)
    ap.add_argument("--seed", type=int, nargs="+", default=None)
    ap.add_argument("--all-seeds", action="store_true", help="seeds 0..3")
    ap.add_argument("--init", choices=["warmstart", "uniform"], default="warmstart")
    ap.add_argument("--campaign", default=None, help="optional namespace (e.g. 'fdm')")
    ap.add_argument("--data-root", default=str(_REPO / "data_results"))
    ap.add_argument("--out-root", default=str(_REPO / "Data"))
    ap.add_argument("--repo-root", default=str(_REPO))
    ap.add_argument("--base-dir", default=None, help="explicit experiment dir (overrides convention)")
    ap.add_argument("--job-id", default=None, help="SLURM job id for sacct resources")
    ap.add_argument("--command", default=None, help="exact launch command, recorded in the manifest")
    args = ap.parse_args()

    seeds = list(range(4)) if args.all_seeds else (args.seed if args.seed is not None else [0])
    data_root, out_root, repo_root = Path(args.data_root), Path(args.out_root), Path(args.repo_root)
    base_override = Path(args.base_dir) if args.base_dir else None
    num, name, variant = resolve_experiment(args.problem)
    exp_label = f"Experiment{num}_{name}" if num else name
    print(f"# consolidate {args.problem} -> {exp_label}"
          f"{' [variant='+variant+']' if variant else ''}  p={args.p} seeds={seeds} init={args.init}")
    print(f"#   data_root={data_root}  out_root={out_root}  campaign={args.campaign or '(flat)'}")
    for p in args.p:
        r = consolidate(args.problem, p, seeds, args.init, args.campaign, data_root,
                        out_root, repo_root, args.job_id, args.command, base_override)
        if not r["seeds_found"]:
            print(f"  [skip] {exp_label}/p{p}: no data under {data_root}")
            continue
        flag = "OK" if not r["missing"] else f"PARTIAL ({len(r['missing'])} missing)"
        print(f"  [{flag}] {exp_label}/p{p}  seeds={r['seeds_found']}  levels={r['levels']}  -> {r['out_dir']}")
        for m in r["missing"]:
            print(f"          missing: {m}")
    print(f"  master index: {out_root / 'index.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
