import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alg1_lshape_Tscan import REPO, split_instances


def summarize(root, variants):
    expected = set(split_instances("val", 60))
    rows = []
    for v in variants:
        for p in (2, 3):
            out = root / v / f"p{p}"
            stopped = out / "stopped_run.json"
            if stopped.exists():
                info = json.loads(stopped.read_text())
                jobs = pd.read_csv(root / "pilot_jobs.psv", sep="|", header=None,
                                   names=["job", "state", "exitcode", "seconds", "cpus", "node"])
                job = jobs[jobs.job == info["slurm_job"]].squeeze()
                last = out / f"checkpoints/seed0/checkpoint_N{info['last_completed_level']}.npz"
                if (not job.state.startswith("CANCELLED") or (out / "validation.csv").exists()
                        or hashlib.sha256(last.read_bytes()).hexdigest() != info["last_checkpoint_sha256"]):
                    raise ValueError(f"inconsistent stopped-run record for {v} p={p}")
                rows.append(dict(variant=v, p=p, status="cancelled", n_val=0, cancelled_wall_s=job.seconds))
                continue
            train = REPO / f"data_results/lshape/p{p}" if v == "V0" else out
            ckpt = train / "checkpoints/seed0/checkpoint_final.npz"
            digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
            d = pd.read_csv(out / "validation.csv", float_precision="round_trip")
            levels = list(sorted(d.N.unique()))
            if levels != [32,64] or len(d) != 120 or not (
                (d.split == "val") & (d.p == p) & (d.variant == v) & (d.checkpoint_sha256 == digest)
            ).all():
                raise ValueError(f"inconsistent {v} p={p} validation rows")
            if not np.isfinite(d[["err","gap","gamma_loc"]]).all().all() or not (d.err > 0).all():
                raise ValueError(f"invalid error or mesh ratio in {out}")
            meta = json.loads((train / "meta_train_lshape_seed0.json").read_text())
            if meta["smoke"] or (meta["seed"],meta["n_train"],meta["n_val"]) != (0,280,60):
                raise ValueError(f"invalid training protocol for {v} p={p}")
            row = dict(variant=v, p=p, status="completed", n_val=60, train_s=meta["elapsed_sec"],
                       pilot_retraining_s=0 if v == "V0" else meta["elapsed_sec"])
            for n,g in d.groupby("N"):
                if set(zip(g.sigma1,g.sigma2)) != expected:
                    raise ValueError(f"missing validation parameters at {n}")
                row[f"err_{n}"] = g.err.median()
                row[f"gamma_{n}"] = g.gamma_loc.median()
                row[f"gamma_max_{n}"] = g.gamma_loc.max()
                row[f"below_resolution_{n}"] = int((g.gap <= 1e-11).sum())
            row["order_32_64"] = np.log2(row["err_32"]/row["err_64"])
            rows.append(row)
    return pd.DataFrame(rows)


def project_training_cost(root):
    rows = []
    for p in (2, 3):
        baseline = REPO / f"data_results_v2/icelake/timing/lshape/p{p}/meta_train_lshape_seed0.json"
        probe = root / f"timing/V3/p{p}/meta_train_lshape_seed0.json"
        past = json.loads(baseline.read_text())
        measured = json.loads(probe.read_text())
        if (measured["epochs_used"] != {"64": 2} or measured["batch_size"] != 16
                or measured["n_train"] != 280 or measured["n_val"] != 60):
            raise ValueError("timing requires two full-split epochs with the production batch size")
        cold, warm = measured["epoch_seconds"]["64"]
        if not (np.isfinite([cold, warm]).all() and min(cold, warm) > 0):
            raise ValueError("invalid epoch durations")
        prefix = past["elapsed_sec"]
        setup = measured["elapsed_sec"] - cold - warm
        epochs = past["epochs_used"]["32"]
        cap = past["epochs_per_level"]["32"]
        def hours(n):
            return (prefix + setup + cold + (n-1)*warm) / 3600
        rows.append(dict(p=p, chip=measured["hardware"]["cpu_model"], prefix_proxy_s=prefix,
                         setup_s=setup, cold_epoch_s=cold, warm_epoch_s=warm, peak_rss_mb=measured["peak_rss_mb"],
                         epoch_proxy=epochs, estimate_h=hours(epochs), cap_epochs=cap, cap_scenario_h=hours(cap)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "data_results_v3/lshape_pilot")
    ap.add_argument("--variants", nargs="+", default=["V0","V1","V2","V3"], choices=["V0","V1","V2","V3"])
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    if args.timing:
        table = project_training_cost(args.root)
        table.to_csv(args.root / "timing_projection.csv", index=False)
        print(table.to_string(index=False))
        return
    summary = summarize(args.root, args.variants)
    tag = "" if len(set(args.variants)) == 4 else "_" + "_".join(args.variants)
    summary.to_csv(args.root / f"validation_summary{tag}.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
