"""Experiment 6 (revision v2): train and evaluate the 1D advection-diffusion-reaction family.

Same protocol as the other experiments: fixed SPLIT_SEED 70/15/15 split of the
parameter samples (shared by all seeds), validation early stopping, coarse-to-fine
continuation 8 -> 64, zero-shot evaluation up to N = 128.

Usage:
    python 1D/scripts/train_adr.py --p 2 --seed 0
    python 1D/scripts/train_adr.py --p 2 --seed 0 --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "1D")]

import jax.numpy as jnp
import numpy as np
import pandas as pd

from common.resources import hardware_info, peak_rss_mb
from src.adr import discretization as D
from src.adr import problem as P
from src.adr.network import from_flat, init_params, logits, to_flat
from src.adr.training import train_continuation
from src.config import SPLIT_SEED, VAL


def evaluate(params, p, levels, test):
    rows = []
    for N in levels:
        for nu in test:
            eps, b = P.unpack(nu)
            for method in ("uniform", "adapted"):
                t0 = time.perf_counter()
                theta = jnp.zeros(N) if method == "uniform" else logits(params, jnp.asarray(nu), N, P.T_CAP)
                kn = D.knots_from_theta(theta, p, P.H_MIN)
                u = D.solve(kn, p, eps, b, D.quad_order(p))
                u.block_until_ready()
                t = time.perf_counter() - t0
                ea, er = D.energy_errors(kn, p, u, eps, b)
                eta = float(jnp.sqrt(D.eta2(kn, p, u, eps, b, D.quad_order(p))))
                h = np.asarray(D.sizes_from_theta(theta, P.H_MIN))
                rows.append(dict(p=p, N=N, log10_eps=float(nu[0]), b=float(nu[1]), method=method,
                                 err_E=float(er), err_E_abs=float(ea), eta=eta, I_eff=eta / float(ea),
                                 gamma_loc=D.gamma_loc(h, p), h_ratio=float(h.max() / h.min()),
                                 h_min=float(h.min()), time_s=t))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, required=True, choices=[2, 3])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=list(P.LEVELS))
    ap.add_argument("--eval-levels", type=int, nargs="+", default=list(P.EVAL_LEVELS))
    ap.add_argument("--n-samples", type=int, default=P.N_SAMPLES)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        VAL.update(max_epochs=8, min_epochs=3, patience=3)
        args.levels, args.eval_levels, args.n_samples = [8, 16], [8, 16, 32], 40

    out = Path(args.output_dir) if args.output_dir else REPO / "data_results_v2" / "adr" / f"p{args.p}"
    out.mkdir(parents=True, exist_ok=True)
    nu = P.sample_parameters(args.n_samples, SPLIT_SEED)
    train, val, test = P.split(nu, SPLIT_SEED)
    np.save(out / "test_nu.npy", test)

    t0 = time.perf_counter()
    params, hist, t_train = train_continuation(init_params(args.seed), args.p, args.levels, train, val, args.seed)
    ck = out / "checkpoints" / f"p{args.p}_seed{args.seed}"
    ck.mkdir(parents=True, exist_ok=True)
    np.savez(ck / "checkpoint_final.npz", **to_flat(params))
    pd.DataFrame(hist).assign(p=args.p, seed=args.seed).to_csv(out / f"history_p{args.p}_seed{args.seed}.csv", index=False)

    t1 = time.perf_counter()
    df = evaluate(from_flat(np.load(ck / "checkpoint_final.npz")), args.p, args.eval_levels, test)
    df.assign(seed=args.seed).to_csv(out / f"summary_p{args.p}_seed{args.seed}.csv", index=False)
    meta = dict(experiment="adr", p=args.p, seed=args.seed, smoke=args.smoke, levels=args.levels,
                eval_levels=args.eval_levels, n_train=len(train), n_val=len(val), n_test=len(test),
                split_seed=SPLIT_SEED, T=P.T_CAP, h_min=P.H_MIN, training_elapsed_sec=t_train,
                eval_elapsed_sec=time.perf_counter() - t1, elapsed_sec=time.perf_counter() - t0,
                peak_rss_mb=float(peak_rss_mb()), hardware=hardware_info())
    (out / f"metadata_p{args.p}_seed{args.seed}.json").write_text(json.dumps(meta, indent=2))
    med = df.groupby(["N", "method"]).err_E.median().unstack()
    med["ratio"] = med.uniform / med.adapted
    print(med.to_string(float_format=lambda v: f"{v:.3g}"))
    print(f"=== DONE train={t_train:.0f}s total={time.perf_counter() - t0:.0f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
