from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from alg1_lshape_Tscan import REPO, reference_energy, realized_hmin, split_instances
from mesh_diagnostics_2d import aspect, gamma_loc, sizes, solve, uniform_knots

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import eigsh

from common.resources import hardware_info
from src.nonparametric.eta_estimator_2d import eta_squared_lshape
from src.nonparametric.lshape.variational_energy import energy_and_load
from src.nonparametric.solver_2d import free_mask_lshape
from src.parametric.continuation import load_checkpoint
from src.parametric.positional_density_network_2d import knots_p3_from_network_axis


def condition_numbers(K, free):
    matrix = csc_matrix(np.asarray(K)[np.ix_(free, free)])
    scaling = diags(1 / np.sqrt(matrix.diagonal()))
    v0 = np.random.default_rng(0).standard_normal(matrix.shape[0])
    def ratio(a):
        if a.shape[0] < 80:
            eig = np.linalg.eigvalsh(a.toarray())
            lo, hi = eig[0], eig[-1]
        else:
            hi = eigsh(a, k=1, which="LA", v0=v0, tol=1e-10, return_eigenvectors=False)[0]
            lo = eigsh(a, k=1, sigma=0.0, which="LM", v0=v0, tol=1e-10, return_eigenvectors=False)[0]
        if lo <= 0 or not np.isfinite(hi / lo):
            raise ValueError("invalid extremal eigenvalues of the free stiffness matrix")
        return float(hi / lo)
    return ratio(matrix), ratio((scaling @ matrix @ scaling).tocsc())


def energy_metrics(coefficients, kx, ky, p, sigma, F, eta2):
    strain, load = energy_and_load(coefficients, kx, ky, p, *sigma)
    gap = float((np.longdouble(F) + strain - 2 * load) / F)
    if not np.isfinite([gap, eta2]).all() or gap < -1e-11 or eta2 < 0:
        raise ValueError(f"invalid energy or estimator: {gap}, {eta2}")
    err = np.sqrt(max(gap, 0))
    eta = np.sqrt(eta2)
    return dict(F_ref=F, strain=float(strain), load=float(load), gap=gap, err=err, eta=eta,
                ieff=eta/(np.sqrt(F)*err) if err > 0 else np.nan, below_resolution=gap <= 1e-11)


def frozen_model(root, p, seed, method, smoke):
    if method == "uniform":
        return None, "uniform"
    folder = root / method / f"p{p}"
    meta = json.loads((folder / f"meta_train_lshape_seed{seed}.json").read_text())
    levels = [4, 8] if smoke else [4, 8, 16, 32]
    if (meta["smoke"] != smoke or meta["levels"] != levels or meta["loss"] != method
            or not meta["level_feature"] or meta["mesh_schedule"] or meta["seed"] != seed
            or meta["protocol"] != "val" or not np.isfinite(meta["final_loss"])):
        raise ValueError(f"unexpected training configuration in {folder}")
    if not smoke and ((meta["n_train"], meta["n_val"], meta["n_test"]) != (280,60,60)
                      or "8362" not in meta["hardware"]["cpu_model"] or meta["hardware"]["slurm_cpus"] != "4"):
        raise ValueError(f"unexpected split or hardware in {folder}")
    ck = folder / "checkpoints" / f"seed{seed}" / "checkpoint_final.npz"
    last = ck.with_name(f"checkpoint_N{levels[-1]}.npz")
    contents = ck.read_bytes()
    if contents != last.read_bytes():
        with np.load(ck) as final, np.load(last) as stage:
            if set(final.files) != set(stage.files) or any(
                not np.array_equal(final[k], stage[k]) for k in final.files
            ):
                raise ValueError("final checkpoint differs from the finest trained level")
    params = load_checkpoint(ck)
    if not params.level_feature or params.mesh_schedule:
        raise ValueError("checkpoint does not implement V1")
    return params, hashlib.sha256(contents).hexdigest()


def evaluate(root, p, seed, method, split):
    smoke = split == "val"
    params, digest = frozen_model(root, p, seed, method, smoke)
    refs = pd.read_csv(REPO / "references_v3/lshape_energy_ngsolve.csv", float_precision="round_trip")
    refs = refs.loc[refs.split == split]
    samples = split_instances(split, 2 if smoke else 60)
    levels = [4, 8] if smoke else [4, 8, 16, 32, 64]
    hw = hardware_info()
    target = root / "observations" / f"{split}_{method}_p{p}_s{seed}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(target, float_precision="round_trip").to_dict("records") if target.exists() else []
    done = {(r["N"], r["sigma1"], r["sigma2"]) for r in rows}
    if len(done) != len(rows) or any(r["checkpoint_sha256"] != digest or r["split"] != split for r in rows):
        raise ValueError(f"invalid previous observations in {target}")
    if not done.issubset({(n,*s) for n in levels for s in samples}):
        raise ValueError(f"unexpected observation keys in {target}")
    for n in levels:
        def forward(sig):
            if params is None:
                x,y = uniform_knots("lshape", p, n)
            else:
                x,y = (knots_p3_from_network_axis(params, sig, n, p, axis) for axis in (0.,1.))
            return x, y, solve("lshape", x, y, p, n, sig)
        if all((n,*s) in done for s in samples):
            continue
        jax.block_until_ready(forward(jnp.array([0.73,1.27])))
        for s in samples:
            if (n,*s) in done:
                continue
            sig = jnp.array(s)
            start = time.perf_counter()
            x,y,res = forward(sig)
            res.u_h.block_until_ready()
            online_s = time.perf_counter() - start
            ne = n+p-1
            e2 = float(eta_squared_lshape(res.u_h, x, y, p, ne, ne, *sig, q_est=4))
            ref = reference_energy(refs, s, split)
            free = np.asarray(free_mask_lshape(x,y,p), dtype=bool).ravel()
            cond, jac = condition_numbers(res.K, free)
            metrics = energy_metrics(res.u_h, x, y, p, s, ref, e2)
            row = dict(split=split, method=method, p=p, seed=seed, N=n, sigma1=s[0], sigma2=s[1],
                       checkpoint_sha256=digest, **metrics, ritz=float(res.ritz_energy),
                       gamma_loc=max(gamma_loc(sizes(x),p),gamma_loc(sizes(y),p)),
                       aspect=aspect(x,y,"lshape"), h_min=realized_hmin(x,y,n), free_dofs=int(free.sum()),
                       cond=cond, cond_jacobi=jac, online_s=online_s, cpu=hw["cpu_model"], cpus=hw["slurm_cpus"], host=hw["host"])
            rows.append(row)
            pd.DataFrame(rows).to_csv(target.with_suffix(".partial"), index=False)
            os.replace(target.with_suffix(".partial"), target)
        print(f"{method} p={p} s={seed} N={n} complete", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--method", choices=("eta","ritz","uniform"), required=True)
    ap.add_argument("--split", choices=("val","test"), required=True)
    ap.add_argument("--p", type=int, choices=(2,3), required=True)
    ap.add_argument("--seed", type=int, choices=range(4), required=True)
    a = ap.parse_args()
    evaluate(a.root, a.p, a.seed, a.method, a.split)
