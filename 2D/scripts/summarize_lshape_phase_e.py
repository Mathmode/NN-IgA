from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from alg1_lshape_Tscan import REPO, reference_energy, split_instances
from lshape_phase_e_eval import frozen_model
from lshape_v2_figures import band_rows

sys.path.insert(0, str(REPO / "scripts"))
from fixed_accuracy import needed_level


LEVELS = (4, 8, 16, 32, 64)
LOSSES = ("eta", "ritz")
KEYS = ["N", "sigma1", "sigma2"]


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def validate_rows(d, samples, refs, p, method):
    expected = {(n, *s) for n in LEVELS for s in samples}
    keys = set(d[KEYS].itertuples(index=False, name=None))
    if len(d) != len(expected) or keys != expected:
        raise ValueError(f"incomplete or duplicated test rows for {method} p={p}")
    if not ((d.split == "test") & (d.p == p) & (d.method == method)
            & d.cpu.str.contains("Platinum 8362 ", regex=False) & (d.cpus == 4)).all():
        raise ValueError(f"wrong split, hardware or method for {method} p={p}")
    positive = ["F_ref", "strain", "load", "eta", "gamma_loc", "aspect", "h_min", "cond", "cond_jacobi"]
    if not (np.isfinite(d[positive]) & (d[positive] > 0)).all().all():
        raise ValueError("invalid energy, mesh or stiffness diagnostic")
    np.testing.assert_array_equal(d.F_ref, [refs[(x, y)] for x, y in zip(d.sigma1, d.sigma2)])
    np.testing.assert_array_equal(d.free_dofs, (d.N+2*p-3)**2 - (d.N//2+p-1)**2)
    if (d.gap < -1e-11).any() or (d.h_min < 1e-7-5e-16).any():
        raise ValueError("gap or minimum element width outside the V1 tolerances")
    np.testing.assert_array_equal(d.below_resolution, d.gap <= 1e-11)
    np.testing.assert_allclose(d.gap, (d.F_ref + d.strain - 2*d.load)/d.F_ref, rtol=1e-13, atol=1.2e-15)
    np.testing.assert_allclose(d.err, np.sqrt(d.gap.clip(lower=0)), rtol=1e-14, atol=0)
    np.testing.assert_allclose(d.ieff, d.eta/(np.sqrt(d.F_ref)*d.err).replace(0, np.nan), rtol=1e-14)
    cross_limit = block_limits(d.N)[1]
    if ((d.gamma_loc < 1-1e-12) | (d.gamma_loc > cross_limit*(1+2e-9))).any():
        raise ValueError("sliding-window mesh ratio outside the two-block bound")


def block_limits(n):
    k = n/2
    r = np.exp(10)
    f = 1e-7
    b = 0.5-k*f
    return r, (f+b*r/(r+k-1))/(f+b/(1+(k-1)*r))


def observations(root):
    samples = split_instances("test", 60)
    paired = split_instances("test", 16)
    reference = read(REPO / "references_v3/lshape_energy_ngsolve.csv")
    refs = {s: reference_energy(reference, s, "test") for s in samples}
    nets, algorithms, trained = [], [], []
    for p in (2, 3):
        for method in (*LOSSES, "uniform"):
            for seed in (range(4) if method in LOSSES else (0,)):
                d = read(root / "observations" / f"test_{method}_p{p}_s{seed}.csv")
                _, digest = frozen_model(root, p, seed, method, False)
                validate_rows(d, samples, refs, p, method)
                if not ((d.seed == seed) & (d.checkpoint_sha256 == digest) & (d.online_s > 0)).all():
                    raise ValueError("unexpected checkpoint, seed or online timing")
                nets.append(d)
                if method in LOSSES:
                    meta = json.loads((root/method/f"p{p}"/f"meta_train_lshape_seed{seed}.json").read_text())
                    trained.append(dict(method=method, p=p, seed=seed, train_s=meta["elapsed_sec"],
                                        peak_rss_mib=meta["peak_rss_mb"], host=meta["hardware"]["host"],
                                        levels_at_cap=sum(meta["epochs_used"][k] == v for k,v in meta["epochs_per_level"].items()),
                                        sha256=digest))
        for i, sample in enumerate(paired):
            d = read(root/"alg1"/f"lshape_p{p}_i{i}.csv")
            if len(d) != 2*len(LEVELS):
                raise ValueError("incomplete paired Algorithm 1 file")
            for method in LOSSES:
                g = d[d.method == method].sort_values("N")
                validate_rows(g, [sample], refs, p, method)
                if not ((g["T"] == 5) & (g.h_min_floor == 1e-7) & (g.time_s > 0) & (g.nit <= 1000)
                        & g.stop.notna()).all() or not g.time_s.is_monotonic_increasing:
                    raise ValueError("invalid per-instance policy, stopping or cumulative timing")
            algorithms.append(d)
    return pd.concat(nets, ignore_index=True), pd.concat(algorithms, ignore_index=True), pd.DataFrame(trained)


def energy_tables(d):
    group = d.groupby(["p", "N", "method"])
    energy = group.agg(n=("err","size"), err=("err","median"), ieff=("ieff","median"),
                       eta=("eta","median"), below_resolution=("below_resolution","sum")).reset_index()
    for suffix, q in (("p25", 0.25), ("p75", 0.75)):
        energy[f"err_{suffix}"] = group.err.quantile(q).to_numpy()
    uniform = energy[energy.method == "uniform"][["p","N","err"]].rename(columns={"err":"uniform_err"})
    energy = energy.merge(uniform, on=["p","N"])
    energy["improvement"] = energy.uniform_err/energy.err
    rates, accuracy = [], []
    for (p, method), g in energy.groupby(["p","method"]):
        curve = g.set_index("N").sort_index()
        e = curve.err
        rates.append(dict(p=p, method=method, order_32_64=np.log2(e[32]/e[64]),
                          order_4_64=np.log2(e[4]/e[64])/4,
                          order_fit_16_64=-np.polyfit(np.log([16,32,64]), np.log(e.loc[[16,32,64]]), 1)[0]))
        if method in LOSSES:
            target = curve.uniform_err.loc[64]
            level, bound = needed_level(e.index.to_numpy(), e.to_numpy(), target)
            crossing = e[e <= target]
            measured = crossing.index[0] if not crossing.empty else np.nan
            dofs = d[d.p == p].groupby("N").free_dofs.first()
            accuracy.append(dict(p=p, method=method, target=target, N_uniform=64, N_interpolated=level,
                                 bound=bound, nominal_dof_ratio=((64+p)/(level+p))**2, N_measured=measured,
                                 uniform_free_dofs=dofs[64], adapted_free_dofs=dofs.get(measured),
                                 measured_free_dof_ratio=dofs[64]/dofs.get(measured, np.nan)))
    return energy, pd.DataFrame(rates), pd.DataFrame(accuracy)


def precision_table(d, alg):
    both = pd.concat([d, alg.assign(method="alg1-"+alg.method)], ignore_index=True)
    both["ritz_discrepancy"] = (2*both.ritz - both.strain + 2*both.load).abs()/both.F_ref
    both["galerkin_defect"] = (both.strain-both.load).abs()/both.F_ref
    delta = 1e-11
    both["err_lo"] = np.sqrt(((both.gap-delta)/(1-delta)).clip(lower=0))
    both["err_hi"] = np.sqrt(((both.gap+delta)/(1+delta)).clip(lower=0))
    result = both.groupby(["p","N","method"]).agg(
        n=("err","size"), min_gap=("gap","min"), below_resolution=("below_resolution","sum"),
        ritz_discrepancy_max=("ritz_discrepancy","max"), galerkin_defect_max=("galerkin_defect","max"),
        err_lo=("err_lo","median"), err_hi=("err_hi","median")).reset_index()
    baseline = result[result.method == "uniform"][["p","N","err_lo","err_hi"]].rename(
        columns={"err_lo":"uniform_lo", "err_hi":"uniform_hi"})
    result = result.merge(baseline, on=["p","N"])
    result["improvement_lo"] = result.uniform_lo/result.err_hi
    result["improvement_hi"] = result.uniform_hi/result.err_lo
    result.loc[result.method.str.startswith("alg1"), ["improvement_lo","improvement_hi"]] = np.nan
    return result


def paired_tables(d, alg, training):
    paired = alg[["sigma1","sigma2"]].drop_duplicates()
    net = d.merge(paired, on=["sigma1","sigma2"], validate="many_to_one")
    comparison, cost = [], []
    for (p, n), a in alg.groupby(["p","N"]):
        subset = net[(net.p == p) & (net.N == n)]
        row = dict(p=p, N=n, uniform=subset.loc[subset.method == "uniform", "err"].median())
        for method in LOSSES:
            g = a[a.method == method]
            nn = subset[subset.method == method]
            t = training[(training.method == method) & (training.p == p)]
            online, optimizer = nn.online_s.median(), g.time_s.median()
            row.update({f"alg1_{method}": g.err.median(), f"net_{method}": nn.err.median(),
                        f"net_over_alg1_{method}": nn.err.median()/g.err.median(),
                        f"active_{method}": int((g.n_active > 0).sum()),
                        f"iteration_caps_{method}": int((g.stop == "maxiter").sum()),
                        f"alg1_below_resolution_{method}": int(g.below_resolution.sum())})
            cost.append(dict(p=p, N=n, method=method, n_parameters=len(g), n_network_rows=len(nn),
                             train_s=t.train_s.median(), peak_rss_mib=t.peak_rss_mib.median(),
                             peak_rss_max_mib=t.peak_rss_mib.max(), online_s=online, alg1_s=optimizer,
                             break_even=t.train_s.median()/(optimizer-online) if optimizer > online else np.nan))
        comparison.append(row)
    return pd.DataFrame(comparison), pd.DataFrame(cost)


def save(root, path, table):
    target = root/path
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)


def summarize(root):
    d, alg, training = observations(root)
    energy, rates, accuracy = energy_tables(d)
    comparison, cost = paired_tables(d, alg, training)
    mesh = pd.concat([d, alg.assign(method="alg1-"+alg.method)]).groupby(["p","N","method"]).agg(
        gamma_median=("gamma_loc","median"), gamma_max=("gamma_loc","max"), aspect_max=("aspect","max"),
        cond_median=("cond","median"), cond_max=("cond","max"), jacobi_median=("cond_jacobi","median"),
        jacobi_max=("cond_jacobi","max"), h_min=("h_min","min")).reset_index()
    mesh["within_block_bound"], mesh["two_block_bound"] = block_limits(mesh.N)
    stops = alg.groupby(["p","N","method","stop"]).size().reset_index(name="count")
    for name, table in (("energy_effectivity",energy), ("orders",rates), ("fixed_accuracy",accuracy),
                        ("alg1_comparison",comparison), ("cost",cost), ("mesh_conditioning",mesh),
                        ("training",training), ("alg1_stops",stops), ("energy_precision",precision_table(d, alg))):
        save(root, f"tables/{name}.csv", table)
    band = band_rows(d[d.method != "ritz"].assign(err_E=lambda x:x.err, method=lambda x:x.method.replace({"eta":"positional"})))
    for p in (2,3):
        save(root, f"figures/2D_P4/05_convergence_energy/P4E_p{p}.csv", band[band.p == p])
        folder = root/"eta"/f"p{p}"
        epochs = json.loads((folder/"meta_train_lshape_seed0.json").read_text())["epochs_used"]
        history = pd.read_csv(folder/f"history/history_p{p}_seed0.csv", comment="#")
        if len(history) != sum(epochs.values()):
            raise ValueError("history length and per-level epoch counts differ")
        plot = history.iloc[epochs["4"]:epochs["4"]+epochs["8"]].copy()
        plot["epoch"] = np.arange(1, len(plot)+1)
        save(root, f"figures/00_losses/loss_trainval_lshape_p{p}_N8.csv",
             plot.rename(columns={"train_loss":"train", "val_eta2":"val"})[["epoch","train","val"]])
    print(f"{len(d)} frozen/uniform observations, {len(alg)} Algorithm 1 observations, {len(training)} training runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO/"data_results_v3/lshape")
    summarize(parser.parse_args().root)
