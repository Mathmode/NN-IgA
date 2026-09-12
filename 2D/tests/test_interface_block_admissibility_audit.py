"""Audit: block/interface admissibility of the L-shape mesh construction.

Invariant under test (both the non-parametric and the parametric path):
  (a) optimized interior knots of the left block lie strictly in (0, 0.5) and of
      the right block strictly in (0.5, 1) -- no knot crosses/reaches the anchor;
  (b) the anchor 0.5 appears with multiplicity EXACTLY p;
  (c) knots are non-decreasing and no element degenerates (parametric path:
      sizes >= h_min; non-parametric path has NO h_min floor by construction, so
      there positivity is asserted and sub-1e-7 sizes are only counted);
  (d) endpoint multiplicities are p+1 at 0 and at 1.

Paths:
  non-parametric  2D/src/nonparametric/r_adapt.py::theta_to_knots_lshape
                  (per-block softmax * 0.5)
  parametric      2D/src/parametric/positional_density_network_2d.py::
                  knots_p3_from_network_axis (per-block policy_step, budget=0.5)

Run standalone for the full audit report:
    python3 2D/tests/test_interface_block_admissibility_audit.py
or via pytest (subset with assertions):
    python -m pytest 2D/tests/test_interface_block_admissibility_audit.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

_TWO_D_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _TWO_D_ROOT.parent
for _p in (str(_PROJECT_ROOT), str(_TWO_D_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import jax
import jax.numpy as jnp

from src.config import T_LSHAPE, H_MIN_LSHAPE, LSHAPE, SPLIT_SEED, TRAIN_VAL
from src.nonparametric.r_adapt import theta_to_knots_lshape
from src.parametric.positional_density_network_2d import (
    PDN2DParams,
    init_params,
    knots_p3_from_network_axis,
    policy_step,
)

T_DEF = float(T_LSHAPE)          # 5.0
HMIN_DEF = float(H_MIN_LSHAPE)   # 1e-7


# --------------------------------------------------------------------------
# Invariant checker
# --------------------------------------------------------------------------
def check_invariants(kn: np.ndarray, p: int, half: int, *, h_min: float | None):
    """Return a dict of violation counts for conditions (a)-(d) on one knot
    vector with layout [0]*(p+1) + interior_l + [0.5]*p + interior_r + [1]*(p+1),
    where each interior block has (half-1) optimized knots."""
    kn = np.asarray(kn, dtype=np.float64)
    v = {"a_left": 0, "a_right": 0, "b_mult": 0, "c_order": 0,
         "c_size": 0, "d_ends": 0}

    n_expected = 2 * (p + 1) + (half - 1) + p + (half - 1)
    assert kn.shape[0] == n_expected, f"unexpected knot count {kn.shape[0]} != {n_expected}"

    il = kn[p + 1: p + 1 + (half - 1)]                       # left optimized knots
    ir = kn[p + 1 + (half - 1) + p: p + 1 + (half - 1) + p + (half - 1)]

    # (a) strict containment in the block
    v["a_left"] += int(np.sum(~((il > 0.0) & (il < 0.5))))
    v["a_right"] += int(np.sum(~((ir > 0.5) & (ir < 1.0))))

    # (b) anchor multiplicity exactly p (exact float compare: 0.5 is exact;
    # near-duplicates within 1e-12 are ALSO counted as violations)
    n_half_exact = int(np.sum(kn == 0.5))
    n_half_near = int(np.sum((np.abs(kn - 0.5) < 1e-12) & (kn != 0.5)))
    v["b_mult"] += int(n_half_exact != p) + n_half_near

    # (c) ordering + element sizes per block (excluding the p-1 zero-length
    # elements at the multiplicity-p anchor, which are structural)
    v["c_order"] += int(np.sum(np.diff(kn) < 0.0))
    sizes_l = np.diff(np.concatenate([[0.0], il, [0.5]]))
    sizes_r = np.diff(np.concatenate([[0.5], ir, [1.0]]))
    sizes = np.concatenate([sizes_l, sizes_r])
    if h_min is not None:
        v["c_size"] += int(np.sum(sizes < h_min * (1.0 - 1e-12)))
    else:
        v["c_size"] += int(np.sum(sizes <= 0.0))

    # (d) endpoint multiplicities
    v["d_ends"] += int(np.sum(kn == 0.0) != p + 1) + int(np.sum(kn == 1.0) != p + 1)
    return v, sizes_l, sizes_r


def _accumulate(total, v):
    for k in v:
        total[k] = total.get(k, 0) + v[k]
    return total


def _linear_params(w_sigma1, w_sigma2, w_xi, w_axis, b):
    """Single-linear-layer PDN2DParams: z = w . (sigma1, sigma2, xi, axis) + b.
    ``forward`` applies tanh to all layers but the last, so one layer = linear."""
    W = jnp.asarray([[w_sigma1], [w_sigma2], [w_xi], [w_axis]], dtype=jnp.float64)
    bb = jnp.asarray([b], dtype=jnp.float64)
    return PDN2DParams(layers=[(W, bb)])


# --------------------------------------------------------------------------
# Test A -- invariant under extreme inputs, both paths
# --------------------------------------------------------------------------
def run_test_A(n_draws: int = 240, verbose: bool = True):
    rng = np.random.default_rng(0)
    report = {}

    # ---- non-parametric path (per input tier) --------------------------------
    tiers = ["N(0,1)", "saturation +/-T", "asymmetric-blocks", "spread-50", "spread-700"]
    per_tier = {t: {"draws": 0, "violations": {}, "min_size": np.inf,
                    "sizes_below_1em7": 0} for t in tiers}
    for k in range(n_draws):
        p = int(rng.choice([2, 3]))
        N = int(rng.choice([8, 40]))
        half = N // 2
        tier = tiers[k % len(tiers)]
        if tier == "N(0,1)":
            tl = rng.standard_normal(half)
            tr = rng.standard_normal(half)
        elif tier == "saturation +/-T":
            tl = T_DEF * rng.choice([-1.0, 1.0], half)
            tr = T_DEF * rng.choice([-1.0, 1.0], half)
        elif tier == "asymmetric-blocks":
            tl = T_DEF + 0.1 * rng.standard_normal(half)
            tr = -T_DEF + 0.1 * rng.standard_normal(half)
        elif tier == "spread-50":
            tl = rng.uniform(-25, 25, half)
            tr = rng.uniform(-25, 25, half)
        else:  # spread-700: float-underflow probe (softmax has no floor here)
            tl = rng.uniform(-350, 350, half)
            tr = rng.uniform(-350, 350, half)
        kn = np.asarray(theta_to_knots_lshape(jnp.asarray(tl), jnp.asarray(tr), p))
        v, sl, sr = check_invariants(kn, p, half, h_min=None)
        rec = per_tier[tier]
        rec["draws"] += 1
        rec["violations"] = _accumulate(rec["violations"], v)
        s = np.concatenate([sl, sr])
        rec["min_size"] = min(rec["min_size"], float(s.min()))
        rec["sizes_below_1em7"] += int(np.sum(s < HMIN_DEF))
    report["non-parametric"] = per_tier

    # ---- parametric path (production function, crafted + scaled networks) ----
    tot_pa = {}
    n_pa = 0
    min_size_pa = np.inf
    min_gap_anchor = np.inf
    for k in range(n_draws):
        p = int(rng.choice([2, 3]))
        N = int(rng.choice([8, 40]))
        half = N // 2
        sigma = jnp.asarray(rng.uniform(0.1, 10.0, size=2))
        mode = k % 4
        if mode == 0:      # random init, production scale
            params = init_params(int(rng.integers(1e6)), sigma_dim=2, hidden_dims=(32, 32))
        elif mode == 1:    # random init, weights x100 (deep saturation)
            params = init_params(int(rng.integers(1e6)), sigma_dim=2, hidden_dims=(32, 32))
            params = PDN2DParams(layers=[(W * 100.0, b) for (W, b) in params.layers])
        elif mode == 2:    # crafted: extreme cross-block asymmetry via axis_id
            params = _linear_params(0.0, 0.0, 0.0, rng.choice([-1, 1]) * 1e3, 0.0)
        else:              # crafted: steep within-block gradient + block offset
            params = _linear_params(0.0, 0.0, rng.choice([-1, 1]) * 1e3,
                                    rng.choice([-1, 1]) * 1e3, rng.uniform(-5, 5))
        for axis_id in (0.0, 1.0):
            kn = np.asarray(knots_p3_from_network_axis(params, sigma, N, p, axis_id))
            v, sl, sr = check_invariants(kn, p, half, h_min=HMIN_DEF)
            tot_pa = _accumulate(tot_pa, v)
            n_pa += 1
            s = np.concatenate([sl, sr])
            min_size_pa = min(min_size_pa, float(s.min()))
            others = kn[(kn != 0.5)]
            min_gap_anchor = min(min_gap_anchor, float(np.min(np.abs(others - 0.5))))
    report["parametric"] = dict(draws=n_pa, violations=tot_pa,
                                min_size=min_size_pa, min_gap_to_anchor=min_gap_anchor)

    if verbose:
        print("== Test A: invariant under extreme inputs ==")
        print("  [non-parametric] per input tier:")
        for t, r in report["non-parametric"].items():
            print(f"    {t:18s} {r['draws']:3d} draws: violations {r['violations']} "
                  f"min_size={r['min_size']:.2e} sizes<1e-7: {r['sizes_below_1em7']}")
        r = report["parametric"]
        print(f"  [parametric] {r['draws']} knot vectors: violations {r['violations']}")
        print(f"           min_size={r['min_size']:.3e} "
              f"min_gap_to_anchor={r['min_gap_to_anchor']:.3e}")
    return report


def test_A_nonparametric_and_parametric_invariants():
    rep = run_test_A(n_draws=240, verbose=False)
    # Parametric: every condition, including the h_min floor, must hold.
    assert all(v == 0 for v in rep["parametric"]["violations"].values()), rep["parametric"]
    # Non-parametric: the invariant must hold on BOUNDED logits (the realizable
    # set after a +-T clamp, i.e. what the corrector/optimizer uses in practice).
    # The unbounded spread tiers document audit finding R3 (no h_min floor:
    # softmax underflow lets knots collide with the anchor in float64) and are
    # reported, not asserted.
    for t in ("N(0,1)", "saturation +/-T", "asymmetric-blocks"):
        r = rep["non-parametric"][t]
        assert all(v == 0 for v in r["violations"].values()), (t, r)


# --------------------------------------------------------------------------
# Test B -- trained checkpoints, held-out sigmas, levels incl. zero-shot N=64
# --------------------------------------------------------------------------
def run_test_B(verbose: bool = True):
    from common.parameter_sampling import build_lshape_grid

    grid = build_lshape_grid(
        n_sigma1=int(LSHAPE["n_sigma1"]), n_sigma2=int(LSHAPE["n_sigma2"]),
        exp_min=float(LSHAPE["sigma_exp_min"]), exp_max=float(LSHAPE["sigma_exp_max"]),
        train_frac=float(TRAIN_VAL["train_frac"]), seed=int(SPLIT_SEED),
        val_frac=float(TRAIN_VAL["val_frac"]),
    )
    test_sigmas = grid["test"]                       # 60 held-out pairs
    full_grid = grid["grid"]                         # all 400 pairs (N=64 sweep)

    report = {}
    for p in (2, 3):
        ck = (_PROJECT_ROOT / "data_results" / "lshape" / f"p{p}" / "checkpoints"
              / "seed0" / "checkpoint_final.npz")
        if not ck.exists():
            report[p] = "checkpoint missing -- skipped"
            continue
        d = np.load(ck)
        layers = []
        i = 0
        while f"W{i}" in d.files:
            layers.append((jnp.asarray(d[f"W{i}"]), jnp.asarray(d[f"b{i}"])))
            i += 1
        params = PDN2DParams(layers=layers)

        tot = {}
        n = 0
        min_gap = np.inf
        cases = ([(s, N) for s in test_sigmas for N in (4, 8, 16, 32, 64)]
                 + [(s, 64) for s in full_grid])
        for s, N in cases:
            half = N // 2
            sig = jnp.asarray([float(s[0]), float(s[1])])
            for axis_id in (0.0, 1.0):
                kn = np.asarray(knots_p3_from_network_axis(params, sig, N, p, axis_id))
                v, _, _ = check_invariants(kn, p, half, h_min=HMIN_DEF)
                tot = _accumulate(tot, v)
                n += 1
                others = kn[(kn != 0.5)]
                min_gap = min(min_gap, float(np.min(np.abs(others - 0.5))))
        report[p] = dict(meshes=n, violations=tot, min_gap_to_anchor=min_gap)
        if verbose:
            print(f"  [p={p}] {n} predicted meshes (60 test sigmas x N in "
                  f"{{4,8,16,32,64}} + 400-grid at zero-shot N=64, both axes): "
                  f"violations {tot}; min |knot-0.5| = {min_gap:.3e}")
    return report


def test_B_trained_checkpoints():
    rep = run_test_B(verbose=False)
    for p, r in rep.items():
        if isinstance(r, str):
            continue
        assert all(v == 0 for v in r["violations"].values()), (p, r)


# --------------------------------------------------------------------------
# Test C -- mass-transfer probe (per-block vs hypothetical global softmax)
# --------------------------------------------------------------------------
def run_test_C(verbose: bool = True):
    p, N = 2, 40
    half = N // 2
    sigma = jnp.asarray([1.0, 1.0])
    # Crafted network: huge +/- offset by block (axis_id) + steep xi gradient.
    params = _linear_params(0.0, 0.0, 40.0, -2000.0, 500.0)
    # (x-axis: left block axis_id=0.0 -> z ~ +500; right block axis_id=0.5 -> z ~ -500)

    kn = np.asarray(knots_p3_from_network_axis(params, sigma, N, p, 0.0))
    il = kn[p + 1: p + 1 + (half - 1)]
    ir = kn[p + 1 + (half - 1) + p: p + 1 + (half - 1) + p + (half - 1)]
    sum_l = float(np.sum(np.diff(np.concatenate([[0.0], il, [0.5]]))))
    sum_r = float(np.sum(np.diff(np.concatenate([[0.5], ir, [1.0]]))))
    spill = int(np.sum(il >= 0.5) + np.sum(ir <= 0.5))

    # Counterfactual: what a GLOBAL softmax over both blocks' logits would do.
    from src.parametric.positional_density_network_2d import forward, cell_midpoints
    xi = cell_midpoints(half)
    z_l = forward(params, sigma, xi, axis_id=0.0)
    z_r = forward(params, sigma, xi, axis_id=0.5)
    z_glob = jnp.concatenate([z_l, z_r])
    sizes_glob = np.asarray(policy_step(z_glob, T=T_DEF, h_min=HMIN_DEF, budget=1.0))
    sum_l_glob = float(np.sum(sizes_glob[:half]))
    sum_r_glob = float(np.sum(sizes_glob[half:]))
    knots_glob = np.cumsum(sizes_glob)[:-1]
    spill_glob = int(np.sum(knots_glob[: half - 1] >= 0.5))

    if verbose:
        print("== Test C: mass-transfer probe (left block wants all the mass) ==")
        print(f"  ACTUAL per-block construction: sum(left)={sum_l:.15f}, "
              f"sum(right)={sum_r:.15f}, knots spilled across 0.5: {spill}")
        print(f"  HYPOTHETICAL global softmax:  sum(left)={sum_l_glob:.6f}, "
              f"sum(right)={sum_r_glob:.6f}, left-block knots past 0.5: {spill_glob}")
    return dict(sum_l=sum_l, sum_r=sum_r, spill=spill,
                sum_l_glob=sum_l_glob, sum_r_glob=sum_r_glob, spill_glob=spill_glob)


def test_C_mass_transfer():
    r = run_test_C(verbose=False)
    assert abs(r["sum_l"] - 0.5) < 1e-12 and abs(r["sum_r"] - 0.5) < 1e-12
    assert r["spill"] == 0
    # sanity of the probe itself: the hypothetical global softmax WOULD spill.
    assert r["sum_l_glob"] > 0.9 and r["spill_glob"] > 0


# --------------------------------------------------------------------------
# Test D -- the anchor is outside the optimization
# --------------------------------------------------------------------------
def run_test_D(verbose: bool = True):
    p, N = 3, 16
    half = N // 2
    sigma = jnp.asarray([2.0, 0.5])
    params = init_params(7, sigma_dim=2, hidden_dims=(32, 32))

    from jax.flatten_util import ravel_pytree
    flat, unravel = ravel_pytree(params)

    def knots_of(flat_params):
        return knots_p3_from_network_axis(unravel(flat_params), sigma, N, p, 0.0)

    J = np.asarray(jax.jacobian(knots_of)(flat))          # (n_knots, n_params)
    kn0 = np.asarray(knots_of(flat))
    anchor_rows = np.where(kn0 == 0.5)[0]
    end_rows = np.where((kn0 == 0.0) | (kn0 == 1.0))[0]
    max_grad_anchor = float(np.max(np.abs(J[anchor_rows]))) if anchor_rows.size else np.nan
    max_grad_ends = float(np.max(np.abs(J[end_rows]))) if end_rows.size else np.nan
    max_grad_free = float(np.max(np.abs(J)))

    # Finite-difference confirmation: random large perturbations leave the
    # anchor knots bit-identical.
    rng = np.random.default_rng(1)
    fd_moved = 0
    for _ in range(20):
        delta = jnp.asarray(rng.standard_normal(flat.shape) * rng.choice([1e-3, 1.0, 50.0]))
        kn1 = np.asarray(knots_of(flat + delta))
        fd_moved += int(np.any(kn1[anchor_rows] != 0.5))

    # h_min flooring near the anchor: the closest optimized knot can approach
    # 0.5 by at most h_min (sizes_l[-1] >= h_min); verify at deep saturation.
    params_sat = _linear_params(0.0, 0.0, 1e3, 0.0, 0.0)   # all mass toward xi=1 (left block end)
    kn_sat = np.asarray(knots_p3_from_network_axis(params_sat, sigma, N, p, 0.0))
    others = kn_sat[kn_sat != 0.5]
    gap = float(np.min(np.abs(others - 0.5)))
    mult = int(np.sum(kn_sat == 0.5))

    if verbose:
        print("== Test D: anchor outside the optimization ==")
        print(f"  jacobian rows at anchor knots: max |d(0.5)/d(params)| = {max_grad_anchor:.1e} "
              f"(endpoints: {max_grad_ends:.1e}; free knots: {max_grad_free:.3e})")
        print(f"  FD: anchor knots moved in {fd_moved}/20 large random perturbations")
        print(f"  saturated grading toward the anchor: min |knot-0.5| = {gap:.3e} "
              f"(>= h_min={HMIN_DEF:.0e}), anchor multiplicity = {mult} (expected {p})")
    return dict(max_grad_anchor=max_grad_anchor, fd_moved=fd_moved, gap=gap, mult=mult)


def test_D_anchor_fixed():
    r = run_test_D(verbose=False)
    assert r["max_grad_anchor"] == 0.0
    assert r["fd_moved"] == 0
    assert r["gap"] >= HMIN_DEF * (1 - 1e-9)
    assert r["mult"] == 3


if __name__ == "__main__":
    print("Block/interface admissibility audit -- L-shape (anchor 0.5, mult p)\n")
    run_test_A()
    print("\n== Test B: trained checkpoints (seed 0), held-out sigmas, N incl. 64 ==")
    run_test_B()
    print()
    run_test_C()
    print()
    run_test_D()
    print("\nDone.")
