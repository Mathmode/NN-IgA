#!/usr/bin/env python3
"""Phase 0 (physics verification) for the contrast-extrapolation study.

MEASURES (does not assume) two things, on the SAME uniform mesh the evaluation uses
as the Galerkin floor (uniform_two_patch_knots), so the conclusion transfers to the
study's arm C:

  1. Points-per-wavelength (ppw) per medium (left: k1 = c*K2 grows with c; right:
     k2 = K2 fixed) for c in {1.5,3,4.5,6} x N in {64,128,256}. Nyquist is "violated"
     when ppw < G_MIN_TARGET (=2.5). This says whether the high-contrast difficulty is
     wave under-resolution.

  2. eta^2 decomposition: element residual (reaction/wave) vs interface jump vs Neumann
     BC, as fractions of the total, plus the true relative H1 error -- on the uniform
     mesh and on the production network mesh (checkpoint), at several contrasts. This
     says whether the error is dominated by the wave or by the transmission interface.

  3. An operational c_split: the contrast where the uniform-mesh H1 error at N=128 first
     exceeds ~2x its value at c_min (used only if Nyquist is not violated).

This is read-only diagnostics: nothing is trained, nothing is written to production.
Run from the 1D tree:
    cd 1D && python scripts/phase0_extrap_physics.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / "1D")]

import numpy as np
import jax.numpy as jnp

import src.helmholtz.mesh_helmholtz_global as G
from src.helmholtz.mesh_helmholtz import uniform_two_patch_knots
from src.helmholtz.network_helmholtz import params_from_flat_dict, knots_from_network_global
from src.config import HELMHOLTZ

P = 2
NQ = G.quad_order(P)
T = float(HELMHOLTZ["T_cap"]); H_MIN = float(HELMHOLTZ["h_min"])
CKPT = Path(__file__).resolve().parents[2] / "data_results/helmholtz/p2/checkpoints/p2_seed0/checkpoint_final.npz"


def _elements(kn):
    kn = np.asarray(kn); n = G.n_elem_from_knots(kn, P)
    a = kn[P:P + n]; b = kn[P + 1:P + n + 1]
    he = b - a; mid = 0.5 * (a + b)
    keep = he > 1e-13
    return he[keep], mid[keep]


def ppw_per_medium(kn, c):
    """Realized points-per-wavelength in each medium: 2*pi / max_e(k_e * h_e)."""
    he, mid = _elements(kn)
    k1 = G.k1_of_c(c); k2 = G.K2
    out = {}
    for name, m, k in (("left(k1=cK2)", mid < G.X_I, k1), ("right(k2=K2)", mid > G.X_I, k2)):
        if m.any():
            out[name] = float(2.0 * math.pi / np.max(k * he[m]))
        else:
            out[name] = float("nan")
    return out


def eta2_terms(kn, c):
    """(total, frac_elem, frac_jump, frac_bc, h1_rel). Total via G.eta2 (the eval's
    estimator); jump and BC computed directly from uh_deriv, element = total-jump-bc."""
    rho1 = jnp.asarray(G.rho1_of_c(c)); knj = jnp.asarray(kn)
    u = G.solve_u(knj, P, rho1, NQ)
    total = float(G.eta2(knj, P, u, rho1, NQ))
    he, _ = _elements(kn)
    duL = float(G.uh_deriv(knj, P, u, jnp.asarray([G.X_I - 1e-9]))[0])
    duR = float(G.uh_deriv(knj, P, u, jnp.asarray([G.X_I + 1e-9]))[0])
    jump = G.SIGMA2 * duR - G.SIGMA1 * duL
    e_jump = float(min(he[0], he[-1]) * jump * jump)
    du1 = float(G.uh_deriv(knj, P, u, jnp.asarray([1.0 - 1e-9]))[0])
    fe = G.G_N - G.SIGMA2 * du1
    e_bc = float(he[-1] * fe * fe)
    e_elem = max(total - e_jump - e_bc, 0.0)
    h1 = float(G.h1_rel(knj, P, u, G.ExactContrast(c)))
    s = total if total > 0 else 1.0
    return total, e_elem / s, e_jump / s, e_bc / s, h1


def main():
    print("=== Phase 0 -- physics verification (contrast-extrapolation study) ===")
    print(f"    X_I={G.X_I}  SIGMA1/2={G.SIGMA1}/{G.SIGMA2}  K2(right)={G.K2:.4f}  "
          f"k1(left)=c*K2  G_MIN_TARGET={G.G_MIN_TARGET}  (p={P}, uniform_two_patch mesh)")
    print()

    # ---- Part 1: points-per-wavelength per medium -----------------------------
    print("Part 1 -- realized points-per-wavelength (ppw) per medium, uniform mesh")
    print(f"  Nyquist 'violated' when ppw < G_MIN_TARGET = {G.G_MIN_TARGET}")
    print(f"  {'c':>4} {'N':>4} | {'ppw_left':>9} {'ppw_right':>9} | {'g_min_eff':>9} | violated?")
    print("  " + "-" * 58)
    any_violation = False
    for c in (1.5, 3.0, 4.5, 6.0):
        for N in (64, 128, 256):
            kn = uniform_two_patch_knots(N, P)
            pm = ppw_per_medium(kn, c)
            gmin = float(G.effective_g_min(kn, P, c))
            viol = gmin < G.G_MIN_TARGET
            any_violation = any_violation or viol
            print(f"  {c:>4} {N:>4} | {pm['left(k1=cK2)']:>9.2f} {pm['right(k2=K2)']:>9.2f} | "
                  f"{gmin:>9.2f} | {'YES' if viol else 'no'}")
    print(f"  --> Nyquist violated anywhere in this grid? {'YES' if any_violation else 'NO'}")
    print()

    # ---- Part 2: eta^2 decomposition (uniform + network) ----------------------
    print("Part 2 -- eta^2 decomposition (fractions of total) + true H1 rel error")
    have_net = CKPT.exists()
    params = params_from_flat_dict(np.load(CKPT)) if have_net else None
    if not have_net:
        print(f"  (network checkpoint not found at {CKPT}; uniform only)")
    for N in (64, 128):
        print(f"  N={N}:  {'c':>4} {'mesh':>9} | {'elem%':>6} {'jump%':>6} {'bc%':>6} | "
              f"{'eta2_tot':>10} | {'H1_rel':>9}")
        for c in (1.5, 3.0, 4.5, 6.0):
            kn_u = uniform_two_patch_knots(N, P)
            tot, fe_, fj, fb, h1 = eta2_terms(kn_u, c)
            print(f"        {c:>4} {'uniform':>9} | {100*fe_:>5.1f}% {100*fj:>5.1f}% {100*fb:>5.1f}% | "
                  f"{tot:>10.3e} | {h1:>9.3e}")
            if have_net:
                kn_n = knots_from_network_global(params, jnp.asarray(c), N, P, T=T, h_min=H_MIN, use_hmax=True)
                tot2, fe2, fj2, fb2, h12 = eta2_terms(np.asarray(kn_n), c)
                print(f"        {c:>4} {'network':>9} | {100*fe2:>5.1f}% {100*fj2:>5.1f}% {100*fb2:>5.1f}% | "
                      f"{tot2:>10.3e} | {h12:>9.3e}")
        print()

    # ---- Part 3: operational c_split (uniform H1 at N=128 vs 2x its c_min value)
    print("Part 3 -- operational c_split: where uniform H1 (N=128) first exceeds 2x its c_min value")
    cs = np.linspace(float(HELMHOLTZ["c_min"]), float(HELMHOLTZ["c_max"]), 19)
    h1s = []
    for c in cs:
        kn = uniform_two_patch_knots(128, P)
        *_, h1 = eta2_terms(kn, float(c))
        h1s.append(h1)
    h1s = np.array(h1s)
    base = h1s[0]
    cross = next((float(cs[i]) for i in range(len(cs)) if h1s[i] > 2.0 * base), None)
    print(f"  H1(uniform,N=128) at c_min={cs[0]:.2f} is {base:.3e}; 2x = {2*base:.3e}")
    print("  c, H1_rel(uniform,N=128):")
    for c, h in zip(cs, h1s):
        mark = "  <-- > 2x c_min" if h > 2.0 * base else ""
        print(f"    c={c:>4.2f}  H1={h:.3e}{mark}")
    print(f"  --> operational c_split (first crossing) = "
          f"{cross if cross is not None else 'none in range (use default 3.0)'}")
    print()
    print("=== read the tables above; the conclusion (wave vs interface, Nyquist, c_split) ")
    print("    is written into REPORT_extrap_study.md ===")


if __name__ == "__main__":
    main()
