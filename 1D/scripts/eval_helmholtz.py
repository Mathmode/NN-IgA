"""Evaluation for helmholtz (contrast Helmholtz, global mesh): uniform + positional.

Loads a helmholtz checkpoint and evaluates uniform vs positional (global-mesh) over
--N-levels for resonance-safe test contrasts c, writing one CSV per seed with the
realized per-layer SPLIT (so the learned c->split map is visible). Corrector-free.

Columns: experiment,p,seed,N,c,method,h1_rel_error,eta,eff_index,split_left,
         split_right,split_left_frac,g_min_eff,dof_count,time_s
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; PROJECT = ROOT.parent
for _p in (str(PROJECT), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.csv_utils import summary_is_complete

FIELDS = ["experiment", "p", "seed", "N", "c", "method", "h1_rel_error", "eta", "eff_index",
          "split_left", "split_right", "split_left_frac", "g_min_eff", "dof_count", "time_s"]


def main() -> None:
    ap = argparse.ArgumentParser(description="helmholtz evaluation: uniform + positional (global mesh).")
    ap.add_argument("--p", type=int, required=True, choices=[2, 3])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--N-levels", type=int, nargs="+", default=[32, 48, 64, 96, 128, 256])
    ap.add_argument("--test-cs", type=Path, default=None)
    ap.add_argument("--max-cs", type=int, default=None)
    ap.add_argument("--corrector-max-iter", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="Re-evaluate even if a complete summary CSV for this "
                         "seed already exists (default: skip complete seeds).")
    args = ap.parse_args()

    # Skip seeds whose summary already covers every requested N (unless --force),
    # so a re-run of the eval array does not recompute finished seeds.
    out = Path(args.output_dir).resolve()
    csv_path = out / f"summary_p5c_seed{args.seed}.csv"
    if not args.force and summary_is_complete(csv_path, args.N_levels):
        print(f"[skip] {csv_path} already complete for N={list(args.N_levels)}; "
              f"pass --force to re-evaluate")
        return

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu" if args.device == "cpu" else "gpu,cpu")

    import numpy as np
    import jax.numpy as jnp
    from src.config import HELMHOLTZ
    import src.helmholtz.mesh_helmholtz_global as G
    from src.helmholtz.network_helmholtz import params_from_flat_dict, knots_from_network_global
    from src.helmholtz.mesh_helmholtz import uniform_two_patch_knots

    if int(args.corrector_max_iter) > 0:
        print("[eval helmholtz] note: helmholtz is corrector-free; ignoring --corrector-max-iter.")

    params = params_from_flat_dict(np.load(args.checkpoint))
    if args.test_cs and Path(args.test_cs).exists():
        cs = np.load(args.test_cs)
    else:
        cs = G.sample_contrasts(float(HELMHOLTZ["c_min"]), float(HELMHOLTZ["c_max"]),
                                int(HELMHOLTZ["n_c"]), float(HELMHOLTZ["resonance_tol_c"]))
    if args.max_cs:
        cs = cs[: int(args.max_cs)]

    p = int(args.p); nq = G.quad_order(p); T = float(HELMHOLTZ["T_cap"]); h_min = float(HELMHOLTZ["h_min"])
    out.mkdir(parents=True, exist_ok=True)   # out / csv_path resolved above (skip check)

    rows = []
    for N in (int(x) for x in args.N_levels):
        for c in cs:
            c = float(c); ex = G.ExactContrast(c); rho1 = jnp.asarray(G.rho1_of_c(c))
            for method in ("uniform", "positional"):
                t0 = time.perf_counter()
                if method == "uniform":
                    kn = uniform_two_patch_knots(N, p)             # honest 50:50 baseline
                else:
                    kn = knots_from_network_global(params, jnp.asarray(c), N, p,
                                                   T=T, h_min=h_min, use_hmax=True)
                u = G.solve_u(kn, p, rho1, nq)
                e2 = float(G.eta2(kn, p, u, rho1, nq))
                re = G.h1_rel(kn, p, u, ex)
                eff = G.eff_index(kn, p, u, rho1, ex, nq)
                nl, nr = G.split_counts(kn, p)
                dt = time.perf_counter() - t0
                rows.append(dict(experiment="helmholtz_contrast", p=p, seed=int(args.seed),
                                 N=N, c=round(c, 6), method=method, h1_rel_error=re,
                                 eta=float(np.sqrt(max(e2, 0.0))), eff_index=eff,
                                 split_left=nl, split_right=nr,
                                 split_left_frac=round(nl / (nl + nr), 4) if (nl + nr) else "",
                                 g_min_eff=round(G.effective_g_min(kn, p, c), 3),
                                 dof_count=int(kn.shape[0] - p - 1), time_s=round(dt, 3)))
        print(f"  [eval helmholtz] N={N}: {len(cs)} contrasts x 2 methods", flush=True)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"[eval helmholtz] wrote {len(rows)} rows -> {csv_path}  "
          f"(methods: {sorted({r['method'] for r in rows})})", flush=True)


if __name__ == "__main__":
    main()
