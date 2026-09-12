"""Re-evaluate a saved ``lshape`` checkpoint on the test set.

Writes the per-seed ``summary_p3_seed{seed}.csv``. Pass --iga-ref-cache
(references/reference_lshape.pkl) for the direct sigma-weighted H1 error used
in the paper; --reference-cache / --ngsolve-cache are legacy diagnostics whose
columns stay blank/NaN when absent. Use --protocol val for the paper's held-out
test subset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TWO_D_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _TWO_D_ROOT.parent
for p in (str(_PROJECT_ROOT), str(_TWO_D_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import hashlib

import numpy as np

from src.config import LSHAPE, SPLIT_SEED, TRAIN_VAL
from src.parametric.continuation import load_checkpoint
from src.parametric.evaluation_v2 import evaluate_lshape
from common.parameter_sampling import build_lshape_grid
from common.csv_utils import summary_is_complete


def _test_set_hash(test_sigmas: np.ndarray) -> str:
    """Stable hash of the test tuples (order-independent) so two eval runs can be
    asserted to share the SAME held-out test set."""
    rows = sorted(tuple(round(float(v), 12) for v in row) for row in np.asarray(test_sigmas))
    return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--N-levels", type=int, nargs="+", default=None)
    ap.add_argument("--n-test-sigmas", type=int, default=None)
    ap.add_argument("--corrector-max-iter", type=int, default=30)
    ap.add_argument("--protocol", choices=["fixed", "val"], default="fixed",
                    help="'val' (paper): the fixed 70/15/15 held-out test set, identical "
                         "across seeds. 'fixed' (default): legacy per-seed 70/30 split.")
    ap.add_argument("--reference-cache", type=str, default=None,
                    help="Ritz r-adapt cache .npz (legacy diagnostic).")
    ap.add_argument("--ngsolve-cache", type=Path, default=None,
                    help="NGSolve H1 reference cache .pkl (legacy; prefer --iga-ref-cache).")
    ap.add_argument("--iga-ref-cache", type=Path, default=None,
                    help="Degree-5 immersed IGA self-reference .pkl (the paper metric): "
                         "the reported h1_rel_error_iga_ref is the direct sigma-weighted "
                         "relative H1-seminorm error against it.")
    ap.add_argument("--force", action="store_true",
                    help="Re-evaluate even if a complete summary CSV for this "
                         "seed already exists (default: skip complete seeds).")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_lshape] seed={args.seed} smoke={args.smoke} -> {out_dir}")

    # 'val': fixed SPLIT_SEED 70/15/15 shared across seeds; else per-seed 70/30.
    val_protocol = (args.protocol == "val")
    grid = build_lshape_grid(
        n_sigma1=int(LSHAPE["n_sigma1"]),
        n_sigma2=int(LSHAPE["n_sigma2"]),
        exp_min=float(LSHAPE["sigma_exp_min"]),
        exp_max=float(LSHAPE["sigma_exp_max"]),
        train_frac=(float(TRAIN_VAL["train_frac"]) if val_protocol else float(LSHAPE["train_frac"])),
        seed=(int(SPLIT_SEED) if val_protocol else int(args.seed)),
        val_frac=(float(TRAIN_VAL["val_frac"]) if val_protocol else None),
    )
    test_sigmas = grid["test"]
    print(f"  protocol={args.protocol}  test_set_hash={_test_set_hash(test_sigmas)}  "
          f"(val mode -> hash is seed-independent)")
    if args.smoke:
        n_test = int(args.n_test_sigmas or 4)
        # val mode subsamples with SPLIT_SEED so the smoke set is seed-independent.
        rng = np.random.default_rng(int(SPLIT_SEED) if val_protocol else args.seed)
        test_sigmas = test_sigmas[rng.choice(test_sigmas.shape[0], n_test, replace=False)]
    elif args.n_test_sigmas is not None:
        test_sigmas = test_sigmas[: int(args.n_test_sigmas)]

    print(f"  test sigmas: {test_sigmas.shape}")

    levels = tuple(args.N_levels) if args.N_levels else (
        (10,) if args.smoke else tuple(LSHAPE["levels"])
    )
    print(f"  levels: {levels}")

    summary_path = out_dir / f"summary_p3_seed{int(args.seed)}.csv"
    if not args.force and summary_is_complete(summary_path, levels):
        print(f"[skip] {summary_path} already complete for N={list(levels)}; "
              f"pass --force to re-evaluate")
        return 0

    # Build Ritz j_ref lookup dictionary keyed by rounded (sigma1, sigma2).
    j_ref_lookup = {}
    if args.smoke and not args.reference_cache:
        # Build a mini Ritz reference on the fly (uniform N=32, p=2). Pure
        # diagnostic — the PRIMARY metric is h1_rel_error_ngsolve, which
        # the smoke pipeline leaves blank unless --ngsolve-cache is given.
        print("  building mini Ritz reference (uniform N=32, p=2) on the fly...")
        from src.nonparametric.lshape.reference import compute_ritz_reference
        for s in test_sigmas:
            entry = compute_ritz_reference(
                float(s[0]), float(s[1]),
                n_elem=32, p=2, use_radapt=False,
            )
            j_ref_lookup[(round(float(s[0]), 12), round(float(s[1]), 12))] = entry.j_ref
    elif args.reference_cache:
        cache = np.load(args.reference_cache)
        st = cache["sigma_tuples"]
        jr = cache["j_ref"]
        for (s1, s2), j in zip(st, jr):
            j_ref_lookup[(round(float(s1), 12), round(float(s2), 12))] = float(j)
    else:
        print("  no Ritz reference cache; ritz_rel_error column will be NaN.")

    # Load NGSolve high-resolution H¹ cache (PRIMARY error metric).
    ngsolve_cache = None
    if args.ngsolve_cache is not None:
        if not args.ngsolve_cache.exists():
            raise SystemExit(f"NGSolve cache not found: {args.ngsolve_cache}")
        from src.nonparametric.lshape.ngsolve_reference import (
            load_p3_reference_cache,
        )
        ngsolve_cache = load_p3_reference_cache(args.ngsolve_cache)
        print(f"  loaded NGSolve cache: {len(ngsolve_cache)} entries")
        # Coverage assertion (val protocol): every test tuple MUST have a
        # reference entry, else h1_rel_error_ngsolve silently falls back to a
        # flat/NaN value (the exact bug seen in the new run: ~98/120). Fail loud
        # so an incomplete cache is caught before producing misleading errors.
        if val_protocol:
            missing = []
            for s in test_sigmas:
                key = (round(float(s[0]), 12), round(float(s[1]), 12))
                if key in ngsolve_cache:
                    continue
                if any(abs(k[0] - float(s[0])) < 1e-9 and abs(k[1] - float(s[1])) < 1e-9
                       for k in ngsolve_cache):
                    continue
                missing.append(key)
            covered = len(test_sigmas) - len(missing)
            print(f"  NGSolve coverage: {covered}/{len(test_sigmas)} test tuples")
            if missing:
                raise SystemExit(
                    f"NGSolve cache is INCOMPLETE for the shared test set: "
                    f"{len(missing)}/{len(test_sigmas)} tuples missing (e.g. {missing[:3]}). "
                    f"Rebuild the cache over the FULL grid with "
                    f"`generate_reference_lshape_conforming.py --include-train` so it covers every test tuple."
                )
    else:
        print("  no NGSolve cache; h1_rel_error_ngsolve column will be blank/NaN.")

    # IGA self-reference cache: entries carry u_coeffs + knots, enabling the
    # direct sigma-weighted H1 metric.
    iga_ref_cache = None
    if args.iga_ref_cache is not None:
        if not args.iga_ref_cache.exists():
            raise SystemExit(f"IGA self-reference cache not found: {args.iga_ref_cache}")
        import pickle
        with open(args.iga_ref_cache, "rb") as f:
            iga_ref_cache = pickle.load(f)
        print(f"  loaded IGA self-ref cache: {len(iga_ref_cache)} entries")
        # Entries must carry the solution field, else the metric silently falls
        # back to energy subtraction.
        _e0 = next(iter(iga_ref_cache.values()))
        _is_iga = isinstance(_e0, dict) and "u_coeffs" in _e0 and "knots_x" in _e0
        _is_amr = isinstance(_e0, dict) and "grad_uxr" in _e0 and "grad_uyr" in _e0
        if not (_is_iga or _is_amr):
            raise SystemExit(
                "Reference cache entries lack BOTH u_coeffs/knots (IGA) AND grad_uxr/grad_uyr "
                "(NGSolve-AMR) -> would fall back to the flat energy-subtraction metric. Use a "
                "self-reference cache built with solution fields (IGA) or precomputed gradients (AMR)."
            )
        print(f"  reference schema: {'IGA (u_coeffs)' if _is_iga else 'NGSolve-AMR (grad_uxr)'}")
        # Coverage assertion (val protocol): every test tuple must be present.
        if val_protocol:
            missing = []
            for s in test_sigmas:
                key = (round(float(s[0]), 12), round(float(s[1]), 12))
                if key in iga_ref_cache:
                    continue
                if any(abs(k[0] - float(s[0])) < 1e-9 and abs(k[1] - float(s[1])) < 1e-9
                       for k in iga_ref_cache):
                    continue
                missing.append(key)
            covered = len(test_sigmas) - len(missing)
            print(f"  IGA self-ref coverage: {covered}/{len(test_sigmas)} test tuples (DIRECT metric)")
            if missing:
                raise SystemExit(
                    f"IGA self-ref cache is INCOMPLETE for the shared test set: "
                    f"{len(missing)}/{len(test_sigmas)} tuples missing (e.g. {missing[:3]}). "
                    f"Rebuild the full-grid IGA self-reference so it covers every test tuple."
                )
    else:
        print("  no IGA self-ref cache; reported H1 error falls back to the (flat) "
              "energy-subtraction NGSolve metric -- NOT recommended.")

    params = load_checkpoint(Path(args.checkpoint))

    out_csv = evaluate_lshape(
        params, test_sigmas, j_ref_lookup,
        p=int(args.p),
        levels=levels,
        seed=int(args.seed),
        output_dir=out_dir,
        q_K=int(args.p) + 1,
        q_F=int(LSHAPE["quad_forcing"]),
        corrector_max_iter=int(args.corrector_max_iter),
        checkpoint_path=str(args.checkpoint),
        ngsolve_cache=ngsolve_cache,
        iga_ref_cache=iga_ref_cache,
    )
    print(f"=== eval_lshape DONE -> {out_csv} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
