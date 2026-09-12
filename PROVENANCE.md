> Historical source record. Claims below were not independently re-established during repository preparation. Superseded artifacts mentioned below are not included.

# Run provenance (remediation T10)

Per-experiment record of the settings actually used for the shipped results, to
close the provenance gaps the audit flagged (§9): learning-rate schedules recorded
only in `meta_train_*.json`, the arctan solver/`q_F` choice, and the per-experiment
quadrature orders (assembly / estimator / error metric). Values below are read from
`data_results/<slug>/**/meta_train_*.json` (authoritative for what ran) and the
`config.py` files (authoritative for defaults). All runs: CPU, float64,
`protocol="val"`, fixed `SPLIT_SEED=20240601`, four seeds {0,1,2,3}, Adam.

## Experiment ↔ directory ↔ degree

| Exp | Slug | Dir | Table | Eval CSV prefix (legacy, does NOT encode degree) |
|---|---|---|---|---|
| 1 | singular | `data_results/singular/` | Table 1 | `summary_p{2,3}_seed*` |
| 2 | helmholtz | `data_results/helmholtz/` | Table 2 | `summary_p5c_seed*` |
| 3 | arctan | `data_results/arctan/` | Table 3 | `summary_p2_seed*` |
| 4 | lshape | `data_results/lshape/` | Table 4 | `summary_p3_seed*` (even under `p2/`) |
| 5 | advdiff | `data_results/advdiff/` | Table 5 | `summary_p4_seed*` |

`data_results/advdiff_OLD_jun15_mac/` is **superseded** (audit D4); it carries a
`SUPERSEDED.txt` marker and the table generator refuses to read it
(`scripts/_data_guards.py`).

## Learning-rate schedule (as run)

The 2D `TRAIN_VAL` config default is `lr_init=1e-2 → lr_end=1e-4`, but the shipped
2D runs used **`lr_init=1e-3 → lr_end=1e-4`** (recorded in each `meta_train`); the
1D runs used `config.VAL` = `1e-2 → 1e-4`. All schedules are a 2-epoch linear
warm-up (`WARMUP_EPOCHS`) into an exponential decay, with global-norm gradient
clipping at `CLIP_NORM=1.0`, re-initialised per continuation level.

| Exp | lr_init → lr_end | batch | extra |
|---|---|---|---|
| 1 singular | 1e-2 → 1e-4 (`config.VAL`) | 16 | — |
| 2 helmholtz | 1e-2 → 1e-4 (`config.VAL`); warm phase `lr_explore=4e-3` | 16 | 60 warm epochs on top-third contrasts (`c_high_frac=0.34`) at the coarsest level |
| 3 arctan | 1e-3 → 1e-4 (meta) | 16 | — |
| 4 lshape | 1e-3 → 1e-4 (meta) | 16 | — |
| 5 advdiff | 1e-3 → 1e-4 (OLD meta; current run has no `meta_train`, see below) | 16 | — |

Early stopping (all): monitor validation η², 1 % relative-improvement tolerance,
`min_epochs=40`, `patience=40`, `max_epochs=400` cap; best-val weights restored and
used to warm-start the next level.

## Continuation levels, mesh policy, architecture

| Exp | Train levels | Eval levels (zero-shot in **bold**) | T (grading cap) | h_min | network |
|---|---|---|---|---|---|
| 1 singular | 2,4,8,16,32,64 | 2,4,8,16,32,64,**128**,**256** | per-(p,N) schedule† | 1e-7 (N≤32) / 1e-8 (else) p=2; 1e-8 p=3 | 2×32 tanh, input (β, ξ, log(ξ+1/N)) |
| 2 helmholtz | 64,96,128 | **32**,**48**,64,96,128,**256** | 5 | 1e-7 | 2×32 tanh, input (c, ξ, side) |
| 3 arctan | 4,8,16,32 | 4,8,16,32,**64** | 5 | 1e-7 | 2×32 tanh, input (α,s₁,s₂, ξ, axis) |
| 4 lshape | 4,8,16,32 | 4,8,16,32,**64** | 5 | 1e-7 | 2×32 tanh, input (σ₁,σ₂, ξ, axis∈{0,.5,1,1.5}) |
| 5 advdiff | 4,8,16,32 | 4,8,16,32,**64** | 5 | 1e-7 | **3×64** tanh, input (ℓε,b, ξ, axis) |

† Exp 1 `T_SCHEDULE`: p=2 → {2,2,2,2,2,3,5,5} over N=(2,4,8,16,32,64,128,256);
p=3 → {6,6,6,6,6,6,6,7}. Note T=5 (p=2) and T=7 (p=3) occur only at the
extrapolation levels 128/256.

## Quadrature orders (assembly / estimator / error metric)

| Exp | Stiffness (assembly) | Forcing | Estimator η² | Error metric |
|---|---|---|---|---|
| 1 singular | GL `p+1` | analytic (monomial ∫xᵅ) | analytic monomial + cancellation-aware GL fallback (q=2p+2, first-element geometric subdivision n_sub=20) | GL 2p+2 with geometric subdivision of the first (singular) element |
| 2 helmholtz | GL `p+2` | — (source-free) | GL `p+2` | GL 24, composite per region |
| 3 arctan | GL `p+1` | GL 50 | GL 50 | GL 50 |
| 4 lshape | GL `p+1` | GL 4 (exact for f=1) | GL 4 | direct σ-weighted H¹ on an 800²-point masked grid vs the p=5 reference |
| 5 advdiff | GL 4 | GL 50 | GL 50 | GL 50 |

**Arctan solver**: the shipped Table-3 run used the **dense** solver with **`q_F=50`**
(the `config.ARCTAN` defaults); the `--solver fdm --q-forcing 15` variant advertised
in `EXPERIMENTS.md` is a *separate* reference-rate study, **not** the table run.

## Loss denominator (eq. 28)

All five experiments train against the **uniform-mesh estimator** denominator
`η²(θ_unif^{(N)}; ν)` recomputed at each level (`LOSS_NORM="uniform_same_level"`),
with `ε=EPSILON_DENOM=1e-12`. The legacy analytic `‖u*‖²_{H¹}` denominators
(decision 7-H) are **deprecated for training** and kept only as H¹-error *metric*
denominators. (The `training_advdiff.py` module docstring previously mis-stated the
analytic norm as the training denominator; corrected in remediation T10.)

## Reference solution (Exp 4)

`references/reference_lshape.pkl`: immersed **degree-5** IGA self-reference, 128
elements/axis (64/half), **geometrically graded toward the re-entrant corner** with
ratio `1/refine_corner = 1/1.15` (symmetric about 0.5), interface knot 0.5 at
multiplicity 5, `q_K=6`, `q_F=3`, 137²=18769 DOFs/entry, full 20×20 σ grid (400
entries). Reconstructed by `2D/scripts/generate_reference_lshape.py` (remediation
T8); see `scripts/reference_accuracy_check.py` for the self-convergence accuracy
bound.

## Known provenance gaps (still open)

- **Current advdiff run has no `meta_train_*.json`** (only checkpoints/eval/history
  under `data_results/advdiff/`). The settings above for Exp 5 are taken from
  `config.ADVDIFF` + the superseded OLD meta, which share the same structure.
  Re-emitting a `meta_train` on the next advdiff run would close this.
- Estimator weights: Exp 2 uses the coefficient-independent convention (ρ_E=h_E,
  documented in the manuscript); Exp 5 now uses the eq.(12) `h_E²/ε` weight
  (remediation T3).
