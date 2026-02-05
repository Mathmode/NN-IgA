# Differentiable r-Adaptivity with IGA (1D and 2D)

Unified repository with **4 experiments** (2 in **1D** and 2 in **2D**) to study
**differentiable r-adaptivity** in **Isogeometric Analysis (IGA)** with **B-splines**.

We treat the *mesh* (breakpoints / internal knots) as optimizable variables and
minimize an **a posteriori residual estimator**. The full pipeline
(assembly → solve → estimator) is implemented in a differentiable way using
**JAX** (Exp. 1–4).

> This repo **does not generate plots**. All outputs are written as **CSV**
> for external post-processing.

---

## Installation

Requirements:
- Python 3.10+
- Core dependencies:
  - `numpy`
  - `jax`, `jaxlib`, `optax` (Exp. 1–4)

```bash
pip install -r requirements.txt
```

---

## Repository structure

```text
.
├── bspline_basis.py          # shared JAX B-spline kernels (1D/2D)
├── quadrature.py             # Gauss–Legendre rules + monomial integrals (1D/2D)
├── optimization.py           # generic Adam loop + loss utilities (1D/2D)
├── csv_utils.py              # shared CSV helpers (1D/2D)
├── export_utils_1d.py         # shared 1D CSV exporter (Exp. 1–2)
├── results_io.py             # standard results layout (1D/2D)
├── 1D/                       # 1D utilities (IGA, estimators, optimization, CSV export)
├── 2D/                       # 2D utilities (tensor-product IGA, estimators, optimization, CSV export)
├── scripts/
│   ├── experiment_utils.py    # reproducibility helpers (run_metadata.json, list parsing)
│   ├── run_experiment1.py     # Exp. 1 (1D Poisson/power — JAX)
│   ├── run_experiment2.py     # Exp. 2 (1D Helmholtz with interface — JAX, JIT)
│   ├── run_experiment3.py     # Exp. 3 (2D square reaction–diffusion — JAX)
│   └── run_experiment4.py     # Exp. 4 (2D L-shape multipatch Laplace — JAX)
└── requirements.txt
```

---

## Common constraint: minimum element size

All r-adapt experiments enforce a strict minimum element size:

- `h_min` is configurable via `--h-min` in each script.
- Default values are script-specific (Exp. 1–3 use `1e-6`; Exp. 4 uses a larger default for stability at `N>=32`).

via the softmax parametrization:

\[
 h_i = h_{min} + (L - n h_{min})\,\mathrm{softmax}(\theta)_i,\qquad \sum_i h_i = L.
\]

This prevents mesh degeneration during optimization.

---

## Standard output layout (CSV)

All scripts write results in a consistent layout:

```text
<EXP_OUT>/
  ├── run_metadata.json
  ├── <tag>_p{p}.csv
  └── p{p}/N{N}/
      ├── summary.csv
      ├── uniform/
      │   ├── breakpoints.csv
      │   ├── knots.csv
      │   ├── greville.csv
      │   └── solution.csv
      └── r_adapt/
          ├── breakpoints.csv
          ├── knots.csv
          ├── greville.csv
          ├── solution.csv
          └── history.csv
```

Notes:
- `run_metadata.json` stores CLI args, package versions, and (if available) git state for reproducibility.
- `<tag>_p{p}.csv` is the per-degree series CSV (rows: `uniform` and `r_adapt` for each `N`).

---

## Console logs

During training (r-adapt) the scripts print a unified log line:

```text
[N=  64 it  2000] loss=... val=... | H1_rel=... | L2_rel=... | h_min=...
```

---

## Experiment 1 (1D): power manufactured solution (JAX)

```bash
python scripts/run_experiment1.py --out results_1D
```

Common arguments:
- `--p-list "2,3"`
- `--n-list "8,16,32,64"`
- `--iters 40000`
- `--h-min 1e-6`

Default output:
- `results_1D/power_solution/`

---

## Experiment 2 (1D): Helmholtz with an interface (JAX)

```bash
python scripts/run_experiment2.py --out results_1D
```

JIT notes (JAX):
- The interface index `n_left` is kept **fixed** during training to keep shapes static.
- `n_left` is initialized with a heuristic proportional to the integrated phase `|k|/p` and then held fixed.

Common arguments:
- `--p-list "2,3"`
- `--n-list "24,32,46,64"`
- `--iters 10000`
- `--h-min 1e-6`

Default output:
- `results_1D/helmholtz/`

---

## Experiment 3 (2D): reaction–diffusion on the unit square (JAX)

```bash
python scripts/run_experiment3.py --out results_exp3_square
```

Common arguments:
- `--n-list "4,8,16,32,64"`
- `--p 2`
- `--iters 20000`
- `--h-min 1e-6`

Default output:
- `results_exp3_square/`

---

## Experiment 4 (2D): Laplace on an L-shape multipatch domain (JAX)

```bash
python scripts/run_experiment4.py --out results_exp4_lshape
```

Common arguments:
- `--n-list "4,8,16,32,64"`
- `--p 2` (default; any degree supported)
- `--iters 2000` (default; override as needed)
- `--lr1 1e-3 --lr2 3e-4` (stable defaults)
- `--h-min 5e-5` (default; prevents mesh degeneration in CG training)
- `--mesh-reg 1e-5` (smoothness regularization; set `0` to disable)
- `--mesh-reg-hmin 0` (optional extra penalty on very small elements)
- `--grad-clip 1.0` (global grad-norm clip; set `0` to disable)
- `--warmstart uniform` (default) or `--warmstart midpoint` (midpoint refinement of the previous r-adapted mesh when doubling `N`)
- `--solver-train auto` and `--solver-eval auto` (default; policy is by `N`, not by DOFs)
- `--precond auto` (default; uses `block_jacobi` for `N>=32`)

Default solver behaviour (with `p=2`):
- `N=4,8,16`: dense solves during training
- `N>=32`: matrix-free CG during training (with `block_jacobi` when `--precond auto`)

Gradient export (Exp. 4 only):
- Each method folder also writes `gradient.csv` with `(ux,uy)` evaluated on the Greville tensor grid.

Default output:
- `results_exp4_lshape/`
