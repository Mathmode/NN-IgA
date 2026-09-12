# Parametric r-adaptive isogeometric analysis

This research code uses a positional density network to predict B-spline knot
distributions. The physical state is computed by a Galerkin solve; the network
is trained through the discrete solver using a residual-estimator objective.
The implementation uses JAX and enables double precision in `common/_precision.py`.

The five implemented problems are singular power solutions and transmission
Helmholtz in one dimension, and arctangent layers, an immersed L-shape, and
advection–diffusion in two dimensions. The experiment drivers support degrees
2 and 3. Numerical kernels, quadrature choices, boundary conditions, normalization,
and reference configurations are retained from the research source.

## Installation and quick example

Use a Python 3.12 environment and run from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/quick_singular.py
```

The example trains for three epochs on four parameter samples at degree 2 and
four elements, then prints the discrete state and residual loss. It is an
execution check, not a reproduction of a complete experiment or an accuracy claim.
No dataset download or GPU is required. The documented interface is execution
from the source checkout; `1D/src` and `2D/src` are separate packages with the
same import name and must be used in separate processes.

## Layout

- `common/`: B-spline values and derivatives, quadrature, masking, metrics,
  parameter sampling, optimization and differentiated linear solves.
- `1D/`, `2D/`: numerical implementations, configurations, training/evaluation
  drivers and tests. Optional Slurm templates need local scheduler settings.
- `scripts/`: table and figure generation, result consolidation, comparisons and
  reference accuracy diagnostics.
- `data_results/`: selected research outputs, small checkpoints and evaluation
  samples used by the retained post-processing workflows.
- `references/`: L-shape reference cache and its provenance/checksum.
- `TiKz_overleaf/`: figure/table sources and associated numeric resources.

## Experiments and reproducibility

See [EXPERIMENTS.md](EXPERIMENTS.md) for entry points and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for reproduction scope and limitations.
[PROVENANCE.md](PROVENANCE.md) records historical run settings, including
settings that differ from current configuration defaults. Existing CSVs and
checkpoints are inherited research artifacts, not newly reproduced results.

```bash
python 1D/scripts/train_singular.py --protocol val --p 2 --seed 0
python 1D/scripts/eval_singular.py --help
python scripts/make_tables.py --check
python scripts/make_figure_bands.py
```

Full training is substantially more expensive than the quick example. Review
output arguments before running: full drivers default to `data_results/` and
may overwrite existing artifacts. Use a separate working copy for regeneration.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest 1D/tests
python -m pytest 2D/tests
python -m pytest scripts/test_data_guards.py
```

Run dimensional suites separately to avoid the shared `src` name. Tests marked
`slow` are excluded by default; enable them explicitly with `-m slow`.
Optional NGSolve checks require a separate installation and are not necessary
for the included immersed IGA reference cache.

## License

The existing source license and copyright notice are preserved in [LICENSE](LICENSE).
No new release date, identifier, author affiliation, or publication citation is asserted.
