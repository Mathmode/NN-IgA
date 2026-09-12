# Post-processing and diagnostics

- `make_tables.py`: historical table aggregation; `--check` checks stored cells.
- `make_figure_bands.py`: bands using the table data pipeline.
- `consolidate_experiment.py`: builds the labelled `Data/` result tree.
- `compare_nonparametric_1d.py`, `compare_nonparametric_2d.py`: per-instance versus
  network mesh studies, requiring retained checkpoints.
- `reference_accuracy_check.py`: expensive L-shape reference self-convergence diagnostic.
- `_data_guards.py`, `test_data_guards.py`: refusal of explicitly superseded data.

Per-problem training/evaluation drivers are in `1D/scripts` and `2D/scripts`.
Use `--help` on command-line drivers. Refer to `REPRODUCIBILITY.md` at the root
before recomputing historical tables or figures.
