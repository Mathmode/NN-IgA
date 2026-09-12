# Reproducibility scope

The prepared tree retains the scientific implementation and selected existing
research artifacts. Historical metadata and scripts are evidence of recorded
settings, not proof that every result can be regenerated identically.

## Tables and figures

In a disposable copy, run:

```bash
python scripts/make_tables.py --check
python scripts/make_tables.py
python scripts/make_figure_bands.py
```

Table generation consumes the canonical `eval_val` CSVs for singular/arctan and
`eval` CSVs for Helmholtz/L-shape/advection–diffusion. It preserves the original
aggregation and post-hoc transformations. In particular, the advection–diffusion
effectivity transformation is intended for the historical CSVs: do not assume
newly generated CSVs require the same correction without checking their schema
and estimator definition. Band and regeneration scripts implement different
historical summaries; their statistics must not be silently interchanged.

The retained `regen_*.py` scripts update existing TikZ sources; their paths now
resolve from the script location. `make_repro_table.py` is an additional historical
summary workflow. TeX compilation requires a separate LaTeX/PGFPlots installation
and was not established by Python validation. Original `.tex`, `.csv` and `.dat`
resources are retained; compiled PDFs, logs and editor backups are excluded.

## Artifacts

`data_results/` includes canonical evaluation outputs, histories needed for loss
plots, parameter samples, metadata and checkpoints. The superseded
advection–diffusion campaign and the extra singular theta0 campaign are excluded.
They are not inputs of the canonical table generator. No external download link
is required or invented. `ARTIFACT_SHA256.txt` records included binary/data assets.
Training can regenerate new checkpoints, but full historical reruns were not
performed, and exact equivalence to stored checkpoints is not established.

The current advection–diffusion campaign lacks complete training provenance in
the source. Historical documentation partly refers to superseded metadata. This
is an unresolved limitation for exact retraining. Defaults and recorded historical
learning rates can also differ. Numerical configuration is not changed to hide
these differences.

## Reference

See `references/README.md`. The cache is retained as a justified large artifact.
Rebuilding it or executing `scripts/reference_accuracy_check.py` is computationally
expensive and is not a routine smoke check. The latter is a preserved diagnostic;
its historical accuracy statements have not been re-certified in this preparation.

## Preparation checks

On Python 3.12.7 / macOS ARM64 CPU with the declared core versions, the reduced
three-epoch example produced finite training loss and matched the source state
and residual exactly. A one-parameter Helmholtz checkpoint evaluation completed.
The five stored tables matched independent recomputation with zero cell
discrepancies; table, band and retained regeneration scripts executed successfully.
This validates post-processing of the included artifacts, not full retraining.

Both dimensional default test suites were executed. A macOS RSS-unit conversion
and an empty optional NGSolve import guard were corrected and their affected
tests rerun successfully. Historical checkpoint-dependent tests and the actual
NGSolve solve remain skipped when their prerequisites are absent. The numerical
PDE kernels, configuration values and test memory threshold were not changed.
