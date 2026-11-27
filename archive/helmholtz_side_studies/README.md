# Helmholtz side studies (archived)

Two exploratory studies on Experiment 2 (1D Helmholtz transmission, contrast
parametrization). They do not feed any table or figure of the paper and are kept
only for provenance of the data they produced.

| Study | Scripts | Data |
|---|---|---|
| Contrast extrapolation (train on low c, evaluate on high c) | `train_helmholtz_extrap.py`, `phase0_extrap_physics.py`, `analyze_extrap_study.py`, `run_extrap_study.sh`, `submit_extrap_{train,eval}.sbatch` | `data_results/helmholtz_extrap/`, `data_results/helmholtz_extrap_evalA/` |
| Loss denominator (`--loss-denom {uniform,exact,robust}` of `train_helmholtz.py`) | `analyze_denom_study.py`, `run_denom_study.sh`, `submit_denom_study.sbatch` | not retained |

The scripts were written to run from the `1D/` tree (`cd 1D && python scripts/...`).
To rerun one, copy it back to `1D/scripts/`.
