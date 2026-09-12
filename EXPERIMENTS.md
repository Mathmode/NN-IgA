# Experiment entry points

Run all commands from the repository root. The configuration files
`1D/src/config.py` and `2D/src/config.py` define defaults. The historical artifacts
may use overrides: consult `PROVENANCE.md` and the per-run metadata before
attempting an exact historical rerun. In particular, the documented historical
2D learning rate differs from the current default.

| Problem | Training | Evaluation |
|---|---|---|
| Singular power, 1D | `1D/scripts/train_singular.py` | `1D/scripts/eval_singular.py` |
| Transmission Helmholtz, 1D | `1D/scripts/train_helmholtz.py` | `1D/scripts/eval_helmholtz.py` |
| Arctangent layer, 2D | `2D/scripts/train_arctan.py` | `2D/scripts/eval_arctan.py` |
| Immersed L-shape, 2D | `2D/scripts/train_lshape.py` | `2D/scripts/eval_lshape.py` |
| Advection–diffusion, 2D | `2D/scripts/train_advdiff.py` | `2D/scripts/eval_advdiff.py` |

Each training entry accepts `--protocol val --p 2 --seed 0`; use degree 2 or 3 and
seeds 0 through 3 for the research protocol. Inspect `--help` for checkpoint,
output, mesh-level and optional solver arguments. These are full runs, not smoke
checks. No convergence rates or historical training results were re-established
by repository preparation.

Example of a reduced evaluation using a retained checkpoint:

```bash
python 1D/scripts/eval_helmholtz.py --p 2 --seed 0 \
  --checkpoint data_results/helmholtz/p2/checkpoints/p2_seed0/checkpoint_final.npz \
  --test-cs data_results/helmholtz/p2/test_cs_p2_seed0.npy \
  --N-levels 32 --max-cs 1 --output-dir outputs/helmholtz_smoke
```

For L-shape evaluation, the included cache can be supplied with
`--iga-ref-cache references/reference_lshape.pkl`. The reference is an immersed
IGA comparison solution; no universal error bound is claimed here.

The preserved Slurm scripts are optional templates: activate the desired Python
environment, create the log directory before submission, and supply your own
account/partition to `sbatch`. They retain historical numerical flags and resource
requests. Slurm submission was not tested.

Additional implementations include nonparametric comparisons
(`scripts/compare_nonparametric_1d.py`, `scripts/compare_nonparametric_2d.py`),
Helmholtz extrapolation/denominator studies and the arctan FDM solver. These
variants are not interchangeable with the five historical table pipelines.
