#!/bin/bash
#SBATCH --job-name=lshape_ref_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/lshape_ref_v2_%j.out
#SBATCH --error=logs/lshape_ref_v2_%j.err
#
# Revision v2, item 2.1: regenerate the degree-5 L-shape reference cache with the
# corrected Dirichlet mask (20x20 sigma grid, 400 entries, 137^2 = 18769 dofs,
# dense K). Measured on a 6-core desktop: ~17 s and 8.3 GB peak RSS
# (16.4 GB committed) per entry, ~2 h in total. The generator refuses to write
# the cache if any entry has a Dirichlet violation above 1e-14.
#
# Submit from the repository root:
#     mkdir -p logs && sbatch slurm_v2/regen_reference_lshape.sbatch
#
# Env overrides:
#     CONDA_ENV=p1d     conda env with JAX            (default: p1d)
#     PYTHON=python     interpreter                   (default: python)
#     SMOKE=1           3 spot-check entries, no write (local test)
#     OUT=<path>        output cache  (default: references_v2/reference_lshape.pkl)

set -eo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"
mkdir -p logs

CONDA_ENV="${CONDA_ENV:-p1d}"
if [ -d "$HOME/.conda/envs/${CONDA_ENV}/bin" ]; then
    export PATH="$HOME/.conda/envs/${CONDA_ENV}/bin:$PATH"
fi
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

OUT="${OUT:-references_v2/reference_lshape.pkl}"

echo "=== lshape reference v2 start $(date -Iseconds) host=$(hostname) smoke=${SMOKE:-0} ==="
if [ "${SMOKE:-0}" = "1" ]; then
    "${PYTHON}" -u 2D/scripts/generate_reference_lshape.py --spot-check
else
    "${PYTHON}" -u 2D/scripts/generate_reference_lshape.py --out "${OUT}"
    "${PYTHON}" -u -m pytest 2D/tests/test_lshape_conformity.py -q -p no:cacheprovider \
        -k "reference_v2"
    sha256sum "${OUT}" | tee "${OUT}.sha256"
fi
echo "=== lshape reference v2 done $(date -Iseconds) ==="
