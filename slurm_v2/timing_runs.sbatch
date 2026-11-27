#!/bin/bash
#SBATCH --job-name=nniga_timing
#SBATCH --array=0-47
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/timing_%A_%a.out
#SBATCH --error=logs/timing_%A_%a.err
#
# Revision v2, item 3.4: one training run per experiment, degree and seed with
# the as-run protocol of PROVENANCE.md, to record wall time, hardware and peak
# memory (common/resources.py) on the same nodes as the paper runs. The paper
# runs recorded time only for Exps. 1-4 and neither hardware nor memory; Exp. 5
# recorded nothing. Evaluation is skipped where the driver allows it.
#   task % 10:  0-1 singular  2-3 helmholtz  4-5 arctan  6-7 lshape  8-9 advdiff   (p = 2, 3)
#   seed = task / 10  (array 0-39: four seeds; array 0-9: seed 0 only)
#   40-47: Exp. 6 (adr), k = task - 40, p = 2 + k % 2, seed = k / 2
# Output: data_results_v2/timing/<exp>/p<p>/ (meta / metadata JSON)
#
# Submit from the repository root:  mkdir -p logs && sbatch slurm_v2/timing_runs.sbatch
# Local smoke of one task:          TASK=8 SMOKE=1 PYTHON=python bash slurm_v2/timing_runs.sbatch

set -eo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"
mkdir -p logs
CONDA_ENV="${CONDA_ENV:-p1d}"
if [ -d "$HOME/.conda/envs/${CONDA_ENV}/bin" ]; then
    export PATH="$HOME/.conda/envs/${CONDA_ENV}/bin:$PATH"
fi
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

ID="${TASK:-${SLURM_ARRAY_TASK_ID:-0}}"
if [ "${ID}" -ge 40 ]; then
    K=$(( ID - 40 )); EXP=adr; P=$(( 2 + K % 2 )); SEED=$(( K / 2 ))
else
    TASK=$(( ID % 10 )); SEED=$(( ID / 10 ))
    EXPS=(singular singular helmholtz helmholtz arctan arctan lshape lshape advdiff advdiff)
    EXP="${EXPS[$TASK]}"
    P=$(( 2 + TASK % 2 ))
fi
OUT="${OUT_ROOT:-data_results_v2}/timing/${EXP}/p${P}"
SMOKE_FLAG=""; LV2D="4 8 16 32"
if [ "${SMOKE:-0}" = "1" ]; then SMOKE_FLAG="--smoke"; LV2D="4 8"; fi

echo "=== timing task ${ID}: ${EXP} p=${P} seed=${SEED} $(date -Iseconds) host=$(hostname) ==="
case "${EXP}" in
  singular)  "${PYTHON}" -u 1D/scripts/train_singular.py --p "${P}" --seed "${SEED}" --protocol val \
                 --eval-levels --output-dir "${OUT}" ${SMOKE_FLAG} ;;
  adr)       "${PYTHON}" -u 1D/scripts/train_adr.py --p "${P}" --seed "${SEED}" --output-dir "${OUT}" ${SMOKE_FLAG} ;;
  helmholtz) "${PYTHON}" -u 1D/scripts/train_helmholtz.py --p "${P}" --seed "${SEED}" --protocol val \
                 --output-dir "${OUT}" ${SMOKE_FLAG} ;;
  *)         "${PYTHON}" -u "2D/scripts/train_${EXP}.py" --p "${P}" --seed "${SEED}" --protocol val \
                 --N-levels ${LV2D} --lr-init 1e-3 --lr-end 1e-4 --output-dir "${OUT}" ${SMOKE_FLAG} ;;
esac
echo "=== timing task ${ID} done $(date -Iseconds) ==="
