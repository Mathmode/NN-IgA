#!/bin/bash
#SBATCH --job-name=lshape_eval
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lshape_eval_%A_%a.out
#SBATCH --error=logs/lshape_eval_%A_%a.err
#
# Evaluate the `lshape` experiment on the held-out test split, for BOTH degrees
# p in {2,3}; one SLURM array task per seed (0..3). The TRUE error is the DIRECT
# H¹ error against the high-resolution immersed IGA self-reference (degree p=5;
# cancels the C⁰ corner cut), passed via --iga-ref-cache. Reads the checkpoint
# written by submit_train_lshape.sbatch and writes the per-seed summary CSV.
#
# SUBMIT FROM THE 2D TREE (after the matching train job finished):
#     cd 2D && sbatch scripts/submit_eval_lshape.sbatch
#
# Env overrides:
#     P_DEGREES="2 3"  degrees to evaluate            (default: both)
#     CONDA_ENV=p1d    conda env with JAX/NGSolve     (default: p1d)
#     CORR=0           --corrector-max-iter (default 0 = corrector-free)
#     FORCE=0          set FORCE=1 to re-evaluate seeds whose summary already exists
#     IGA_REF=<path>   immersed IGA self-reference cache
#                      (default: references/reference_lshape.pkl)
#
# Reads  <repo>/data_results/lshape/p<degree>/checkpoints/seed<seed>/checkpoint_final.npz
# Writes <repo>/data_results/lshape/p<degree>/eval/

set -eo pipefail

EXP_DIR="${SLURM_SUBMIT_DIR:-$PWD}"        # the 2D tree (submit from here)
REPO_ROOT="$(cd "${EXP_DIR}/.." && pwd)"   # repository root (holds references/)
cd "${EXP_DIR}"
mkdir -p logs

CONDA_ENV="${CONDA_ENV:-p1d}"
if [ -d "$HOME/.conda/envs/${CONDA_ENV}/bin" ]; then
    export PATH="$HOME/.conda/envs/${CONDA_ENV}/bin:$PATH"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export JAX_ENABLE_X64=true
export JAX_PLATFORM_NAME=cpu
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SEED="${SLURM_ARRAY_TASK_ID:-0}"
CORR="${CORR:-0}"
FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"

# Immersed IGA self-reference (degree-independent solution truth; one cache serves
# both p=2 and p=3). A large (~57 MB) .gitignored artifact — rsync it into place or
# rebuild it with scripts/generate_reference_lshape.py (see references/README.md).
IGA_REF="${IGA_REF:-${REPO_ROOT}/references/reference_lshape.pkl}"
if [ ! -f "${IGA_REF}" ]; then
    echo "ERROR: IGA self-reference cache not found: ${IGA_REF}"
    echo "       Rebuild it (4-task array of 100 sigmas each + merge):"
    echo "         for t in 0 1 2 3; do python scripts/generate_reference_lshape.py \\"
    echo "             --N 128 --p 3 --grading 1.05 --task-id \$t --n-tasks 4 \\"
    echo "             --output \$(dirname ${IGA_REF})/parts/part_\$t.pkl; done"
    echo "       then merge the parts into ${IGA_REF} (see references/README.md)."
    exit 1
fi

for P_FE in ${P_DEGREES:-2 3}; do
    DATA_DIR="${REPO_ROOT}/data_results/lshape/p${P_FE}"
    CKPT="${DATA_DIR}/checkpoints/seed${SEED}/checkpoint_final.npz"
    OUTPUT_DIR="${DATA_DIR}/eval"
    echo "=== lshape eval seed=${SEED} p=${P_FE} corr=${CORR} (DIRECT H1 via IGA self-ref) at $(date -Iseconds) ==="
    echo "    checkpoint=${CKPT}"
    echo "    iga-ref-cache=${IGA_REF}  ->  ${OUTPUT_DIR}"
    [ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT} (run submit_train_lshape.sbatch first)"; exit 1; }
    python -u scripts/eval_lshape.py \
        --protocol val \
        --seed "${SEED}" \
        --p "${P_FE}" \
        --N-levels 10 20 30 40 \
        --corrector-max-iter "${CORR}" \
        --checkpoint "${CKPT}" \
        --iga-ref-cache "${IGA_REF}" \
        --output-dir "${OUTPUT_DIR}" \
        ${FORCE_FLAG}
    echo "=== lshape eval seed=${SEED} p=${P_FE} DONE at $(date -Iseconds) ==="
done
