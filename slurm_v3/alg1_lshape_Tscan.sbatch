#!/bin/bash
#SBATCH --job-name=nniga_lsh_Tscan
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lsh_Tscan_%A_%a.out
#SBATCH --error=logs/lsh_Tscan_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-p1d_v2}"
export PATH="$HOME/.conda/envs/$CONDA_ENV/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cpu

TASK="${SLURM_ARRAY_TASK_ID:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
[[ "$CHUNK_SIZE" == 4 || "$CHUNK_SIZE" == 1 ]]
CHUNKS=$(( 16 / CHUNK_SIZE ))
[[ "$TASK" =~ ^[0-9]+$ && "$TASK" -lt $(( 8 * CHUNKS )) ]]
P=$(( 2 + TASK / (4 * CHUNKS) ))
TS=(5 8 10 15)
T="${TS[$(( TASK / CHUNKS % 4 ))]}"
CHUNK=$(( TASK % CHUNKS ))
LO=$(( CHUNK_SIZE * CHUNK ))
HI=$(( LO + CHUNK_SIZE ))
SPLIT="${SCAN_SPLIT:-val}"
if [[ "$SPLIT" == test && -n "${SLURM_JOB_ID:-}" ]]; then
    if ! grep -q 'model name.*Platinum 8362 ' /proc/cpuinfo; then
        echo "Phase E requires Platinum 8362" >&2
        exit 42
    fi
fi
LEVELS=(4 8 16 32 64 128)
MAXITER=1000
OUT_DIR="${OUT_DIR:-data_results_v3/lshape_alg1_Tscan}"
REFERENCE="${REFERENCE:-references_v3/lshape_energy_ngsolve.csv}"
if [[ "${SMOKE:-0}" == 1 ]]; then
    [[ "$SPLIT" == val ]]
    LEVELS=(4 8)
    MAXITER=5
    HI=$(( LO + 1 ))
    OUT_DIR="${OUT_DIR}/smoke_${SLURM_ARRAY_JOB_ID:-local}"
fi
if [[ -n "${PROBE_N:-}" ]]; then
    [[ "${SMOKE:-0}" == 1 && "$PROBE_N" == 128 ]]
    LEVELS=(128)
    MAXITER=1
fi
for (( instance=LO; instance<HI; instance++ )); do
    "${PYTHON:-python}" -u 2D/scripts/alg1_lshape_Tscan.py --split "$SPLIT" --p "$P" --T "$T" \
        --levels "${LEVELS[@]}" --maxiter "$MAXITER" --instance-range "$instance" "$((instance+1))" \
        --tag "i$instance" --reference "$REFERENCE" --out-dir "$OUT_DIR"
done
