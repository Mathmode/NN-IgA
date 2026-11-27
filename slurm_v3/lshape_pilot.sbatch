#!/bin/bash
#SBATCH --job-name=lshape_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --constraint=icelake
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --qos=bcam-exclusive
#SBATCH --output=logs/lshape_pilot_%A_%a.out
#SBATCH --error=logs/lshape_pilot_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PATH="$HOME/.conda/envs/${CONDA_ENV:-p1d_v2}/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false JAX_ENABLE_X64=true JAX_PLATFORMS=cpu
task="${SLURM_ARRAY_TASK_ID:-0}"
p=$((2 + task % 2))
v=$((task / 2))
phase="${PHASE:-pilot}"
base="${PILOT_ROOT:-data_results_v3/lshape_pilot}"
out="$base/V$v/p$p"
levels=(4 8 16 32)
features=()
extra=()
eval_extra=()
if (( v >= 1 )); then features+=(--level-feature); fi
if (( v >= 2 )); then features+=(--mesh-schedule data_results_v3/lshape_alg1_Tscan/validation_schedule.csv); fi
if (( v == 3 )); then levels+=(64); fi
case "$phase" in
    smoke) out="$base/smoke/V$v/p$p"; levels=(4 8); extra=(--smoke --n-train 2 --n-test 2 --batch-size 2 --epochs-per-level 1); eval_extra=(--smoke);;
    timing) [[ "$v" == 3 ]]; out="$base/timing/V$v/p$p"; levels=(64); extra=(--epochs-per-level 2);;
    pilot) ;;
    *) exit 2;;
esac
if (( v > 0 )); then
    "${PYTHON:-python}" -u 2D/scripts/train_lshape.py --p "$p" --seed 0 --protocol val --batch-size 16 \
        --lr-init 1e-3 --lr-end 1e-4 --N-levels "${levels[@]}" "${features[@]}" "${extra[@]}" --output-dir "$out"
    ck="$out/checkpoints/seed0/checkpoint_final.npz"
else
    ck="data_results/lshape/p$p/checkpoints/seed0/checkpoint_final.npz"
fi
if [[ "$phase" != timing ]]; then
    "${PYTHON:-python}" -u 2D/scripts/lshape_pilot_eval.py --p "$p" --variant "V$v" --checkpoint "$ck" \
        --output "$out/validation.csv" "${eval_extra[@]}"
fi
