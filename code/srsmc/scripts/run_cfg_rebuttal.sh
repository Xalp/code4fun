#!/usr/bin/env bash
# ===== CFG Rebuttal: Baseline vs CFG(1.5) vs SR-SMC(4) =====
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIMIT_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT_ARG="--limit $2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Wait for GPUs to be idle
bash "${SCRIPT_DIR}/wait.sh"

MODELS=("llada1.5" "dream")
TASKS=("gsm8k" "math500" "mbpp" "humaneval")

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo ">>> ${model} / ${task} / baseline"
    bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
      --use_smc false --num_particles 1 --temperature 1.0 \
      --output_dir "./results_cfg/${model}/${task}/nosmc_t1.0" \
      $LIMIT_ARG || true

    echo ">>> ${model} / ${task} / cfg=1.5"
    bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
      --use_smc false --num_particles 1 --temperature 1.0 --cfg_scale 1.5 \
      --output_dir "./results_cfg/${model}/${task}/cfg1.5_t1.0" \
      $LIMIT_ARG || true

    echo ">>> ${model} / ${task} / smc_p4"
    bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
      --use_smc true --num_particles 4 --temperature 1.0 \
      --output_dir "./results_cfg/${model}/${task}/smc_p4_t1.0" \
      $LIMIT_ARG || true
  done
done

echo "Done. Results in ./results_cfg/"
