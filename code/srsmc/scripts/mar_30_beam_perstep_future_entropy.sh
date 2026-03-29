#!/usr/bin/env bash
# Mar 30: Per-step beam search + future entropy
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIMIT_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT_ARG="--limit $2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

bash "${SCRIPT_DIR}/wait.sh"

MODELS=("llada1.5" "dream")
TASKS=("gsm8k" "math500" "humaneval" "mbpp")
OUT="./results_rebuttal"

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do

    # Per-step beam search (resample after every diffusion step)
    echo ">>> ${model} / ${task} / beam_perstep"
    bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
      --use_smc true --num_particles 4 --temperature 1.0 \
      --resample_strategy per_step \
      --output_dir "${OUT}/${model}/${task}/beam_perstep_p4" \
      $LIMIT_ARG || true

    # Future entropy: resample based on entropy of next block predictions
    echo ">>> ${model} / ${task} / future_entropy"
    bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
      --use_smc true --num_particles 4 --temperature 1.0 \
      --weight_type future_entropy \
      --output_dir "${OUT}/${model}/${task}/smc_p4_future_entropy" \
      $LIMIT_ARG || true

  done
done

echo "Done."
