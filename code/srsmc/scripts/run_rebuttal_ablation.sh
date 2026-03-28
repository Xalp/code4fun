#!/usr/bin/env bash
# ===== Rebuttal Ablations =====
# 1. SMC Pass@4: return all particles, save for offline pass@4
# 2. Resample frequency: every 2 blocks, every 4 blocks (vs default every 1)
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

    # --- SMC Pass@4: save all 4 particles ---
    echo ">>> ${model} / ${task} / smc_p4_all"
    bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
      --use_smc true --num_particles 4 --temperature 1.0 \
      --return_all --save \
      --output_dir "${OUT}/${model}/${task}/smc_p4_all" \
      $LIMIT_ARG || true

    # --- Resample every 2 blocks ---
    echo ">>> ${model} / ${task} / smc_p4_freq2"
    bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
      --use_smc true --num_particles 4 --temperature 1.0 \
      --resample_freq 2 \
      --output_dir "${OUT}/${model}/${task}/smc_p4_freq2" \
      $LIMIT_ARG || true

    # --- Resample every 4 blocks ---
    echo ">>> ${model} / ${task} / smc_p4_freq4"
    bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
      --use_smc true --num_particles 4 --temperature 1.0 \
      --resample_freq 4 \
      --output_dir "${OUT}/${model}/${task}/smc_p4_freq4" \
      $LIMIT_ARG || true

  done
done

echo "========================================="
echo "Done. Results in: ${OUT}/"
echo ""
echo "For SMC Pass@4: check saved_generations/ in smc_p4_all dirs"
echo "  Each question has 4 particle outputs saved."
echo "========================================="
