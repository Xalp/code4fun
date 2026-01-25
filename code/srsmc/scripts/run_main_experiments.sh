#!/usr/bin/env bash
# ===== Main Experiments =====
# Temperature = 1.0
# Compare: No-SMC (particles=1) vs SMC (particles=4)
# Models: llada, llada1.5, dream
# Tasks: gsm8k, math500, mbpp, humaneval

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval.sh"

MODELS=("llada" "llada1.5" "dream")
TASKS=("gsm8k" "math500" "mbpp" "humaneval")
TEMPERATURE=1.0

echo "========================================="
echo "Main Experiments: SMC vs No-SMC"
echo "Temperature: ${TEMPERATURE}"
echo "========================================="

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo ""
        echo ">>> Running ${model} on ${task} without SMC..."
        bash "${EVAL_SCRIPT}" \
            --model_type "$model" \
            --task "$task" \
            --temperature "$TEMPERATURE" \
            --use_smc false \
            --num_particles 1

        echo ""
        echo ">>> Running ${model} on ${task} with SMC (4 particles)..."
        bash "${EVAL_SCRIPT}" \
            --model_type "$model" \
            --task "$task" \
            --temperature "$TEMPERATURE" \
            --use_smc true \
            --num_particles 4
    done
done

echo ""
echo "========================================="
echo "Main Experiments Completed!"
echo "========================================="
