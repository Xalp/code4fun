#!/usr/bin/env bash
# ===== Particles Ablation =====
# Temperature = 1.0
# Particles: 2, 3 (1 and 4 are tested in main experiments)
# Models: llada, llada1.5, dream
# Tasks: gsm8k, math500, mbpp, humaneval

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval.sh"

MODELS=("llada" "llada1.5" "dream")
TASKS=("gsm8k" "math500" "mbpp" "humaneval")
PARTICLES=("2" "3")
TEMPERATURE=1.0

echo "========================================="
echo "Particles Ablation Study"
echo "Temperature: ${TEMPERATURE}"
echo "Particles: ${PARTICLES[*]} (1 and 4 in main experiments)"
echo "========================================="

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        for np in "${PARTICLES[@]}"; do
            echo ""
            echo ">>> Running ${model} on ${task} with particles=${np}..."
            bash "${EVAL_SCRIPT}" \
                --model_type "$model" \
                --task "$task" \
                --temperature "$TEMPERATURE" \
                --use_smc true \
                --num_particles "$np"
        done
    done
done

echo ""
echo "========================================="
echo "Particles Ablation Completed!"
echo "========================================="
