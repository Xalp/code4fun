#!/usr/bin/env bash
# ===== Temperature Ablation =====
# With SMC (particles=4)
# Temperatures: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
# Models: llada, llada1.5, dream
# Tasks: gsm8k, math500, mbpp, humaneval

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval.sh"

MODELS=("llada" "llada1.5" "dream")
TASKS=("gsm8k" "math500" "mbpp" "humaneval")
TEMPERATURES=("0.0" "0.2" "0.4" "0.6" "0.8" "1.0")
NUM_PARTICLES=4

echo "========================================="
echo "Temperature Ablation Study"
echo "SMC: enabled, Particles: ${NUM_PARTICLES}"
echo "Temperatures: ${TEMPERATURES[*]}"
echo "========================================="

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        for temp in "${TEMPERATURES[@]}"; do
            echo ""
            echo ">>> Running ${model} on ${task} with temp=${temp}..."
            bash "${EVAL_SCRIPT}" \
                --model_type "$model" \
                --task "$task" \
                --temperature "$temp" \
                --use_smc true \
                --num_particles "$NUM_PARTICLES"
        done
    done
done

echo ""
echo "========================================="
echo "Temperature Ablation Completed!"
echo "========================================="
