#!/usr/bin/env bash
# ===== Quick Test for New Eval Tasks =====
# Runs all 3 models on all 4 new tasks with --limit 8
# Temperature = 0.0 (default for 0-shot in script is 1.0 but typically we might want greedy, 
# but user asked for run script which usually defaults to standard temp. 
# Re-reading prompt: "eval_new_test.sh, which run on 3 models, 4 datasets with limit=8"
# I will adhere to the defaults in eval_new.sh but just add limit.)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_new.sh"

MODELS=("llada" "llada1.5" "dream")
TASKS=("gsm8k" "math500" "mbpp" "humaneval")
LIMIT=8

echo "========================================="
echo "Running Quick Test for New Tasks"
echo "Models: ${MODELS[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Limit: ${LIMIT}"
echo "========================================="

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo ""
        echo ">>> Testing ${model} on ${task} (SMC)..."
        bash "${EVAL_SCRIPT}" \
            --model_type "$model" \
            --task "$task" \
            --use_smc true \
            --num_particles 4 \
            --limit "$LIMIT"
    done
done

echo ""
echo "========================================="
echo "Quick Test Completed!"
echo "========================================="
