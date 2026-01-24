#!/bin/bash

# ===== Dream SMC Experiments =====
# This script runs Dream model experiments with SMC vs No-SMC configurations.
# 
# Dataset list: 
# gsm8k, minerva_math500, humaneval, mbpp, ifeval

# Dream model variants to test
MODELS=(
    "Dream-org/Dream-v0-Instruct-7B"
    # Add more Dream variants here if needed
)

# We'll run two configurations per model
# 1. SMC: use_smc=true, temperature=1.0, particles=4 (default)
# 2. NoSMC: use_smc=false, temperature=0.1

RESULT_BASE="./results_dream_experiments"

for model_path in "${MODELS[@]}"; do
    # Extract model name from path for tagging
    model_name=$(basename "$model_path" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    
    echo "========================================================"
    echo "Running experiments for Model: $model_path"
    echo "========================================================"

    # --- Configuration 1: SMC (Temp=1.0) ---
    TAG="${model_name}_smc_temp1.0"
    OUT_DIR="${RESULT_BASE}/${TAG}"
    echo ">> [SMC] Running $TAG -> $OUT_DIR"
    
    bash dllm/examples/dream_smc/eval.sh \
        --model_name_or_path "$model_path" \
        --use_smc true \
        --use_cache true \
        --temperature 1.0 \
        --threshold 0.9 \
        --resample_freq 1 \
        --output_dir "$OUT_DIR" \
        --num_gpu 8
        # --limit 10 # Uncomment for testing

    # --- Configuration 2: No-SMC (Temp=0.1) ---
    TAG="${model_name}_nosmc_temp0.1"
    OUT_DIR="${RESULT_BASE}/${TAG}"
    echo ">> [No-SMC] Running $TAG -> $OUT_DIR"

    bash dllm/examples/dream_smc/eval.sh \
        --model_name_or_path "$model_path" \
        --use_smc false \
        --use_cache true \
        --temperature 0.1 \
        --output_dir "$OUT_DIR" \
        --num_gpu 8
        # --limit 10 # Uncomment for testing

done

echo "All Dream SMC experiments completed."

