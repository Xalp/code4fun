#!/bin/bash

# Dataset list: 
# gsm8k_cot, math500, humaneval_instruct_dream, mbpp_instruct_dream, ifeval

MODELS=("llada" "llada1.5" "dream")
# MODELS=("llada" "dream") # Start with these if 1.5 path is unsure

# We'll run two configurations per model
# 1. SMC: use_smc=true, temperature=1.0, particles=4
# 2. NoSMC: use_smc=false, temperature=0.1, particles=1

RESULT_BASE="./results_experiments"

for model in "${MODELS[@]}"; do
    # Define model path logic if needed, or rely on defaults in eval.sh
    # If llada1.5 path needs to be explicit:
    # if [ "$model" == "llada1.5" ]; then model_arg="--model_name_or_path GSAI-ML/LLaDA-1.5-8B-Instruct"; else model_arg=""; fi
    
    echo "========================================================"
    echo "Running experiments for Model: $model"
    echo "========================================================"

    # --- Configuration 1: SMC (Temp=1.0) ---
    TAG="${model}_smc_temp1.0"
    OUT_DIR="${RESULT_BASE}/${TAG}"
    echo ">> [SMC] Running $TAG -> $OUT_DIR"
    
    # Using default particles=4 for SMC
    bash dllm/examples/smsrc/eval.sh \
        --model_type $model \
        --use_smc true \
        --temperature 1.0 \
        --output_dir "$OUT_DIR" \
        --num_gpu 8 \
        # --limit 10 # Uncomment for testing

    # --- Configuration 2: No-SMC (Temp=0.1) ---
    TAG="${model}_nosmc_temp0.1"
    OUT_DIR="${RESULT_BASE}/${TAG}"
    echo ">> [No-SMC] Running $TAG -> $OUT_DIR"

    # use_smc=false implicitly sets particles=1 in eval.py
    bash dllm/examples/smsrc/eval.sh \
        --model_type $model \
        --use_smc false \
        --temperature 0.1 \
        --output_dir "$OUT_DIR" \
        --num_gpu 8 \
        # --limit 10 # Uncomment for testing

done

echo "All experiments completed."
