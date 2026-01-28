#!/usr/bin/env bash
export SMC_TRACE_FILE="./smc_trace"
# Clear previous traces
rm -f ./smc_trace*.jsonl

# Run 10 samples of math500 (Minerva Math) with SMC
# Using default gsm8k-cot-dream like model settings but for math500
./code/srsmc/scripts/eval_new.sh \
    --task math500 \
    --limit 20 \
    --num_particles 4 \
    --temperature 1.0 \
    --output_dir ./results_ablation/math500_smc_analysis

# Analyze results
python3 analyze_smc.py
