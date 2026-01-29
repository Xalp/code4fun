#!/usr/bin/env bash
mkdir -p timing_logs

echo "Running LLaDA No-SMC..."
./code/srsmc/scripts/eval.sh \
    --task math500 --model_type llada --use_smc false --show_speed \
    > timing_logs/llada_nosmc.log 2>&1

echo "Running LLaDA SMC..."
./code/srsmc/scripts/eval.sh \
    --task math500 --model_type llada --use_smc true --show_speed \
    > timing_logs/llada_smc.log 2>&1

echo "Running Dream No-SMC..."
./code/srsmc/scripts/eval.sh \
    --task math500 --model_type dream --use_smc false --show_speed \
    > timing_logs/dream_nosmc.log 2>&1

echo "Running Dream SMC..."
./code/srsmc/scripts/eval.sh \
    --task math500 --model_type dream --use_smc true --show_speed \
    > timing_logs/dream_smc.log 2>&1

echo "All timing runs complete. Parsing results..."
python3 summarize_time.py
