#!/bin/bash

# Ensure we are in the repo root
if [ ! -d "dllm/examples/smsrc" ]; then
    echo "Error: Please run this script from the repository root (srsmc/)."
    exit 1
fi

echo "==========================================="
echo "Running LLaDA Integration Test (Limit 8)"
echo "==========================================="
bash dllm/examples/smsrc/eval.sh --model_type llada --limit 8 --num_gpu 8

echo ""
echo "==========================================="
echo "Running Dream Integration Test (Limit 8)"
echo "==========================================="
bash dllm/examples/smsrc/eval.sh --model_type dream --limit 8 --num_gpu 8

echo ""
echo "Integration tests initiated."
