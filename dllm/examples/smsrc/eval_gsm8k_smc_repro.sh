#!/bin/bash
# Reproduction script for GSM8K SMC exactly matching reference settings
# Reference: code/srsmc/llada/eval_gsm8k_smc.sh

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model Selection
# model_path="GSAI-ML/LLaDA-8B-Instruct" 
# Or local path if preferred
model_type="llada"

# SMC Settings from reference
steps=8         # 256/32 = 8. This implies 1 step per block.
length=256
block_size=32
num_particles=4
temperature=1.0 # Reference uses 1.0 with float64
threshold=0.9   # Reference uses 0.9
use_smc=true

# Task Settings
task="gsm8k"
num_fewshot=5
batch_size=1

echo ">>> Running GSM8K Reproduction (Match Reference)"
echo "    Model: $model_type"
echo "    Steps: $steps (1 step/block)"
echo "    Temp: $temperature"
echo "    Threshold: $threshold"

python -m accelerate.commands.launch --num_processes 8 dllm/examples/smsrc/eval.py \
    --model ${model_type}_smc \
    --tasks $task \
    --num_fewshot $num_fewshot \
    --batch_size $batch_size \
    --apply_chat_template \
    --output_path "./results_repro" \
    --log_samples \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,max_new_tokens=${length},steps=${steps},block_size=${block_size},num_particles=${num_particles},use_smc=${use_smc},threshold=${threshold},temperature=${temperature}"
