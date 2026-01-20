#!/usr/bin/env bash
# ===== Mandatory for proper import and evaluation =====
export PYTHONPATH=.:$PYTHONPATH             
export HF_ALLOW_CODE_EVAL=1                 # Allow code evaluation
export HF_DATASETS_TRUST_REMOTE_CODE=True   # For cmmlu dataset

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Enable async error handling
export NCCL_DEBUG=warn                      # Show NCCL warnings

# ===== Input Arguments =====
model_name_or_path="GSAI-ML/LLaDA-8B-Instruct"
num_gpu=8
num_particles=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)
      model_name_or_path="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
    --num_particles)
      num_particles="$2"; shift 2 ;;
    *) 
      echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

echo ">>> Running SMC Eval with ${num_particles} particles"

# Using 'llada_smc' which is registered in dllm/examples/smsrc/eval.py
# We use a simple task like gsm8k_cot for testing

python -m accelerate.commands.launch --num_processes "${num_gpu}" dllm/examples/smsrc/eval.py \
    --tasks gsm8k_cot --num_fewshot 0 \
    --model llada_smc \
    --apply_chat_template \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=512,steps=512,block_size=512,cfg=0.0,num_particles=${num_particles}" \
    --limit 2 # Limit to 2 samples for quick verification
