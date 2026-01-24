#!/usr/bin/env bash
# ===== Dream SMC Evaluation Script =====
# This script is dedicated to Dream model evaluation with SMC support.
# It uses the proper SMC block generation with particles and resampling.

export PYTHONPATH=.:$PYTHONPATH             
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ===== Default Configuration =====
max_new_tokens=256
steps=256
block_size=32
temperature=1.0  # Default 1.0 for SMC
threshold="0.9"
use_smc=true
use_cache=true
model_name_or_path="Dream-org/Dream-v0-Instruct-7B"
num_gpu=8
limit=""
resample_freq=1
output_dir="./results_dream_smc"

# ===== Argument Parsing =====
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)
      model_name_or_path="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
    --limit)
      limit="$2"; shift 2 ;;
    --max_new_tokens)
      max_new_tokens="$2"; shift 2 ;;
    --steps)
      steps="$2"; shift 2 ;;
    --block_size)
      block_size="$2"; shift 2 ;;
    --threshold)
      threshold="$2"; shift 2 ;;
    --use_smc)
      use_smc="$2"; shift 2 ;;
    --use_cache)
      use_cache="$2"; shift 2 ;;
    --temperature)
      temperature="$2"; shift 2 ;;
    --resample_freq)
      resample_freq="$2"; shift 2 ;;
    --output_dir)
      output_dir="$2"; shift 2 ;;
    *) 
      echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

echo ">>> Running Dream SMC Eval"
echo "    Model: ${model_name_or_path}"
echo "    SMC Config: use_smc=${use_smc}, use_cache=${use_cache}, threshold=${threshold}, resample_freq=${resample_freq}"
echo "    Gen Config: max_new_tokens=${max_new_tokens}, steps=${steps}, block=${block_size}, temp=${temperature}"
echo "    Output Dir: ${output_dir}"

common_args="--model dream_smc --output_path ${output_dir} --log_samples --apply_chat_template"
if [ -n "$limit" ]; then
    common_args="$common_args --limit $limit"
fi

base_model_args="pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${block_size},use_smc=${use_smc},use_cache=${use_cache},threshold=${threshold},resample_freq=${resample_freq}"

CMD="python -m accelerate.commands.launch --num_processes ${num_gpu} dllm/examples/dream_smc/eval.py"

# ================
# Tasks
# ================

# GSM8K (5-shot)
$CMD --tasks gsm8k --num_fewshot 5 ${common_args} \
    --model_args "${base_model_args},temperature=${temperature},top_p=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False"

# MATH500 (4-shot)
$CMD --tasks minerva_math500 --num_fewshot 4 ${common_args} \
    --model_args "${base_model_args},temperature=${temperature},top_p=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False"

# HumanEval (0-shot)
$CMD --tasks humaneval --num_fewshot 0 ${common_args} \
    --model_args "${base_model_args},temperature=${temperature},top_p=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False" \
    --confirm_run_unsafe_code

# MBPP (3-shot)
$CMD --tasks mbpp --num_fewshot 3 ${common_args} \
    --model_args "${base_model_args},temperature=${temperature},top_p=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False" \
    --confirm_run_unsafe_code

# IFEval (0-shot)
$CMD --tasks ifeval --num_fewshot 0 ${common_args} \
    --model_args "${base_model_args},temperature=${temperature},top_p=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False"

echo "Dream SMC evaluation completed."

