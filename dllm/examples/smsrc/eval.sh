#!/usr/bin/env bash
# ===== Mandatory for proper import and evaluation =====
export PYTHONPATH=.:$PYTHONPATH             
export HF_ALLOW_CODE_EVAL=1                 # Allow code evaluation
export HF_DATASETS_TRUST_REMOTE_CODE=True   # For cmmlu dataset

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Enable async error handling for multi-GPU communication to avoid deadlocks
export NCCL_DEBUG=warn                      # Show NCCL warnings for better diagnosis without flooding logs
export TORCH_DISTRIBUTED_DEBUG=DETAIL       # Provide detailed logging for PyTorch distributed debugging

# ===== Default Configuration Variables =====
# These can be overridden by command line arguments
length=256
steps=256
block_length=256
num_particles=4
confidence_threshold="" # Set to a float (e.g. 0.9) to enable validity check
use_smc=true
model_type="llada"
model_name_or_path=""
instruct=true
num_gpu=8
limit=""

# ===== Argument Parsing =====
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_type)
      model_type="$2"; shift 2 ;;
    --model_name_or_path)
      model_name_or_path="$2"; shift 2 ;;
    --instruct)
      instruct="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
    --num_particles)
      num_particles="$2"; shift 2 ;;
    --limit)
      limit="$2"; shift 2 ;;
    --length)
      length="$2"; shift 2 ;;
    --steps)
      steps="$2"; shift 2 ;;
    --block_length)
      block_length="$2"; shift 2 ;;
    --confidence_threshold)
      confidence_threshold="$2"; shift 2 ;;
    --use_smc)
      use_smc="$2"; shift 2 ;;
    *) 
      echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

# Set default model path if not provided
if [ -z "$model_name_or_path" ]; then
    if [ "$model_type" == "llada" ]; then
        model_name_or_path="GSAI-ML/LLaDA-8B-Instruct"
    elif [ "$model_type" == "dream" ]; then
        model_name_or_path="Dream-org/Dream-v0-Instruct-7B"
    else
        echo "Error: Must provide --model_name_or_path for custom model type $model_type"
        exit 1
    fi
fi

echo ">>> Running SMC Eval"
echo "    Model: ${model_type} (${model_name_or_path})"
echo "    SMC Config: particles=${num_particles}, use_smc=${use_smc}, threshold=${confidence_threshold}"
echo "    Gen Config: length=${length}, steps=${steps}, block=${block_length}"

# Construct model_args string
# Note: we use ${model_type}_smc which must be registered in eval.py
common_args="--model ${model_type}_smc"
if [ "$instruct" = "true" ]; then
    common_args="$common_args --apply_chat_template"
fi
if [ -n "$limit" ]; then
    common_args="$common_args --limit $limit"
fi

# Base model_args passed to generic tasks
base_model_args="pretrained=${model_name_or_path},max_new_tokens=${length},steps=${steps},block_length=${block_length},num_particles=${num_particles},use_smc=${use_smc},output_path="./results",log_samples"

if [ -n "$confidence_threshold" ]; then
    base_model_args="${base_model_args},threshold=${confidence_threshold}"
fi

# Define Python command
CMD="python -m accelerate.commands.launch --num_processes ${num_gpu} dllm/examples/smsrc/eval.py"

# ================
# Tasks
# ================

if [ "$instruct" = "true" ]; then
    # Instruct Tasks
    
    $CMD --tasks mmlu_generative_dream --num_fewshot 4 ${common_args} \
        --model_args "${base_model_args},temperature=0.1,top_p=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False"

    $CMD --tasks mmlu_pro --num_fewshot 4 ${common_args} \
        --model_args "${base_model_args},temperature=0.1,top_p=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False"

    $CMD --tasks gsm8k_cot --num_fewshot 0 ${common_args} \
        --model_args "${base_model_args},temperature=0.1,top_p=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False"

    # REPLACED minerva_math with math500
    $CMD --tasks math500 --num_fewshot 0 ${common_args} \
        --model_args "${base_model_args},temperature=0.1,top_p=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False"

    $CMD --tasks gpqa_main_n_shot --num_fewshot 5 ${common_args} \
        --model_args "${base_model_args},mc_num=32"

    $CMD --tasks humaneval_instruct_dream --num_fewshot 0 ${common_args} \
        --model_args "${base_model_args},temperature=0.1,top_p=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False" \
        --confirm_run_unsafe_code

    $CMD --tasks mbpp_instruct_dream --num_fewshot 0 ${common_args} \
        --model_args "${base_model_args},temperature=0.1,top_p=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False" \
        --confirm_run_unsafe_code

    $CMD --tasks ifeval --num_fewshot 0 ${common_args} \
        --model_args "${base_model_args},temperature=0.1,top_p=0.9,dtype=bfloat16,add_bos_token=False,escape_until=False"

else
    # Base Tasks
    
    $CMD --tasks humaneval --num_fewshot 0 ${common_args} \
        --model_args "${base_model_args},temperature=0.2,top_p=0.95,add_bos_token=True,escape_until=False" \
        --confirm_run_unsafe_code

    $CMD --tasks gsm8k_cot --num_fewshot 8 ${common_args} \
        --model_args "${base_model_args},temperature=0.0,top_p=0.95,add_bos_token=True,escape_until=False"

    $CMD --tasks mbpp --num_fewshot 3 ${common_args} \
        --model_args "${base_model_args},temperature=0.2,top_p=0.95,add_bos_token=True,escape_until=False" \
        --confirm_run_unsafe_code

    # REPLACED minerva_math with math500
    $CMD --tasks math500 --num_fewshot 4 ${common_args} \
        --model_args "${base_model_args},temperature=0.0,top_p=0.95,add_bos_token=True,escape_until=False"

    $CMD --tasks bbh --num_fewshot 3 ${common_args} \
        --model_args "${base_model_args},temperature=0.0,top_p=0.95,add_bos_token=True,escape_until=False"

    $CMD --tasks mmlu --num_fewshot 5 ${common_args} \
        --model_args "${base_model_args},add_bos_token=True"

    $CMD --tasks arc_easy --num_fewshot 0 ${common_args} \
        --model_args "${base_model_args},add_bos_token=True"

    $CMD --tasks arc_challenge --num_fewshot 0 ${common_args} \
        --model_args "${base_model_args},add_bos_token=True"

    $CMD --tasks hellaswag --num_fewshot 0 ${common_args} \
        --model_args "${base_model_args},add_bos_token=True"

    $CMD --tasks piqa --num_fewshot 0 ${common_args} \
        --model_args "${base_model_args},add_bos_token=True"

    $CMD --tasks gpqa_main_n_shot --num_fewshot 5 ${common_args} \
        --model_args "${base_model_args},add_bos_token=True"

    $CMD --tasks winogrande --num_fewshot 5 ${common_args} \
        --model_args "${base_model_args},add_bos_token=True"

    $CMD --tasks race --num_fewshot 0 ${common_args} \
        --model_args "${base_model_args},add_bos_token=True"
fi
