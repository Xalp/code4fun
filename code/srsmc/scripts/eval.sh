#!/usr/bin/env bash
# ===== SMC Evaluation Script =====
# Unified evaluation script for LLaDA, LLaDA-1.5, and Dream models
# Supports: gsm8k (5-shot), math500 (4-shot), mbpp (3-shot), humaneval (0-shot)

set -e

# ===== Environment Variables =====
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== Default Configuration =====
model_type="llada"
task="gsm8k"
temperature=1.0
use_smc=true
num_particles=4
length=256
block_length=32
threshold=0.9
output_dir=""
limit=""
save_dir=""

# ===== Argument Parsing =====
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_type) model_type="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    --temperature) temperature="$2"; shift 2 ;;
    --use_smc) use_smc="$2"; shift 2 ;;
    --num_particles) num_particles="$2"; shift 2 ;;
    --length) length="$2"; shift 2 ;;
    --block_length) block_length="$2"; shift 2 ;;
    --threshold) threshold="$2"; shift 2 ;;
    --output_dir) output_dir="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --save_dir) save_dir="$2"; shift 2 ;;
    *) echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

# ===== Compute Steps Based on Task =====
# Use longer generation for math/mbpp
if [[ "$task" == "math500" ]] || [[ "$task" == "mbpp" ]]; then
    length=512
fi
steps=$((length / block_length))

# ===== Set Model Path =====
if [[ "$model_type" == "llada" ]]; then
    model_path="GSAI-ML/LLaDA-8B-Instruct"
elif [[ "$model_type" == "llada1.5" ]]; then
    model_path="GSAI-ML/LLaDA-1.5"
elif [[ "$model_type" == "dream" ]]; then
    model_path="Dream-org/Dream-v0-Instruct-7B"
else
    echo "Error: Unknown model_type: $model_type (must be llada, llada1.5, or dream)"
    exit 1
fi

# ===== Set Task-Specific Parameters =====
case "$task" in
    gsm8k) num_fewshot=5; task_name="gsm8k" ;;
    math500) num_fewshot=4; task_name="minerva_math500" ;;
    mbpp)
        if [[ "$model_type" == "dream" ]]; then
            num_fewshot=3; task_name="mbpp_instruct_dream"
        else
            num_fewshot=3; task_name="mbpp_llada_instruct"
        fi
        ;;
    humaneval)
        if [[ "$model_type" == "dream" ]]; then
            num_fewshot=0; task_name="humaneval_instruct_dream"
        else
            num_fewshot=0; task_name="humaneval_instruct"
        fi
        ;;
    *) echo "Error: Unknown task: $task (must be gsm8k, math500, mbpp, or humaneval)"; exit 1 ;;
esac

# ===== Set Output Directory =====
smc_label=""
if [[ "$use_smc" == "true" ]]; then
    smc_label="smc_p${num_particles}"
else
    smc_label="nosmc"
fi

if [[ -z "$output_dir" ]]; then
    output_dir="./results/${model_type}/${task}/${smc_label}_t${temperature}"
fi

# ===== Print Configuration =====
echo "========================================="
echo "SMC Evaluation Configuration"
echo "========================================="
echo "Model: ${model_type} (${model_path})"
echo "Task: ${task} (${task_name}, ${num_fewshot}-shot)"
echo "SMC: use_smc=${use_smc}, num_particles=${num_particles}"
echo "Gen Config: length=${length}, steps=${steps}, block=${block_length}, temp=${temperature}"
echo "Output: ${output_dir}"
echo "========================================="

# ===== Build Common Arguments =====
common_args="--output_path ${output_dir} --log_samples --apply_chat_template"
if [[ -n "$limit" ]]; then
    common_args="$common_args --limit $limit"
fi

# ===== Check if Results Already Exist =====
# Skip evaluation if results.json already exists in output directory
# if [[ -d "$output_dir" ]]; then
#     # Check for any results.json file in the output directory
#     if ls "${output_dir}"/**/results*.json 1> /dev/null 2>&1 || ls "${output_dir}"/results*.json 1> /dev/null 2>&1; then
#         echo "========================================="
#         echo "[SKIP] Results already exist in: ${output_dir}"
#         echo "========================================="
#         exit 0
#     fi
# fi

# ===== Run Evaluation =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

if [[ "$model_type" == "dream" ]]; then
    # Dream model
    cd "$BASE_DIR/dream"
    
    model_args="pretrained=${model_path},max_new_tokens=${length},diffusion_steps=${steps}"
    model_args="${model_args},temperature=${temperature},threshold=${threshold}"
    model_args="${model_args},use_cache=true,use_smc=${use_smc},num_particles=${num_particles}"
    model_args="${model_args},alg=confidence_threshold"
    
    if [[ -n "$save_dir" ]]; then
        model_args="${model_args},save_dir=${save_dir}"
    fi
    
    CMD="accelerate launch eval.py --model dream --tasks ${task_name} --num_fewshot ${num_fewshot}"
    CMD="${CMD} ${common_args} --model_args \"${model_args}\""
    
    if [[ "$task" == "humaneval" ]] || [[ "$task" == "mbpp" ]]; then
        CMD="${CMD} --confirm_run_unsafe_code"
    fi
    
    echo "Running: $CMD"
    eval $CMD
else
    # LLaDA or LLaDA-1.5
    cd "$BASE_DIR/llada"
    
    model_args="model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length}"
    model_args="${model_args},temperature=${temperature},threshold=${threshold}"
    model_args="${model_args},use_cache=True,use_smc=${use_smc}"
    
    if [[ -n "$save_dir" ]]; then
        model_args="${model_args},save_dir=${save_dir}"
    fi
    
    CMD="accelerate launch eval_llada.py --model llada_dist --tasks ${task_name} --num_fewshot ${num_fewshot}"
    CMD="${CMD} ${common_args} --model_args \"${model_args}\""
    
    if [[ "$task" == "humaneval" ]] || [[ "$task" == "mbpp" ]]; then
        CMD="${CMD} --confirm_run_unsafe_code"
    fi
    
    echo "Running: $CMD"
    eval $CMD
fi

echo "========================================="
echo "Evaluation complete. Results saved to: ${output_dir}"
echo "========================================="
