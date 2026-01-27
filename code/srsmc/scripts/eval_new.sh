#!/usr/bin/env bash
# ===== New SMC Evaluation Script (0-shot for all) =====
# Models: llada, llada1.5, dream
# Tasks (all 0-shot, new prompt): gsm8k, math500, mbpp, humaneval

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
save="false"

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
    --save) save="true"; shift 1 ;;
    *) echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

# # ===== Compute Steps Based on Task =====
# if [[ "$task" == "math500" ]] || [[ "$task" == "mbpp" ]]; then
#     length=512
# fi
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

# ===== Set Task-Specific Parameters using NEW configs =====
# All are 0-shot now
num_fewshot=0
case "$task" in
    gsm8k) task_name="gsm8k-cot-dream"; num_fewshot=5 ;;
    math500) task_name="minerva_math500-new" ;;
    mbpp) task_name="mbpp-new" ;;
    humaneval) task_name="humaneval-instruct-new" ;;
    *) echo "Error: Unknown task: $task"; exit 1 ;;
esac

# ===== Set Output Directory =====
smc_label=""
if [[ "$use_smc" == "true" ]]; then
    smc_label="smc_p${num_particles}"
else
    smc_label="nosmc"
fi

if [[ -z "$output_dir" ]]; then
    output_dir="./results_new/${model_type}/${task}/${smc_label}_t${temperature}"
fi

if [[ "$save" == "true" ]] && [[ -z "$save_dir" ]]; then
    save_dir="${output_dir}/saved_generations"
fi

# ===== Print Configuration =====
echo "========================================="
echo "NEW SMC Evaluation Configuration"
echo "========================================="
echo "Model: ${model_type} (${model_path})"
echo "Task: ${task} -> ${task_name} (0-shot)"
echo "SMC: use_smc=${use_smc}, num_particles=${num_particles}"
echo "Gen Config: length=${length}, steps=${steps}, block=${block_length}, temp=${temperature}"
echo "Output: ${output_dir}"
echo "Save Dir: ${save_dir}"
echo "========================================="

# ===== Build Common Arguments =====
common_args="--output_path ${output_dir} --log_samples --apply_chat_template"
if [[ -n "$limit" ]]; then
    common_args="$common_args --limit $limit"
fi

# ===== Run Evaluation =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Note: We deliberately use the standard num_fewshot=0 for all these new tasks
if [[ "$model_type" == "dream" ]]; then
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
