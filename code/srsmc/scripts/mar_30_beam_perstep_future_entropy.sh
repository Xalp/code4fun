#!/usr/bin/env bash
# Mar 30: Per-step beam search + future entropy
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIMIT_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT_ARG="--limit $2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

bash "${SCRIPT_DIR}/wait.sh"

# --- Quick test: SMC p4 on gsm8k (not gsm8k-cot-dream), 5-shot ---
OUT="./results_rebuttal"
for model in llada1.5 dream; do
  echo ">>> ${model} / gsm8k_original / smc_p4"
  if [[ "$model" == "dream" ]]; then
    cd "$(dirname "$SCRIPT_DIR")/dream"
    model_path="Dream-org/Dream-v0-Instruct-7B"
    CMD="accelerate launch eval.py --model dream --tasks gsm8k --num_fewshot 5"
    CMD="${CMD} --output_path ${OUT}/${model}/gsm8k_orig/smc_p4 --log_samples --apply_chat_template"
    CMD="${CMD} --model_args \"pretrained=${model_path},max_new_tokens=256,diffusion_steps=8,temperature=1.0,threshold=0.9,use_cache=true,use_smc=true,num_particles=4,alg=confidence_threshold\""
  else
    cd "$(dirname "$SCRIPT_DIR")/llada"
    model_path="GSAI-ML/LLaDA-1.5"
    CMD="accelerate launch eval_llada.py --model llada_dist --tasks gsm8k --num_fewshot 5"
    CMD="${CMD} --output_path ${OUT}/${model}/gsm8k_orig/smc_p4 --log_samples --apply_chat_template"
    CMD="${CMD} --model_args \"model_path=${model_path},gen_length=256,steps=8,block_length=32,temperature=1.0,threshold=0.9,use_cache=True,use_smc=true\""
  fi
  CMD="${CMD} ${LIMIT_ARG}"
  echo "Running: $CMD"
  eval $CMD || true
done

MODELS=("llada1.5" "dream")
TASKS=("gsm8k" "math500" "humaneval" "mbpp")
OUT="./results_rebuttal"

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do

    # Per-step beam search (resample after every diffusion step)
    echo ">>> ${model} / ${task} / beam_perstep"
    bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
      --use_smc true --num_particles 4 --temperature 1.0 \
      --resample_strategy per_step \
      --output_dir "${OUT}/${model}/${task}/beam_perstep_p4" \
      $LIMIT_ARG || true

    # Future entropy: resample based on entropy of next block predictions
    echo ">>> ${model} / ${task} / future_entropy"
    bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
      --use_smc true --num_particles 4 --temperature 1.0 \
      --weight_type future_entropy \
      --output_dir "${OUT}/${model}/${task}/smc_p4_future_entropy" \
      $LIMIT_ARG || true

  done
done

echo "Done."
