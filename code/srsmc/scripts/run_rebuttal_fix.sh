#!/usr/bin/env bash
# Re-run failed: MCMC + 4 seeds for both models
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIMIT_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT_ARG="--limit $2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

bash "${SCRIPT_DIR}/wait.sh"

MODELS=("llada1.5" "dream")
TASKS=("gsm8k" "math500" "mbpp" "humaneval")
OUT="./results_rebuttal"

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do

    # MCMC (LLaDA only)
    if [[ "$model" != "dream" ]]; then
      echo ">>> ${model} / ${task} / mcmc"
      bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
        --use_smc false --num_particles 1 --temperature 1.0 \
        --mcmc_steps 3 \
        --output_dir "${OUT}/${model}/${task}/mcmc3" \
        $LIMIT_ARG || true
    fi

    # 4 seeds
    for seed in 1 2 3 4; do
      echo ">>> ${model} / ${task} / seed=${seed}"
      bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
        --use_smc false --num_particles 1 --temperature 1.0 \
        --seed "$seed" --save \
        --output_dir "${OUT}/${model}/${task}/seed${seed}" \
        $LIMIT_ARG || true
    done

  done
done

echo "Done. Run: python aggregate_seeds.py --results_dir ${OUT}"
