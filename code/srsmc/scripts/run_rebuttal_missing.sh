#!/usr/bin/env bash
# Re-run missing experiments:
# 1. LLaDA-1.5 CFG(1.5): gsm8k, humaneval, mbpp
# 2. Dream MCMC(3): all 4 tasks
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIMIT_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT_ARG="--limit $2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

bash "${SCRIPT_DIR}/wait.sh"

OUT="./results_rebuttal"

# --- LLaDA-1.5 CFG(1.5) missing tasks ---
for task in gsm8k humaneval mbpp; do
  echo ">>> llada1.5 / ${task} / cfg=1.5"
  bash "${SCRIPT_DIR}/eval.sh" --model_type llada1.5 --task "$task" \
    --use_smc false --num_particles 1 --temperature 1.0 --cfg_scale 1.5 \
    --output_dir "${OUT}/llada1.5/${task}/cfg1.5" \
    $LIMIT_ARG || true
done

# --- Dream MCMC(3) all tasks ---
for task in gsm8k math500 humaneval mbpp; do
  echo ">>> dream / ${task} / mcmc3"
  bash "${SCRIPT_DIR}/eval.sh" --model_type dream --task "$task" \
    --use_smc false --num_particles 1 --temperature 1.0 --mcmc_steps 3 \
    --output_dir "${OUT}/dream/${task}/mcmc3" \
    $LIMIT_ARG || true
done

echo "Done."
