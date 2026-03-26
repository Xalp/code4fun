#!/usr/bin/env bash
# ===== Re-run Dream CFG only =====
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIMIT_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT_ARG="--limit $2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

bash "${SCRIPT_DIR}/wait.sh"

MODEL="dream"
TASKS=("gsm8k" "math500" "mbpp" "humaneval")

for task in "${TASKS[@]}"; do
  echo ">>> ${MODEL} / ${task} / cfg=1.5"
  bash "${SCRIPT_DIR}/eval.sh" --model_type "$MODEL" --task "$task" \
    --use_smc false --num_particles 1 --temperature 1.0 --cfg_scale 1.5 \
    --output_dir "./results_cfg/${MODEL}/${task}/cfg1.5_t1.0" \
    $LIMIT_ARG || true
done

echo "Done. Results in ./results_cfg/${MODEL}/"
