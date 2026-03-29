#!/usr/bin/env bash
# Re-run Dream entropy & margin with weight_type fix
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
TASKS=("gsm8k" "math500" "humaneval" "mbpp")

for task in "${TASKS[@]}"; do
  echo ">>> dream / ${task} / entropy"
  bash "${SCRIPT_DIR}/eval.sh" --model_type dream --task "$task" \
    --use_smc true --num_particles 4 --temperature 1.0 \
    --weight_type entropy \
    --output_dir "${OUT}/dream/${task}/smc_p4_entropy" \
    $LIMIT_ARG || true

  echo ">>> dream / ${task} / margin"
  bash "${SCRIPT_DIR}/eval.sh" --model_type dream --task "$task" \
    --use_smc true --num_particles 4 --temperature 1.0 \
    --weight_type margin \
    --output_dir "${OUT}/dream/${task}/smc_p4_margin" \
    $LIMIT_ARG || true
done

echo "Done."
