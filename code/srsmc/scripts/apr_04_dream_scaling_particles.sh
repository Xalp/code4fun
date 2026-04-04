#!/usr/bin/env bash
# Test Dream single-GPU particle scaling: 8, 16, 32 particles
# 8 GPUs in parallel via accelerate (data parallel, each GPU handles different questions)
# MATH-500, 4-shot
set -e
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIMIT_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT_ARG="--limit $2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

bash "${SCRIPT_DIR}/wait.sh"

OUT="./results_rebuttal/dream/math500"

for np in 8 16 32; do
  echo ">>> Dream / math500 / smc_p${np}"
  bash "${SCRIPT_DIR}/eval.sh" --model_type dream --task math500 \
    --use_smc true --num_particles "$np" --temperature 1.0 \
    --output_dir "${OUT}/smc_p${np}" \
    $LIMIT_ARG || true
done

echo "Done."
