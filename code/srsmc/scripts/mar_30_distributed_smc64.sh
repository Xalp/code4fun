#!/usr/bin/env bash
# Distributed SMC@64 on MATH (8 GPUs × 8 particles)
# Produces: SR-SMC@64, BoN@64, Pass@64 in two runs
set -e

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")/llada"

LIMIT_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT_ARG="--limit $2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

bash "${SCRIPT_DIR}/wait.sh"

OUT="./results_distributed"

# 1. SR-SMC@64 (with distributed resampling)
echo ">>> SR-SMC@64 on MATH"
torchrun --nproc_per_node=8 eval_math_distributed.py \
  --mode smc --local_particles 8 \
  --output_dir "${OUT}/smc64" \
  $LIMIT_ARG

# 2. BoN@64 / Pass@64 (no resampling, independent particles)
echo ">>> BoN@64 / Pass@64 on MATH"
torchrun --nproc_per_node=8 eval_math_distributed.py \
  --mode bon --local_particles 8 \
  --output_dir "${OUT}/bon64" \
  $LIMIT_ARG

echo "Done. Results in ${OUT}/"
