#!/usr/bin/env bash
# Full scaling experiments for Dream-7B on MATH-500 and GSM8K
#
# Part 1: Generate 32 independent samples → compute Pass@K, BoN@K, MajVote@K for K=8,16,32
# Part 2: SR-SMC@8/16/32 (already in apr_04_dream_scaling_particles.sh for MATH)
set -e
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAM_DIR="$(dirname "$SCRIPT_DIR")/dream"

LIMIT_ARG=""
LIMIT_PY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT_ARG="--limit $2"; LIMIT_PY="--limit $2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

bash "${SCRIPT_DIR}/wait.sh"

# ============================
# Part 1: Independent samples (Pass@K, BoN@K, MajVote@K)
# ============================
cd "$DREAM_DIR"

echo ">>> MATH-500: 32 independent samples"
torchrun --nproc_per_node=8 eval_scaling.py \
  --task minerva_math500 --total_samples 32 --batch_size 8 \
  --n_shot 4 --output_dir ./results_scaling/math500 \
  $LIMIT_PY

echo ">>> GSM8K: 32 independent samples"
torchrun --nproc_per_node=8 eval_scaling.py \
  --task gsm8k-cot-dream --total_samples 32 --batch_size 8 \
  --n_shot 5 --output_dir ./results_scaling/gsm8k \
  $LIMIT_PY

# ============================
# Part 2: SR-SMC@8/16/32
# ============================

# MATH SR-SMC (reuse apr_04 script if not already run)
for np in 8 16 32; do
  echo ">>> Dream / math500 / smc_p${np}"
  bash "${SCRIPT_DIR}/eval.sh" --model_type dream --task math500 \
    --use_smc true --num_particles "$np" --temperature 1.0 \
    --output_dir "./results_rebuttal/dream/math500/smc_p${np}" \
    $LIMIT_ARG || true
done

# GSM8K SR-SMC
for np in 8 16 32; do
  echo ">>> Dream / gsm8k / smc_p${np}"
  bash "${SCRIPT_DIR}/eval.sh" --model_type dream --task gsm8k \
    --use_smc true --num_particles "$np" --temperature 1.0 \
    --output_dir "./results_rebuttal/dream/gsm8k/smc_p${np}" \
    $LIMIT_ARG || true
done

echo "========================================="
echo "Done. Check:"
echo "  Independent: $DREAM_DIR/results_scaling/"
echo "  SR-SMC:      $DREAM_DIR/../scripts/results_rebuttal/dream/"
echo "========================================="
