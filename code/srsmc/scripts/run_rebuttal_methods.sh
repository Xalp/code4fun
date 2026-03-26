#!/usr/bin/env bash
# ===== Rebuttal: All methods comparison =====
# Baseline / BoN / Beam / MCMC / SR-SMC (all compute-matched at ~4x)
# + 4-seed runs for MajVote / Pass@4
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

MODELS=("llada1.5" "dream")
TASKS=("gsm8k" "math500" "mbpp" "humaneval")
OUT="./results_rebuttal"

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do

    # 1. Baseline (1 particle, no SMC)
    echo ">>> ${model} / ${task} / baseline"
    bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
      --use_smc false --num_particles 1 --temperature 1.0 \
      --output_dir "${OUT}/${model}/${task}/baseline" \
      $LIMIT_ARG || true

    # 2. Best-of-N (4 particles, no resampling, pick best by confidence)
    echo ">>> ${model} / ${task} / bon"
    bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
      --use_smc true --num_particles 4 --temperature 1.0 \
      --resample_strategy never \
      --output_dir "${OUT}/${model}/${task}/bon_p4" \
      $LIMIT_ARG || true

    # 3. Beam Search (4 particles, deterministic top-1 selection)
    echo ">>> ${model} / ${task} / beam"
    bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
      --use_smc true --num_particles 4 --temperature 1.0 \
      --resample_strategy deterministic \
      --output_dir "${OUT}/${model}/${task}/beam_p4" \
      $LIMIT_ARG || true

    # 4. MCMC / Gibbs (3 refinement steps ≈ 4x compute)
    if [[ "$model" != "dream" ]]; then
      # MCMC only implemented for LLaDA
      echo ">>> ${model} / ${task} / mcmc"
      bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
        --use_smc false --num_particles 1 --temperature 1.0 \
        --mcmc_steps 3 \
        --output_dir "${OUT}/${model}/${task}/mcmc3" \
        $LIMIT_ARG || true
    fi

    # 5. SR-SMC (4 particles, adaptive resampling)
    echo ">>> ${model} / ${task} / smc"
    bash "${SCRIPT_DIR}/eval.sh" --model_type "$model" --task "$task" \
      --use_smc true --num_particles 4 --temperature 1.0 \
      --output_dir "${OUT}/${model}/${task}/smc_p4" \
      $LIMIT_ARG || true

    # 6. Majority Vote / Pass@4: 4 independent seeds
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

echo "========================================="
echo "All rebuttal experiments done."
echo "Results in: ${OUT}/"
echo ""
echo "Next: run aggregate_seeds.py to compute MajVote/Pass@4"
echo "========================================="
