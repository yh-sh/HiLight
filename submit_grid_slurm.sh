#!/usr/bin/env bash
set -euo pipefail

# =======================
#  MODEL GRID
# =======================

# Main LLMs used for scoring / generation
MAIN_LLMS=(
  "Qwen/Qwen3-14B"
  "google/gemma-3-27b-it"
  "Qwen/Qwen3-32B"
  "meta-llama/Meta-Llama-3-70B-Instruct"
)

# Actor type: "lm" means LM-based emphasis actor
ACTOR_TYPE="lm"

# Emphasis actors (backbone LM for the actor)
ACTOR_MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
)

# =======================
#  GPU HEURISTIC
# =======================
# Decide how many GPUs to request based on the MAIN_LLM size.
gpus_needed() {
  local main="$1"

  # crude string-based heuristic; tweak for your cluster
  if [[ "$main" == *"70B"* || "$main" == *"70b"* ]]; then
    echo 8
  elif [[ "$main" == *"32B"* || "$main" == *"32b"* || "$main" == *"27B"* || "$main" == *"27b"* ]]; then
    echo 6
  elif [[ "$main" == *"14B"* || "$main" == *"14b"* ]]; then
    echo 4
  else
    echo 1
  fi
}

# =======================
#  SUBMIT JOBS
# =======================

for MAIN_LLM in "${MAIN_LLMS[@]}"; do
  for ACTOR_MODEL in "${ACTOR_MODELS[@]}"; do
    GPUS="$(gpus_needed "$MAIN_LLM")"

    # Safe job name (replace / with _)
    JOBNAME="emph_${MAIN_LLM//\//_}_actor_${ACTOR_MODEL//\//_}"

    echo "[INFO] Submitting job: ${JOBNAME}"
    echo "       MAIN_LLM    = ${MAIN_LLM}"
    echo "       ACTOR_TYPE  = ${ACTOR_TYPE}"
    echo "       ACTOR_MODEL = ${ACTOR_MODEL}"
    echo "       GPUS        = ${GPUS}"
    echo

    sbatch \
      --job-name="$JOBNAME" \
      --gpus="$GPUS" \
      --export=ALL,MAIN_LLM="$MAIN_LLM",ACTOR_TYPE="$ACTOR_TYPE",ACTOR_MODEL="$ACTOR_MODEL" \
      scripts/run_one.sbatch
  done
done
