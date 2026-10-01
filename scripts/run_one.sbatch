#!/bin/bash
#SBATCH --job-name=llm-emph-one
#SBATCH --time=8-00:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
# DO NOT set --gpus here; launcher passes it via sbatch
#SBATCH --output=logs/rec/%x-%j.out
#SBATCH --error=logs/rec/%x-%j.err

set -euo pipefail

mkdir -p logs/rec ckpt_emph

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_emph

echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[INFO] MAIN_LLM=$MAIN_LLM"
echo "[INFO] ACTOR_TYPE=$ACTOR_TYPE"
echo "[INFO] ACTOR_MODEL=${ACTOR_MODEL:-}"
echo "[INFO] SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-${SLURM_GPUS:-unknown}}"
nvidia-smi || true

LLM_INPUTS="data/lm_inputs.jsonl"
TRAIN_LABELS="data/train_split.jsonl"
TEST_LABELS="data/test_split.jsonl"
ADJ_PATH="data/covis_adj.jsonl"
META_PATH="data/compact_meta.jsonl"

TRAIN_STEPS="${TRAIN_STEPS:-10000}"
TOPK="${TOPK:-10}"
CANDIDATES="${CANDIDATES:-40}"
SAVE_ROOT="${SAVE_ROOT:-ckpt_emph}"
ALGO="${ALGO:-GRPO}"

# Which logical GPU index your code uses (within CUDA_VISIBLE_DEVICES)
GPU_INDEX="${GPU_INDEX:-0}"

RUN_NAME="beauty/${MAIN_LLM//\//_}/${ALGO}/actor-${ACTOR_TYPE}"
if [[ "${ACTOR_TYPE}" == "lm" ]]; then
  RUN_NAME="beauty/${MAIN_LLM//\//_}/${ALGO}/actor-${ACTOR_MODEL//\//_}"
fi
SAVE_DIR="${SAVE_ROOT}/${RUN_NAME}"
mkdir -p "$SAVE_DIR"

echo "[INFO] Training -> $SAVE_DIR"

# ---- TRAIN ----
stdbuf -oL -eL python -u -m rec.train_emphasis \
  --llm-inputs "${LLM_INPUTS}" \
  --labels "${TRAIN_LABELS}" \
  --adj "${ADJ_PATH}" \
  --train-steps "${TRAIN_STEPS}" \
  --topk "${TOPK}" \
  --candidates "${CANDIDATES}" \
  --meta "${META_PATH}" \
  --model-id "${MAIN_LLM}" \
  --save_dir "${SAVE_DIR}" \
  --algorithm "${ALGO}" \
  --actor "${ACTOR_TYPE}" \
  ${ACTOR_MODEL:+--actor-model-id "${ACTOR_MODEL}"} \
  --gpu "${GPU_INDEX}"

# ---- OPTIONAL EVAL ----
# if [[ "${DO_EVAL:-1}" == "1" ]]; then
#   PER_BIN="${PER_BIN:-250}"
#   ACTOR_PARAM="${SAVE_DIR}/actor_emph_step${TRAIN_STEPS}.pt"

#   echo "[INFO] Eval -> $ACTOR_PARAM"
#   stdbuf -oL -eL python -u -m rec.test_emphasis \
#     --llm-inputs "${LLM_INPUTS}" \
#     --labels "${TEST_LABELS}" \
#     --adj "${ADJ_PATH}" \
#     --meta "${META_PATH}" \
#     --model-id "${MAIN_LLM}" \
#     --actor_param "${ACTOR_PARAM}" \
#     --actor "${ACTOR_TYPE}" \
#     ${ACTOR_MODEL:+--actor-model-id "${ACTOR_MODEL}"} \
#     --candidates "${CANDIDATES}" \
#     --topk "${TOPK}" \
#     --per-bin "${PER_BIN}" \
#     --gpu "${GPU_INDEX}"
# fi
