#!/usr/bin/env bash
set -euo pipefail

# =======================
#  ENV / PATHS
# =======================
mkdir -p logs/rec ckpt_emph

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_emph

LLM_INPUTS="${LLM_INPUTS:-data/lm_inputs.jsonl}"
TRAIN_LABELS="${TRAIN_LABELS:-data/train_split.jsonl}"
ADJ_PATH="${ADJ_PATH:-data/covis_adj.jsonl}"
META_PATH="${META_PATH:-data/compact_meta.jsonl}"

TRAIN_STEPS="${TRAIN_STEPS:-10000}"
TOPK="${TOPK:-10}"
CANDIDATES="${CANDIDATES:-40}"
SAVE_ROOT="${SAVE_ROOT:-ckpt_emph}"
ALGO="${ALGO:-GRPO}"

# Heuristic thresholds for "available" GPU (tweak as needed)
MIN_FREE_MB="${MIN_FREE_MB:-30000}"     # require >= this free memory per GPU
MAX_UTIL="${MAX_UTIL:-20}"             # require <= this utilization (%)
POLL_SECS="${POLL_SECS:-20}"           # wait interval

ts() { date +"%Y%m%d_%H%M%S"; }
slug() { echo "$1" | sed 's#[/ ]#_#g'; }
JOB_ID="${JOB_ID:-local_$(ts)_pid$$}"

# =======================
#  MODEL GRID
# =======================
MAIN_LLMS=(
  "Qwen/Qwen3-14B"
  "google/gemma-3-27b-it"
  "Qwen/Qwen3-32B"
  "meta-llama/Meta-Llama-3-70B-Instruct"
)

ACTOR_TYPE="lm"

ACTOR_MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
)

# =======================
#  GPU HEURISTIC
# =======================
gpus_needed() {
  local main="$1"
  if [[ "$main" == *"70B"* || "$main" == *"70b"* ]]; then
    echo 4
  elif [[ "$main" == *"32B"* || "$main" == *"32b"* || "$main" == *"27B"* || "$main" == *"27b"* ]]; then
    echo 2
  elif [[ "$main" == *"14B"* || "$main" == *"14b"*]]; then
    echo 1
  else
    echo 1
  fi
}

# =======================
#  GPU PICKING (local)
# =======================
# Prints candidate GPUs, sorted best-first: "idx freeMB util"
gpu_table() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
  | awk -F',' '{gsub(/ /,""); print $1" "$2" "$3}' \
  | sort -k2,2nr -k3,3n
}

# Pick N GPUs meeting thresholds. Echoes "0,1,2" (comma-separated) or empty if not enough.
pick_gpus() {
  local n="$1"
  local min_free="$2"
  local max_util="$3"

  gpu_table \
    | awk -v mf="$min_free" -v mu="$max_util" '$2 >= mf && $3 <= mu {print $1}' \
    | head -n "$n" \
    | paste -sd, -
}

# Wait until we can pick N GPUs.
wait_for_gpus() {
  local n="$1"
  while true; do
    local picked
    picked="$(pick_gpus "$n" "$MIN_FREE_MB" "$MAX_UTIL" || true)"
    if [[ -n "${picked}" ]]; then
      # Ensure we actually got N GPUs (paste can return fewer)
      local cnt
      cnt="$(echo "$picked" | awk -F',' '{print NF}')"
      if [[ "$cnt" -ge "$n" ]]; then
        echo "$picked"
        return 0
      fi
    fi
    echo "[INFO] Need $n GPU(s) with free>=${MIN_FREE_MB}MB util<=${MAX_UTIL}%. Waiting ${POLL_SECS}s..."
    sleep "$POLL_SECS"
  done
}

# =======================
#  RUN ONE (sequential)
# =======================
run_one() {
  local MAIN_LLM="$1"
  local ACTOR_TYPE="$2"
  local ACTOR_MODEL="$3"
  local GPUS="$4"

  local run_name="beauty/$(slug "$MAIN_LLM")/${ALGO}/actor-$(slug "$ACTOR_MODEL")"
  local save_dir="${SAVE_ROOT}/${run_name}"
  mkdir -p "$save_dir"

  local log_base="logs/rec/$(slug "$run_name")-gpus${GPUS//,/}-$(slug "$JOB_ID")"
  local out_log="${log_base}.out"
  local err_log="${log_base}.err"

  echo "[INFO] =============================================="
  echo "[INFO] MAIN_LLM=$MAIN_LLM"
  echo "[INFO] ACTOR_TYPE=$ACTOR_TYPE"
  echo "[INFO] ACTOR_MODEL=$ACTOR_MODEL"
  echo "[INFO] CUDA_VISIBLE_DEVICES=$GPUS"
  echo "[INFO] save_dir=$save_dir"
  echo "[INFO] stdout=$out_log"
  echo "[INFO] stderr=$err_log"

  # Key line: restrict this run to the selected GPUs
  export CUDA_VISIBLE_DEVICES="$GPUS"

  # If your python still expects a "logical index", it should be 0 now
  # (because within CUDA_VISIBLE_DEVICES, the first GPU becomes cuda:0).
  local GPU_INDEX=0

  stdbuf -oL -eL python -u -m rec.train_emphasis \
    --llm-inputs "${LLM_INPUTS}" \
    --labels "${TRAIN_LABELS}" \
    --adj "${ADJ_PATH}" \
    --train-steps "${TRAIN_STEPS}" \
    --topk "${TOPK}" \
    --candidates "${CANDIDATES}" \
    --meta "${META_PATH}" \
    --model-id "${MAIN_LLM}" \
    --save_dir "${save_dir}" \
    --algorithm "${ALGO}" \
    --actor "${ACTOR_TYPE}" \
    --actor-model-id "${ACTOR_MODEL}" \
    --gpu "${GPU_INDEX}" \
    1> >(tee -a "$out_log") \
    2> >(tee -a "$err_log" >&2)

  echo "[INFO] Done: $run_name"
}

# =======================
#  MAIN LOOP (sequential)
# =======================
echo "[INFO] JOB_ID=$JOB_ID"
nvidia-smi || true

for MAIN_LLM in "${MAIN_LLMS[@]}"; do
  GPUS_N="$(gpus_needed "$MAIN_LLM")"

  # sanity: if machine has fewer GPUs than needed, skip early
  TOTAL_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
  if [[ "$GPUS_N" -gt "$TOTAL_GPUS" ]]; then
    echo "[WARN] Skip $MAIN_LLM: needs $GPUS_N GPU(s) but only $TOTAL_GPUS present."
    continue
  fi

  for ACTOR_MODEL in "${ACTOR_MODELS[@]}"; do
    echo "[INFO] Waiting GPUs for MAIN_LLM=$(slug "$MAIN_LLM") (need=$GPUS_N)..."
    GPUS="$(wait_for_gpus "$GPUS_N")"
    run_one "$MAIN_LLM" "$ACTOR_TYPE" "$ACTOR_MODEL" "$GPUS"
  done
done
