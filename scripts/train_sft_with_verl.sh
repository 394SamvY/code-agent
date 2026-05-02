#!/bin/bash
# 用 verl 的 multiturn SFT trainer 训练 OJ-like tool-call warm-start 模型。
#
# 默认输入：
#   data/verl/sft/sft_accepted_train.parquet
#   data/verl/sft/sft_accepted_val.parquet
#
# 用法：
#   bash scripts/train_sft_with_verl.sh
#   MODEL_PATH=/root/autodl-tmp/models/Qwen3-8B CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_sft_with_verl.sh
#   bash scripts/train_sft_with_verl.sh /root/autodl-tmp/models/Qwen3-8B /root/autodl-tmp/checkpoints/code-agent-sft
#
# 常用覆盖：
#   TRAIN_BATCH_SIZE=16 MICRO_BATCH_SIZE_PER_GPU=1 MAX_LENGTH=32768 bash scripts/train_sft_with_verl.sh
#   TOTAL_TRAINING_STEPS=20 bash scripts/train_sft_with_verl.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERL_DIR="${VERL_DIR:-/Users/yang/code/verl}"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-8B}"
SAVE_PATH="${SAVE_PATH:-}"

if [ "${1:-}" != "" ]; then
    MODEL_PATH="$1"
    shift
fi

if [ "${1:-}" != "" ]; then
    SAVE_PATH="$1"
    shift
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [ -z "$SAVE_PATH" ]; then
    SAVE_PATH="$PROJECT_DIR/outputs/verl_sft/qwen3_8b_oj_sft_$TIMESTAMP"
fi

TRAIN_FILE="${TRAIN_FILE:-$PROJECT_DIR/data/verl/sft/sft_accepted_train.parquet}"
VAL_FILE="${VAL_FILE:-$PROJECT_DIR/data/verl/sft/sft_accepted_val.parquet}"
RUN_NAME="${RUN_NAME:-$(basename "$SAVE_PATH")}"
LOG_FILE="$SAVE_PATH/train.log"

DEFAULT_CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-0,1}"

detect_cuda_visible_devices() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "$CUDA_VISIBLE_DEVICES"
        return
    fi
    if [ -n "$DEFAULT_CUDA_VISIBLE_DEVICES" ]; then
        echo "$DEFAULT_CUDA_VISIBLE_DEVICES"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local devices
        devices="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
        if [ -n "$devices" ]; then
            echo "$devices"
            return
        fi
    fi
    echo "0"
}

count_cuda_devices() {
    local devices="$1"
    if [ -z "$devices" ] || [ "$devices" = "-1" ]; then
        echo "0"
        return
    fi
    local compact="${devices// /}"
    IFS=',' read -ra parts <<< "$compact"
    local count=0
    for part in "${parts[@]}"; do
        if [ -n "$part" ]; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

CUDA_VISIBLE_DEVICES_RESOLVED="$(detect_cuda_visible_devices)"
VISIBLE_GPU_COUNT="$(count_cuda_devices "$CUDA_VISIBLE_DEVICES_RESOLVED")"
if [ "$VISIBLE_GPU_COUNT" -lt 1 ]; then
    echo "[ERROR] No visible CUDA devices detected."
    exit 1
fi

NUM_GPUS="${NUM_GPUS:-$VISIBLE_GPU_COUNT}"
SP_SIZE="${SP_SIZE:-$NUM_GPUS}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
TRUNCATION="${TRUNCATION:-error}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PAD_MODE="${PAD_MODE:-no_padding}"
LOGGER="${LOGGER:-console}"
RESUME_MODE="${RESUME_MODE:-disable}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
IGNORE_INPUT_IDS_MISMATCH="${IGNORE_INPUT_IDS_MISMATCH:-true}"

if [ "$NUM_GPUS" -gt "$VISIBLE_GPU_COUNT" ]; then
    echo "[ERROR] NUM_GPUS=$NUM_GPUS exceeds visible GPU count $VISIBLE_GPU_COUNT from CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_RESOLVED"
    exit 1
fi

if [ "$SP_SIZE" -lt 1 ] || [ $((NUM_GPUS % SP_SIZE)) -ne 0 ]; then
    echo "[ERROR] SP_SIZE=$SP_SIZE must be >=1 and divide NUM_GPUS=$NUM_GPUS"
    exit 1
fi

if [ ! -d "$VERL_DIR" ]; then
    echo "[ERROR] VERL_DIR not found: $VERL_DIR"
    exit 1
fi

if [ ! -e "$MODEL_PATH" ]; then
    echo "[ERROR] MODEL_PATH not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "[ERROR] TRAIN_FILE not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$VAL_FILE" ]; then
    echo "[ERROR] VAL_FILE not found: $VAL_FILE"
    exit 1
fi

mkdir -p "$SAVE_PATH"
if [ -e "$LOG_FILE" ]; then
    echo "[ERROR] Log file already exists and will not be overwritten: $LOG_FILE"
    echo "Use a different SAVE_PATH or RUN_NAME."
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_RESOLVED"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export VERL_SFT_LOGGING_LEVEL="${VERL_SFT_LOGGING_LEVEL:-INFO}"
export PYTHONPATH="$PROJECT_DIR:$VERL_DIR${PYTHONPATH:+:$PYTHONPATH}"

exec > >(tee "$LOG_FILE") 2>&1

echo "==== verl OJ-like multiturn SFT ===="
echo ""
echo "-- 路径 --"
echo "  project_dir:          $PROJECT_DIR"
echo "  verl_dir:             $VERL_DIR"
echo "  train_file:           $TRAIN_FILE"
echo "  val_file:             $VAL_FILE"
echo "  model_path:           $MODEL_PATH"
echo "  save_path:            $SAVE_PATH"
echo ""
echo "-- 训练参数 --"
echo "  cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
echo "  num_gpus:             $NUM_GPUS"
echo "  sp_size:              $SP_SIZE"
echo "  train_batch_size:     $TRAIN_BATCH_SIZE"
echo "  micro_batch_per_gpu:  $MICRO_BATCH_SIZE_PER_GPU"
echo "  max_length:           $MAX_LENGTH"
echo "  truncation:           $TRUNCATION"
echo "  total_epochs:         $TOTAL_EPOCHS"
echo "  total_training_steps: ${TOTAL_TRAINING_STEPS:-<by epochs>}"
echo "  lr:                   $LR"
echo "  lora_rank:            $LORA_RANK"
echo "  lora_alpha:           $LORA_ALPHA"
echo ""
echo "start: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

cmd=(
    torchrun
    --nnodes=1
    --nproc_per_node="$NUM_GPUS"
    -m verl.trainer.sft_trainer
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.messages_key=messages
    data.tools_key=tools
    data.enable_thinking_key=enable_thinking
    data.enable_thinking_default=false
    data.pad_mode="$PAD_MODE"
    data.max_length="$MAX_LENGTH"
    data.truncation="$TRUNCATION"
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU"
    data.num_workers="$NUM_WORKERS"
    data.ignore_input_ids_mismatch="$IGNORE_INPUT_IDS_MISMATCH"
    model.path="$MODEL_PATH"
    model.trust_remote_code="$TRUST_REMOTE_CODE"
    model.use_remove_padding="$USE_REMOVE_PADDING"
    model.lora_rank="$LORA_RANK"
    model.lora_alpha="$LORA_ALPHA"
    optim.lr="$LR"
    trainer.default_local_dir="$SAVE_PATH"
    trainer.project_name=code-agent-sft
    trainer.experiment_name="$RUN_NAME"
    trainer.logger="$LOGGER"
    trainer.total_epochs="$TOTAL_EPOCHS"
    trainer.resume_mode="$RESUME_MODE"
    engine.ulysses_sequence_parallel_size="$SP_SIZE"
)

if [ -n "$TOTAL_TRAINING_STEPS" ]; then
    cmd+=(trainer.total_training_steps="$TOTAL_TRAINING_STEPS")
fi

cmd+=("$@")

cd "$VERL_DIR"
"${cmd[@]}"

echo ""
echo "end:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "==== SFT complete ===="
echo "Checkpoint dir: $SAVE_PATH"
echo "Log file:       $LOG_FILE"
