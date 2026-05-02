#!/bin/bash
# 用 verl 的 multiturn SFT trainer 全量微调 OJ-like tool-call warm-start 模型。
#
# 默认输入：
#   data/verl/sft/sft_accepted_train.parquet
#   data/verl/sft/sft_accepted_val.parquet
#
# 用法：
#   bash scripts/train_sft_with_verl.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 模型位置
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

# 物理 GPU 索引（nvidia-smi 最左列的编号）。CUDA 驱动会把可见 GPU 从 0 开始重映射给应用。
# 单机：CUDA_VISIBLE_DEVICES=0,1 → 应用内 cuda:0=物理 GPU 0, cuda:1=物理 GPU 1
# 多机：每台机器各自设置，local_rank 只对应本机可见设备，不存在跨机器引用。
DEFAULT_CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-0,1}"

# 检测最终可用的 GPU 列表，优先级：用户设置 > 脚本默认 > nvidia-smi 探测 > 兜底 0
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

# 统计逗号分隔的设备数量
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

normalize_positive_int() {
    local value="${1:-}"
    local fallback="$2"
    if [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value"
    else
        echo "$fallback"
    fi
}

CUDA_VISIBLE_DEVICES_RESOLVED="$(detect_cuda_visible_devices)"
VISIBLE_GPU_COUNT="$(count_cuda_devices "$CUDA_VISIBLE_DEVICES_RESOLVED")"
if [ "$VISIBLE_GPU_COUNT" -lt 1 ]; then
    echo "[ERROR] No visible CUDA devices detected."
    exit 1
fi

# 实际使用的 GPU 数。默认吃满当前 CUDA_VISIBLE_DEVICES，smoke 时可设 NUM_GPUS=1。
NUM_GPUS="${NUM_GPUS:-$VISIBLE_GPU_COUNT}"

# Ulysses sequence parallel size。长 trajectory 全量微调时默认跨所有可见 GPU 切序列。
SP_SIZE="${SP_SIZE:-$NUM_GPUS}"

# 全局 batch size。435 条 SFT train 数据下，16 大约是 28 step/epoch。
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"

# 每张 GPU 的 micro batch。长序列全量微调默认 1，优先保稳定。
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"

# 单条 multi-turn trajectory 的最大 token 长度；当前数据 P90 约 18k，最长约 53k。
MAX_LENGTH="${MAX_LENGTH:-20480}"

# dynamic batch 每张 GPU 每个 micro batch 的 token 上限；默认等于 MAX_LENGTH，配合 SP 后实际可承载更长序列。
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-$MAX_LENGTH}"

# 超长样本处理策略。right 会截断极少数超过 MAX_LENGTH 的长 trajectory，避免默认长跑中途崩掉。
TRUNCATION="${TRUNCATION:-right}"

# 435 条小数据全量微调，2-3 epoch 让模型充分学习工具调用格式；低 LR 降低过拟合风险。
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"

# 设置后覆盖 epoch 推导的 step 数，主要用于 smoke test，例如 TOTAL_TRAINING_STEPS=1。
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"

# 全量微调学习率。比 LoRA 小一个量级，降低小 SFT 数据上灾难性漂移风险。
LR="${LR:-1e-5}"

# AdamW weight decay。
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

# warmup 比例；0.03 对几十 step 的小跑也只引入很短 warmup。
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.03}"

# 模型加载 dtype。bf16 显著降低 8B 全量微调显存占用，A800 原生支持。
FSDP_MODEL_DTYPE="${FSDP_MODEL_DTYPE:-bf16}"

# FSDP 训练混精 dtype。
FSDP_DTYPE="${FSDP_DTYPE:-bfloat16}"

# 是否启用梯度检查点；全量微调长序列默认开启以换显存。
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-true}"

# 是否把 optimizer state offload 到 CPU；默认关，OOM 时可设 true，速度会下降。
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-false}"

# 是否把参数 offload 到 CPU；比 optimizer offload 更慢，只作为 OOM 兜底。
PARAM_OFFLOAD="${PARAM_OFFLOAD:-false}"

NUM_WORKERS="${NUM_WORKERS:-4}"
PAD_MODE="${PAD_MODE:-no_padding}"
LOGGER="${LOGGER:-console}"
RESUME_MODE="${RESUME_MODE:-disable}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
IGNORE_INPUT_IDS_MISMATCH="${IGNORE_INPUT_IDS_MISMATCH:-true}"
SAVE_FREQ="${SAVE_FREQ:-after_each_epoch}"
# 每 N 步验证一次；27 步/epoch 下 5 步一验可及时发现过拟合。
TEST_FREQ="${TEST_FREQ:-5}"

# 只保存合并后的 HuggingFace 模型（bf16，8B 约 16GB），不存 FSDP 分片，不需要 resume。
# CHECKPOINT_LOAD_CONTENTS="${CHECKPOINT_LOAD_CONTENTS:-[model]}"
CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[hf_model]}"

# 多 epoch 时默认只保留最近一个 checkpoint，避免输出目录持续膨胀。
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-1}"

if [ "$NUM_GPUS" -gt "$VISIBLE_GPU_COUNT" ]; then
    echo "[ERROR] NUM_GPUS=$NUM_GPUS exceeds visible GPU count $VISIBLE_GPU_COUNT from CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_RESOLVED"
    exit 1
fi

if [ "$SP_SIZE" -lt 1 ] || [ $((NUM_GPUS % SP_SIZE)) -ne 0 ]; then
    echo "[ERROR] SP_SIZE=$SP_SIZE must be >=1 and divide NUM_GPUS=$NUM_GPUS"
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

# 限制 CUDA 可见设备，torchrun 子进程只能看到这里指定的 GPU
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_RESOLVED"
# 限制每个进程的 OpenMP 线程数，避免多 GPU 进程同时做 CPU 操作时抢占所有核
export OMP_NUM_THREADS="$(normalize_positive_int "${OMP_NUM_THREADS:-}" 4)"
# 允许 HuggingFace tokenizer 多进程并行，加速数据预处理
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
# 禁用 torch.compile（dynamo），FSDP + 长序列场景下 JIT 编译兼容性不稳定
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
# verl SFT trainer 日志级别：CRITICAL < ERROR < WARNING(verl默认) < INFO(脚本默认) < DEBUG
export VERL_SFT_LOGGING_LEVEL="${VERL_SFT_LOGGING_LEVEL:-INFO}"
# 把项目目录加入 Python 路径，让 torchrun 能 import 项目自定义模块（如 src/）
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

exec > >(tee "$LOG_FILE") 2>&1

echo "==== verl OJ-like multiturn SFT ===="
echo ""
echo "-- 路径 --"
echo "  project_dir:          $PROJECT_DIR"
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
echo "  max_token_len/gpu:    $MAX_TOKEN_LEN_PER_GPU"
echo "  truncation:           $TRUNCATION"
echo "  total_epochs:         $TOTAL_EPOCHS"
echo "  total_training_steps: ${TOTAL_TRAINING_STEPS:-<by epochs>}"
echo "  lr:                   $LR"
echo "  weight_decay:         $WEIGHT_DECAY"
echo "  warmup_ratio:         $LR_WARMUP_STEPS_RATIO"
echo "  finetune_mode:        full"
echo "  fsdp_model_dtype:     $FSDP_MODEL_DTYPE"
echo "  fsdp_dtype:           $FSDP_DTYPE"
echo "  grad_checkpointing:   $ENABLE_GRADIENT_CHECKPOINTING"
echo "  optimizer_offload:    $OPTIMIZER_OFFLOAD"
echo "  param_offload:        $PARAM_OFFLOAD"
echo "  checkpoint_contents:  $CHECKPOINT_SAVE_CONTENTS"
echo "  max_ckpt_to_keep:     $MAX_CKPT_TO_KEEP"
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
    data.custom_cls.path="$PROJECT_DIR/src/verl_sft_dataset_fix.py"
    data.custom_cls.name=FixedMultiTurnSFTDataset
    data.pad_mode="$PAD_MODE"
    data.max_length="$MAX_LENGTH"
    data.truncation="$TRUNCATION"
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU"
    data.max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU"
    data.num_workers="$NUM_WORKERS"
    data.ignore_input_ids_mismatch="$IGNORE_INPUT_IDS_MISMATCH"
    model.path="$MODEL_PATH"
    model.trust_remote_code="$TRUST_REMOTE_CODE"
    model.use_remove_padding="$USE_REMOVE_PADDING"
    model.enable_gradient_checkpointing="$ENABLE_GRADIENT_CHECKPOINTING"
    model.lora_rank=0
    model.lora_adapter_path=null
    optim.lr="$LR"
    optim.weight_decay="$WEIGHT_DECAY"
    optim.lr_warmup_steps_ratio="$LR_WARMUP_STEPS_RATIO"
    checkpoint.save_contents="$CHECKPOINT_SAVE_CONTENTS"
    trainer.default_local_dir="$SAVE_PATH"
    trainer.project_name=code-agent-sft
    trainer.experiment_name="$RUN_NAME"
    trainer.logger="$LOGGER"
    trainer.total_epochs="$TOTAL_EPOCHS"
    trainer.resume_mode="$RESUME_MODE"
    trainer.save_freq="$SAVE_FREQ"
    trainer.test_freq="$TEST_FREQ"
    trainer.max_ckpt_to_keep="$MAX_CKPT_TO_KEEP"
    engine.model_dtype="$FSDP_MODEL_DTYPE"
    engine.dtype="$FSDP_DTYPE"
    engine.optimizer_offload="$OPTIMIZER_OFFLOAD"
    engine.param_offload="$PARAM_OFFLOAD"
    engine.ulysses_sequence_parallel_size="$SP_SIZE"
)

if [ -n "$TOTAL_TRAINING_STEPS" ]; then
    cmd+=(trainer.total_training_steps="$TOTAL_TRAINING_STEPS")
fi

cmd+=("$@")

cd "$PROJECT_DIR"
"${cmd[@]}"

echo ""
echo "end:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "==== SFT complete ===="
echo "Checkpoint dir: $SAVE_PATH"
echo "Log file:       $LOG_FILE"
