#!/bin/bash
# Run one OJ-like baseline eval set through verl validation.
#
# This script intentionally uses verl's main_ppo validation path so the run goes
# through the same multi-turn agent loop, tool layer, and reward function as GRPO.
#
# Usage:
#   bash scripts/evaluate_baseline_with_verl.sh codecontests_test
#   bash scripts/evaluate_baseline_with_verl.sh livecodebench_test
#   bash scripts/evaluate_baseline_with_verl.sh /root/autodl-tmp/code-agent/data/verl/codecontests_test.parquet /root/autodl-tmp/models/Qwen3-8B
#
# Useful overrides:
#   MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=8192 bash scripts/evaluate_baseline_with_verl.sh codecontests_test
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/evaluate_baseline_with_verl.sh livecodebench_test
#   VAL_MAX_SAMPLES=8 bash scripts/evaluate_baseline_with_verl.sh codecontests_test

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

DATASET_ARG="${1:-codecontests_test}"
MODEL_PATH="${2:-/root/autodl-tmp/models/Qwen3-8B}"

CONFIG_PATH="$PROJECT_DIR/configs/verl"
CONFIG_NAME="${CONFIG_NAME:-grpo_qwen3_8b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/verl_baseline_eval}"
DEFAULT_CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-0,1}"

normalize_positive_int() {
    local value="${1:-}"
    local fallback="$2"
    if [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value"
    else
        echo "$fallback"
    fi
}

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

# Current baseline eval defaults target the 2xA800-80GB server.
# All of these remain overrideable from the shell, e.g. VAL_MAX_SAMPLES=32 ...
NUM_GPUS="${NUM_GPUS:-$VISIBLE_GPU_COUNT}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-28672}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-1000}"
TRUNCATION="${TRUNCATION:-middle}"
FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-true}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
AGENT_WORKERS="${AGENT_WORKERS:-16}"
FSDP_MODEL_DTYPE="${FSDP_MODEL_DTYPE:-bf16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-1}"
TRAIN_STUB_FILE="${TRAIN_STUB_FILE:-$PROJECT_DIR/data/verl/codecontests_valid.parquet}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
VAL_TOP_K="${VAL_TOP_K:-20}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-true}"

if [ "$NUM_GPUS" -gt "$VISIBLE_GPU_COUNT" ]; then
    echo "[ERROR] NUM_GPUS=$NUM_GPUS exceeds visible GPU count $VISIBLE_GPU_COUNT from CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_RESOLVED"
    exit 1
fi

if [ "$ROLLOUT_TP" -lt 1 ] || [ $((NUM_GPUS % ROLLOUT_TP)) -ne 0 ]; then
    echo "[ERROR] ROLLOUT_TP=$ROLLOUT_TP must be >=1 and divide NUM_GPUS=$NUM_GPUS"
    exit 1
fi

# Multi-GPU baseline defaults. Caller may override these before invoking.
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_RESOLVED"
export OMP_NUM_THREADS="$(normalize_positive_int "${OMP_NUM_THREADS:-}" 4)"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export RAY_DISABLE_DOCKER_CPU_WARNING="${RAY_DISABLE_DOCKER_CPU_WARNING:-1}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

case "$DATASET_ARG" in
    codecontests_valid)
        VAL_FILE="$PROJECT_DIR/data/verl/codecontests_valid.parquet"
        DATASET_TAG="codecontests_valid"
        ;;
    codecontests_test)
        VAL_FILE="$PROJECT_DIR/data/verl/codecontests_test.parquet"
        DATASET_TAG="codecontests_test"
        ;;
    livecodebench_test)
        VAL_FILE="$PROJECT_DIR/data/verl/livecodebench_test.parquet"
        DATASET_TAG="livecodebench_test"
        ;;
    *.parquet)
        if [ -f "$DATASET_ARG" ]; then
            VAL_FILE="$DATASET_ARG"
        elif [ -f "$PROJECT_DIR/data/verl/$DATASET_ARG" ]; then
            VAL_FILE="$PROJECT_DIR/data/verl/$DATASET_ARG"
        else
            VAL_FILE="$DATASET_ARG"
        fi
        DATASET_TAG="$(basename "$DATASET_ARG" .parquet)"
        ;;
    *)
        echo "[ERROR] Unknown dataset alias or parquet path: $DATASET_ARG"
        echo "Supported aliases: codecontests_valid, codecontests_test, livecodebench_test"
        exit 1
        ;;
esac

if [ ! -f "$VAL_FILE" ]; then
    echo "[ERROR] Eval parquet not found: $VAL_FILE"
    exit 1
fi

if [ ! -f "$TRAIN_STUB_FILE" ]; then
    echo "[ERROR] Train stub parquet not found: $TRAIN_STUB_FILE"
    echo "Set TRAIN_STUB_FILE to any small verl parquet if needed."
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model path not found: $MODEL_PATH"
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-${DATASET_TAG}_$(basename "$MODEL_PATH")_mp${MAX_PROMPT_LENGTH}_mr${MAX_RESPONSE_LENGTH}_${TIMESTAMP}}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
VALIDATION_DIR="$RUN_DIR/generations"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
LOG_FILE="$RUN_DIR/verl_eval.log"
if [ -e "$LOG_FILE" ]; then
    echo "[ERROR] Log file already exists and will not be overwritten: $LOG_FILE"
    echo "Use a different RUN_NAME or remove the existing run directory."
    exit 1
fi
mkdir -p "$VALIDATION_DIR" "$CHECKPOINT_DIR"

# Save the full console stream next to generations/checkpoints for reproducibility.
exec > >(tee "$LOG_FILE") 2>&1

if [ "${GENERATE_TOOL_CONFIG:-1}" = "1" ]; then
    echo "Generating tool_config.yaml from TOOLS_SCHEMA..."
    python3 -m src.env.tools
    echo ""
fi

echo "==== verl OJ-like baseline eval ===="
echo ""
echo "-- 运行环境 --"
echo "  dataset:              $DATASET_TAG"
echo "  model:                $MODEL_PATH"
echo "  cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
echo "  num_gpus:             $NUM_GPUS  (rollout tensor_parallel=$ROLLOUT_TP)"
echo "  output:               $RUN_DIR"

echo ""
echo "-- 评测策略（常用覆盖）--"
echo "  val_max_samples:      $VAL_MAX_SAMPLES  (跑多少条，设小值做 smoke test)"
echo "  max_prompt_length:    $MAX_PROMPT_LENGTH  (首轮 prompt 最大 token 数，超出样本被过滤)"
echo "  max_response_length:  $MAX_RESPONSE_LENGTH  (整条 trajectory 总 token 预算，含所有轮次)"
echo "  max_model_len:        $MAX_MODEL_LEN  (prompt+response 总容量，自动=mp+Mr)"
echo "  val_temperature:      $VAL_TEMPERATURE  (采样温度，0=贪心)"
echo "  val_top_p:            $VAL_TOP_P  (nucleus sampling)"
echo "  val_top_k:            $VAL_TOP_K  (top-k sampling)"
echo "  val_do_sample:        $VAL_DO_SAMPLE  (true=采样, false=贪心解码)"

echo ""
echo "-- GPU 吞吐（偶尔调）--"
echo "  gpu_memory_util:      $GPU_MEMORY_UTILIZATION  (SGLang 预留显存比例，越高 KV-cache 越大)"
echo "  max_num_seqs:         $MAX_NUM_SEQS  (单 GPU 最大并发序列数)"
echo "  max_batched_tokens:   $MAX_NUM_BATCHED_TOKENS  (单次迭代最多打包多少 token，限制算力峰值)"
echo "  agent_workers:        $AGENT_WORKERS  (异步 agent loop worker 数)"
echo "  val_batch_size:       $VAL_BATCH_SIZE  (dataloader 每批取多少条)"

echo "-- 其他 --"
echo "  val_file:             $VAL_FILE"
echo "  fsdp_model_dtype:     $FSDP_MODEL_DTYPE"
echo "  dataloader_workers:   $DATALOADER_NUM_WORKERS"
echo "  run_name:             $RUN_NAME"

echo ""
echo "start: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python3 "$PROJECT_DIR/scripts/verl_main_wrapper.py" \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.project_name=code-agent-baseline-eval \
    trainer.experiment_name="$RUN_NAME" \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.logger='["console"]' \
    trainer.log_val_generations="$LOG_VAL_GENERATIONS" \
    trainer.rollout_data_dir="$RUN_DIR/rollout_data" \
    trainer.validation_data_dir="$VALIDATION_DIR" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYTHONPATH" \
    data.train_files="$TRAIN_STUB_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.val_max_samples="$VAL_MAX_SAMPLES" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.truncation="$TRUNCATION" \
    data.filter_overlong_prompts="$FILTER_OVERLONG_PROMPTS" \
    data.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
    data.custom_cls.path=src/verl_dataset_adapter.py \
    data.custom_cls.name=OJLikeRLHFDataset \
    data.tool_config_path=configs/verl/tool_config.yaml \
    data.return_raw_chat=true \
    data.apply_chat_template_kwargs.enable_thinking=true \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.model_dtype="$FSDP_MODEL_DTYPE" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.model_dtype="$FSDP_MODEL_DTYPE" \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.enforce_eager="$ENFORCE_EAGER" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
    actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P" \
    actor_rollout_ref.rollout.val_kwargs.top_k="$VAL_TOP_K" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="$VAL_DO_SAMPLE" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=configs/verl/tool_config.yaml \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable

echo ""
echo "end:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "==== eval complete ===="
echo "Validation generations: $VALIDATION_DIR/0.jsonl"
echo "Log file: $LOG_FILE"
