#!/bin/bash
# 用 verl validation 跑 OJ-like 评测，复用和训练一致的 tool / rollout / reward 链路。
#
# 用法:
#   bash scripts/evaluate_with_verl.sh
#   bash scripts/evaluate_with_verl.sh livecodebench_test
#   bash scripts/evaluate_with_verl.sh codecontests_test /path/to/model 1

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

DATASET_NAME="${1:-livecodebench_test}"
MODEL_PATH="${2:-/root/autodl-tmp/models/Qwen3-8B}"
NUM_GPUS="${3:-1}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/verl_eval}"
CONFIG_PATH="$PROJECT_DIR/configs/verl"
CONFIG_NAME="${CONFIG_NAME:-grpo_qwen3_8b}"

# 评测走和训练一致的 verl main_ppo validation 链路，而不是 `verl.trainer.main_eval`。
# 后者只是对已生成 responses 的离线 reward 打分，不会驱动在线 tool loop。
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

case "$DATASET_NAME" in
    codecontests_valid)
        VAL_FILE="$PROJECT_DIR/data/verl/codecontests_valid.parquet"
        ;;
    codecontests_test)
        VAL_FILE="$PROJECT_DIR/data/verl/codecontests_test.parquet"
        ;;
    livecodebench_test)
        VAL_FILE="$PROJECT_DIR/data/verl/livecodebench_test.parquet"
        ;;
    *)
        echo "[ERROR] Unknown dataset alias: $DATASET_NAME"
        echo "Supported: codecontests_valid, codecontests_test, livecodebench_test"
        exit 1
        ;;
esac

REQUIRED_PARQUETS=(
    "$PROJECT_DIR/data/verl/codecontests_train.parquet"
    "$PROJECT_DIR/data/verl/codecontests_valid.parquet"
    "$PROJECT_DIR/data/verl/codecontests_test.parquet"
    "$PROJECT_DIR/data/verl/livecodebench_test.parquet"
)

MISSING_PARQUET=0
for parquet_path in "${REQUIRED_PARQUETS[@]}"; do
    if [ ! -f "$parquet_path" ]; then
        MISSING_PARQUET=1
        break
    fi
done

if [ "$MISSING_PARQUET" -eq 1 ]; then
    echo "Preparing verl datasets..."
    python -m src.data.verl_dataset --data_dir "$DATA_DIR"
    echo ""
fi

echo "Generating tool_config.yaml from TOOLS_SCHEMA..."
python3 -m src.env.tools
echo ""

RUN_NAME="${DATASET_NAME}_$(basename "$MODEL_PATH")"
VALIDATION_DIR="$OUTPUT_ROOT/$RUN_NAME/generations"
CHECKPOINT_DIR="$OUTPUT_ROOT/$RUN_NAME/checkpoints"
mkdir -p "$VALIDATION_DIR" "$CHECKPOINT_DIR"

echo "==== verl eval ===="
echo "  dataset:   $DATASET_NAME"
echo "  val_file:  $VAL_FILE"
echo "  model:     $MODEL_PATH"
echo "  gpus:      $NUM_GPUS"
echo "  output:    $OUTPUT_ROOT/$RUN_NAME"
echo ""

python3 "$PROJECT_DIR/scripts/verl_main_wrapper.py" \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.project_name=code-agent-eval \
    trainer.experiment_name="$RUN_NAME" \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="$VALIDATION_DIR" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    data.val_files="$VAL_FILE" \
    actor_rollout_ref.model.path="$MODEL_PATH"
