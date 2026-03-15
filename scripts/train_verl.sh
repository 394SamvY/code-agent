#!/bin/bash
# Agentic GRPO Training with verl
#
# 用法:
#   bash scripts/train_verl.sh              # 默认配置
#   bash scripts/train_verl.sh 4            # 指定 GPU 数量
#
# 前置步骤:
#   1. pip install verl sglang pandas pyarrow
#   2. python -m src.data.verl_dataset      # 准备 parquet 数据

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

NUM_GPUS=${1:-2}
CONFIG_PATH="$PROJECT_DIR/configs/verl"
CONFIG_NAME="grpo_qwen_7b"

echo "=== Agentic GRPO Training ==="
echo "  GPUs: $NUM_GPUS"
echo "  Config: $CONFIG_PATH/$CONFIG_NAME.yaml"
echo "  Project dir: $PROJECT_DIR"
echo ""

# Step 1: 准备数据（如果尚未准备）
if [ ! -f "$PROJECT_DIR/data/verl/train.parquet" ]; then
    echo "Preparing verl datasets..."
    python -m src.data.verl_dataset
    echo ""
fi

# Step 2: 从 TOOLS_SCHEMA 生成 tool_config.yaml（保证训练用的 schema 和代码一致）
echo "Generating tool_config.yaml from TOOLS_SCHEMA..."
python3 -m src.env.tools
echo ""

# Step 3: 启动训练
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.tool_kwargs.tools_config_file="$PROJECT_DIR/configs/verl/tool_config.yaml" \
    trainer.total_epochs=10

echo ""
echo "=== Training complete ==="
echo "Checkpoints saved to: ./outputs/verl_grpo/"
