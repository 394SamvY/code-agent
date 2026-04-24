#!/bin/bash
# Prepare and launch Agentic GRPO training with verl
#
# 用法:
#   bash scripts/prepare_verl_training.sh                 # 默认 2 GPU + Qwen3-8B 全量微调
#   bash scripts/prepare_verl_training.sh 4               # 指定 GPU 数量
#   bash scripts/prepare_verl_training.sh 2 grpo_qwen_7b  # 指定配置名
#
# 前置步骤:
#   1. pip install verl sglang pandas pyarrow
#   2. python -m src.data.verl_dataset      # 准备 parquet 数据

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

NUM_GPUS=${1:-2}
CONFIG_PATH="$PROJECT_DIR/configs/verl"
CONFIG_NAME="${2:-grpo_qwen3_8b}"

# ─── 环境修正 ──────────────────────────────────────────────────────
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
[ "$OMP_NUM_THREADS" -lt 1 ] 2>/dev/null && export OMP_NUM_THREADS=4

# PyTorch 2.9.1 Inductor bug: "duplicate template name" in TritonTemplate
export TORCHDYNAMO_DISABLE=1

# ─── 日志与诊断 ───────────────────────────────────────────────────
# verl FSDP workers 显存日志: fsdp_workers.py 里 24+ 个 log_gpu_memory_usage() 调用
# 默认 WARN 级别看不到，设 DEBUG 后会输出每个阶段的显存变化
export VERL_LOGGING_LEVEL=DEBUG

# Ray worker 日志级别
export RAY_DEDUP_LOGS=0           # 不去重 Ray 日志，看到每个 worker 的完整输出

# NCCL 通信诊断（可选，打开后输出非常多，默认关闭）
# export NCCL_DEBUG=INFO

# PyTorch 显存分配策略: 使用 expandable_segments 减少碎片
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== Agentic GRPO Training ==="
echo "  GPUs:    $NUM_GPUS"
echo "  Config:  $CONFIG_PATH/$CONFIG_NAME.yaml"
echo "  Project: $PROJECT_DIR"
echo "  Log lvl: VERL_LOGGING_LEVEL=$VERL_LOGGING_LEVEL"
echo ""

# Step 1: 准备数据（如果尚未准备）
if [ ! -f "$PROJECT_DIR/data/verl/train.parquet" ]; then
    echo "Preparing verl datasets..."
    python -m src.data.verl_dataset --data_dir /root/autodl-tmp/datasets
    echo ""
fi

# Step 2: 从 TOOLS_SCHEMA 生成 tool_config.yaml（保证训练用的 schema 和代码一致）
echo "Generating tool_config.yaml from TOOLS_SCHEMA..."
python3 -m src.env.tools
echo ""

# Step 3: 启动训练
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# 用 verl_main_wrapper.py 而不是直接 `python -m verl.trainer.main_ppo`，是为了在
# verl 启动前 patch 标准库 json，避免 numpy 指标写 JSON 时崩溃。
# 下面这些参数会保留在 sys.argv 里，由 verl 的 Hydra main 读取：
# --config-path / --config-name 是 Hydra 配置入口，trainer.n_gpus_per_node 是覆盖项。
python3 "$PROJECT_DIR/scripts/verl_main_wrapper.py" \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    trainer.n_gpus_per_node=$NUM_GPUS

echo ""
echo "=== Training complete ==="
echo "Checkpoints: ./outputs/verl_grpo/checkpoints/"
echo "Rollout data: ./outputs/verl_grpo/rollout_data/"
echo "TensorBoard: tensorboard --logdir=./outputs/verl_grpo/"
