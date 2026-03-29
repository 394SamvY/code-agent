#!/bin/bash
# verl SFT 多轮工具调用训练
#
# 用法:
#   bash scripts/sft_train.sh 4                    # 4 卡（torchrun 模式）
#   bash scripts/sft_train.sh 4 --ray               # 4 卡（Ray 模式，支持多机扩展）
#
# 前置步骤:
#   1. pip install verl
#   2. 准备数据: python -m src.data.generate_sft_data generate --dataset mbpp_train --output data/sft/train.parquet
#
# 多机训练（Ray 模式）:
#   在各节点启动 Ray:
#     ray start --head                          # 主节点
#     ray start --address=<主节点IP>:6379       # 从节点
#   然后:
#     bash scripts/sft_train.sh 4 --ray trainer.nnodes=2
#
# 命令行可覆盖 YAML 中的任意参数（Hydra 语法）:
#   bash scripts/sft_train.sh 4 optim.lr=1e-5 trainer.total_epochs=5

# Bash 的安全模式，三个 flag：
#  - -e (errexit)：任何命令报错立即退出脚本，不会继续往下跑
#  - -u (nounset)：使用未定义的变量时报错，而不是当空字符串
#  - -o pipefail：管道命令中任何一步失败，整个管道就算失败

set -euo pipefail

# --- 解析参数 ---
NPROC=${1:-2}                               # 第一个参数：每个节点的 GPU 数，默认 2
shift 1 || true                             # 移除第一个参数，剩下的传给 verl

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# 解析 --ray 标志，其余参数透传给 verl（可覆盖 YAML 配置）
USE_RAY=false
EXTRA_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--ray" ]; then
        USE_RAY=true
    else
        EXTRA_ARGS+=("$arg")                # 收集额外参数，如 optim.lr=1e-5
    fi
done

# --- 检查数据 ---
if [ ! -f "data/sft/train.parquet" ]; then
    echo "[ERROR] data/sft/train.parquet not found."
    echo "Run: python -m src.data.generate_sft_data generate --dataset mbpp_train --output data/sft/train.parquet"
    exit 1
fi

# --- 配置路径 ---
# Hydra 通过 --config-path + --config-name 定位 YAML
# 实际加载: configs/sft/sft_qwen_7b.yaml
CONFIG_PATH="$PROJECT_DIR/configs/sft"
CONFIG_NAME="sft_qwen_7b"

echo "=== verl SFT Training ==="
echo "  GPUs per node: $NPROC"
echo "  Mode: $([ "$USE_RAY" = true ] && echo 'Ray' || echo 'torchrun')"
echo "  Config: $CONFIG_PATH/$CONFIG_NAME.yaml"
echo "  Extra args: ${EXTRA_ARGS[*]:-none}"
echo ""

if [ "$USE_RAY" = true ]; then
    # Ray 模式：支持多机，Ray 负责资源调度和进程管理
    # 需要先在各节点启动 ray start
    python -m verl.trainer.main_sft \
        --config-path="$CONFIG_PATH" \
        --config-name="$CONFIG_NAME" \
        trainer.n_gpus_per_node=$NPROC \
        "${EXTRA_ARGS[@]}"
else
    # torchrun 模式：单机多卡，PyTorch 原生启动器
    # --standalone: 单机模式，不需要 rdzv 服务
    # --nnodes=1: 单机
    # --nproc_per_node: 每机启动几个进程（= GPU 数）
    torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC \
        -m verl.trainer.sft_trainer \
        --config-path="$CONFIG_PATH" \
        --config-name="$CONFIG_NAME" \
        "${EXTRA_ARGS[@]}"
fi

echo ""
echo "=== SFT Training complete ==="
echo "Checkpoint saved to: outputs/sft_ckpt/"
