#!/bin/bash
# ============================================================================
# 一键启动训练 — 整合 GPU 监控 + DEBUG 日志 + 训练
# ============================================================================
#
# 用法:
#   bash scripts/start_training.sh                    # 默认 2 GPU + GSPO
#   bash scripts/start_training.sh 4                  # 4 GPU + GSPO
#   bash scripts/start_training.sh 2 grpo_qwen3_8b    # 指定配置名
#
# 会自动:
#   1. 启动 GPU 监控（每 5 秒采样 → gpu_monitor.csv）
#   2. 启动训练（所有日志保存到文件 + 终端同时输出）
#   3. 训练结束后自动停止 GPU 监控
#
# 训练结束后你能得到:
#   outputs/verl_gspo/
#   ├── gpu_monitor.csv          # GPU 显存/利用率时间序列
#   ├── train.log                # 完整训练日志（含 DEBUG 显存变化）
#   ├── rollout_data/            # 每步模型生成内容 (JSONL)
#   ├── checkpoints/             # FSDP checkpoints
#   └── tensorboard/             # TensorBoard 日志

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

NUM_GPUS=${1:-2}
CONFIG_NAME="${2:-gspo_qwen3_8b}"
OUTPUT_DIR="$PROJECT_DIR/outputs/verl_gspo"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$OUTPUT_DIR/train_${RUN_ID}.log"

mkdir -p "$OUTPUT_DIR"

# ─── 保存配置快照 ─────────────────────────────────────────────────
CONFIG_SNAPSHOT="$OUTPUT_DIR/config_${RUN_ID}.yaml"
cp "$PROJECT_DIR/configs/verl/${CONFIG_NAME}.yaml" "$CONFIG_SNAPSHOT"
echo "=== Config snapshot saved ==="
echo "  → $CONFIG_SNAPSHOT"

# ─── 启动 GPU 监控 ───────────────────────────────────────────────
echo ""
echo "=== Starting GPU monitor (every 5s) ==="
bash "$PROJECT_DIR/scripts/monitor_gpu.sh" 5 "$OUTPUT_DIR/gpu_monitor.csv" &
GPU_MONITOR_PID=$!
echo "  → PID: $GPU_MONITOR_PID → $OUTPUT_DIR/gpu_monitor.csv"

# 确保退出时杀掉 GPU 监控
cleanup() {
    echo ""
    echo "=== Stopping GPU monitor (PID: $GPU_MONITOR_PID) ==="
    kill $GPU_MONITOR_PID 2>/dev/null || true
    wait $GPU_MONITOR_PID 2>/dev/null || true
}
trap cleanup EXIT

# ─── 启动训练 ─────────────────────────────────────────────────────
echo ""
echo "=== Starting verl training ==="
echo "  Config: $CONFIG_NAME"
echo "  Log: $LOG_FILE"
echo "  (tee: 同时输出到终端和文件)"
echo ""

bash "$PROJECT_DIR/scripts/prepare_verl_training.sh" "$NUM_GPUS" "$CONFIG_NAME" 2>&1 | tee "$LOG_FILE"
