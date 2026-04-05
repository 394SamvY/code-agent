#!/bin/bash
# ============================================================================
# 一键启动训练 — 整合 GPU 监控 + DEBUG 日志 + 训练
# ============================================================================
#
# 用法:
#   bash scripts/start_training.sh              # 默认 2 GPU
#   bash scripts/start_training.sh 4            # 4 GPU
#
# 会自动:
#   1. 启动 GPU 监控（每 5 秒采样 → gpu_monitor.csv）
#   2. 记录训练前环境快照
#   3. 启动训练（所有日志保存到文件 + 终端同时输出）
#   4. 训练结束后自动停止 GPU 监控
#   5. 生成训练摘要报告
#
# 训练结束后你能得到:
#   outputs/verl_grpo/
#   ├── gpu_monitor.csv          # GPU 显存/利用率时间序列
#   ├── train.log                # 完整训练日志（含 DEBUG 显存变化）
#   ├── env_snapshot.txt         # 训练前环境信息
#   ├── rollout_data/            # 每步模型生成内容 (JSONL)
#   ├── checkpoints/             # FSDP checkpoints
#   └── tensorboard/             # TensorBoard 日志

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

NUM_GPUS=${1:-2}
OUTPUT_DIR="$PROJECT_DIR/outputs/verl_grpo"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$OUTPUT_DIR/train_${RUN_ID}.log"
ENV_SNAPSHOT="$OUTPUT_DIR/env_snapshot.txt"

mkdir -p "$OUTPUT_DIR"

# ─── 保存配置快照 ─────────────────────────────────────────────────
CONFIG_SNAPSHOT="$OUTPUT_DIR/config_${RUN_ID}.yaml"
cp "$PROJECT_DIR/configs/verl/grpo_qwen3_8b.yaml" "$CONFIG_SNAPSHOT"
echo "=== Config snapshot saved ==="
echo "  → $CONFIG_SNAPSHOT"

# ─── 记录环境快照 ─────────────────────────────────────────────────
echo "=== Recording environment snapshot ==="
{
    echo "═══════════════════════════════════════════════════"
    echo "  Training Environment Snapshot"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "═══════════════════════════════════════════════════"
    echo ""
    echo "── Hardware ──"
    nvidia-smi 2>/dev/null || echo "No GPU available (will fail at training)"
    echo ""
    echo "── Python ──"
    python3 --version
    echo ""
    echo "── Key packages ──"
    pip list 2>/dev/null | grep -iE "torch|transformers|verl|sglang|peft|accelerate|datasets|ray|trl" || true
    echo ""
    echo "── Model ──"
    ls -lh /root/autodl-tmp/models/Qwen3-8B/*.safetensors 2>/dev/null || echo "Model weights not found"
    echo ""
    echo "── Training data ──"
    for f in data/verl/*.parquet; do
        python3 -c "import pandas as pd; df=pd.read_parquet('$f'); print(f'  $f: {len(df)} records')" 2>/dev/null || true
    done
    echo ""
    echo "── Config ──"
    cat "configs/verl/grpo_qwen3_8b.yaml"
    echo ""
    echo "── Disk ──"
    df -h /root/autodl-tmp
    echo ""
    echo "── Memory ──"
    free -h
} > "$ENV_SNAPSHOT" 2>&1
echo "  → $ENV_SNAPSHOT"

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

    # 生成简要统计
    if [ -f "$OUTPUT_DIR/gpu_monitor.csv" ]; then
        echo ""
        echo "=== GPU Monitor Summary ==="
        python3 -c "
import pandas as pd
df = pd.read_csv('$OUTPUT_DIR/gpu_monitor.csv')
if len(df) > 0:
    print(f'  Total samples: {len(df)}')
    print(f'  Duration: {df[\"timestamp\"].iloc[0]} → {df[\"timestamp\"].iloc[-1]}')
    for gpu_id in df['gpu_id'].unique():
        g = df[df['gpu_id'] == gpu_id]
        print(f'  GPU {gpu_id}:')
        print(f'    Memory: avg={g[\"mem_used_mb\"].mean():.0f}MB, peak={g[\"mem_used_mb\"].max():.0f}MB / {g[\"mem_total_mb\"].iloc[0]:.0f}MB')
        print(f'    Utilization: avg={g[\"gpu_util%\"].mean():.1f}%, peak={g[\"gpu_util%\"].max():.1f}%')
else:
    print('  No data collected (possibly no GPU)')
" 2>/dev/null || echo "  (Could not parse GPU data)"
    fi

    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  训练产出文件:"
    echo "    日志:       $LOG_FILE"
    echo "    GPU 监控:   $OUTPUT_DIR/gpu_monitor.csv"
    echo "    环境快照:   $ENV_SNAPSHOT"
    echo "    Rollout:    $OUTPUT_DIR/rollout_data/"
    echo "    Checkpoint: $OUTPUT_DIR/checkpoints/"
    echo "═══════════════════════════════════════════════════"
    echo ""
    echo "查看 TensorBoard:"
    echo "  tensorboard --logdir=$OUTPUT_DIR --bind_all"
}
trap cleanup EXIT

# ─── 启动训练 ─────────────────────────────────────────────────────
echo ""
echo "=== Starting GRPO training ==="
echo "  Log: $LOG_FILE"
echo "  (tee: 同时输出到终端和文件)"
echo ""

bash "$PROJECT_DIR/scripts/train_verl.sh" "$NUM_GPUS" 2>&1 | tee "$LOG_FILE"
