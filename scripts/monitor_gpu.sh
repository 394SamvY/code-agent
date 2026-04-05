#!/bin/bash
# GPU 显存/利用率实时采样器
#
# 用法:
#   bash scripts/monitor_gpu.sh                    # 默认每 5 秒采样
#   bash scripts/monitor_gpu.sh 2                  # 每 2 秒采样
#   bash scripts/monitor_gpu.sh 5 logs/gpu.csv     # 指定输出文件
#
# 输出:
#   CSV 格式日志，可用 pandas 分析：
#     timestamp, gpu_id, gpu_util%, mem_used_mb, mem_total_mb, mem_util%, temp_c, power_w
#
# 训练结束后分析:
#   python3 -c "
#   import pandas as pd
#   df = pd.read_csv('outputs/verl_grpo/gpu_monitor.csv')
#   print(df.groupby('gpu_id')['mem_used_mb'].describe())
#   print(df.groupby('gpu_id')['gpu_util%'].describe())
#   "

INTERVAL=${1:-5}
OUTPUT=${2:-outputs/verl_grpo/gpu_monitor.csv}

mkdir -p "$(dirname "$OUTPUT")"

echo "timestamp,gpu_id,gpu_name,gpu_util%,mem_used_mb,mem_total_mb,mem_util%,temp_c,power_w" > "$OUTPUT"

echo "[GPU Monitor] Sampling every ${INTERVAL}s → $OUTPUT"
echo "[GPU Monitor] Press Ctrl+C to stop"

while true; do
    TS=$(date '+%Y-%m-%d %H:%M:%S')
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,utilization.memory,temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r idx name util mem_used mem_total mem_util temp power; do
        echo "${TS},${idx},${name},${util},${mem_used},${mem_total},${mem_util},${temp},${power}" >> "$OUTPUT"
    done
    sleep "$INTERVAL"
done
