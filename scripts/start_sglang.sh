#!/bin/bash
# 启动 SGLang 推理服务
#
# 用法:
#   bash scripts/start_sglang.sh                              # 默认启动基座模型
#   bash scripts/start_sglang.sh 30000 /path/to/sft_model     # 指定端口和模型
#
# 评估 SFT 模型前，需要先用 convert_checkpoint.py --mode merge 合并出完整模型

set -euo pipefail

BASE_MODEL=/root/autodl-tmp/models/Qwen3-8B
PORT=${1:-30000}
MODEL=${2:-$BASE_MODEL}
MEM_FRACTION=0.8

echo "==== 启动 SGLang (port=$PORT, mem=$MEM_FRACTION) ===="
echo "模型: $MODEL"
echo "等待 'The server is fired up and ready to roll!' ..."

python -m sglang.launch_server \
    --model "$MODEL" \
    --port "$PORT" \
    --mem-fraction-static "$MEM_FRACTION" \
    --dtype bfloat16
