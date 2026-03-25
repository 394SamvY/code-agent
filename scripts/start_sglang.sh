#!/bin/bash
# 启动 SGLang 推理服务
#
# 用法:
#   bash scripts/start_sglang.sh          # 默认启动
#   bash scripts/start_sglang.sh 30000    # 指定端口

set -euo pipefail

MODEL=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
PORT=${1:-30000}
MEM_FRACTION=0.8

echo "==== 启动 SGLang (port=$PORT, mem=$MEM_FRACTION) ===="
echo "模型: $MODEL"
echo "等待 'The server is fired up and ready to roll!' ..."

python -m sglang.launch_server \
    --model "$MODEL" \
    --port "$PORT" \
    --mem-fraction-static "$MEM_FRACTION" 
