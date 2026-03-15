#!/bin/bash
# 评测脚本
#
# 用法:
#   bash scripts/evaluate.sh                         # multi_turn 评测
#   bash scripts/evaluate.sh one_shot                # one_shot 评测
#
# multi_turn 模式需要先启动 SGLang server:
#   python3 -m sglang.launch_server \
#       --model-path $MODEL \
#       --tool-call-parser qwen \
#       --port 30000 --host 0.0.0.0

set -euo pipefail

cd /root/autodl-tmp/code-agent

MODEL=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
DATA_DIR=/root/autodl-tmp/datasets
MODE=${1:-multi_turn}

echo "==== 评测 (mode=$MODE) ===="
python -m src.eval.evaluate \
    --model "$MODEL" \
    --datasets mbpp_test humaneval \
    --output_dir "./outputs/eval_${MODE}" \
    --mode "$MODE" \
    --sglang_url http://localhost:30000/v1 \
    --temperature 0.7 \
    --max_turns 7 \
    --data_dir "$DATA_DIR"
