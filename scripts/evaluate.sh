#!/bin/bash
# 评测脚本
#
# 用法:
#   bash scripts/evaluate.sh                                          # 默认：multi_turn 评测基座模型
#   bash scripts/evaluate.sh multi_turn mbpp_test /path/to/sft_model  # 评测 SFT 合并后模型
#   bash scripts/evaluate.sh one_shot                                 # one_shot 评测
#
# multi_turn 模式需要先启动 SGLang server（模型路径须与此处一致）:
#   bash scripts/start_sglang.sh [port] [model_path]

set -euo pipefail

cd /root/autodl-tmp/code-agent

# ---- 配置 ----
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
DATA_DIR=/root/autodl-tmp/datasets
SGLANG_URL=http://localhost:30000/v1

# ---- 参数 ----
MODE=${1:-multi_turn}
DATASETS=${2:-"mbpp_test humaneval"}
MODEL=${3:-$BASE_MODEL}
MAX_TURNS=8
TEMPERATURE=0.7
MAX_TOKENS=1024

echo "==== 评测 (mode=$MODE, model=$MODEL, datasets=$DATASETS) ===="
python -m src.eval.evaluate \
    --model "$MODEL" \
    --datasets $DATASETS \
    --output_dir "./outputs/eval_${MODE}" \
    --mode "$MODE" \
    --sglang_url "$SGLANG_URL" \
    --temperature $TEMPERATURE \
    --max_turns $MAX_TURNS \
    --max_new_tokens $MAX_TOKENS \
    --data_dir "$DATA_DIR"
