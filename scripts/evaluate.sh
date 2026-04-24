#!/bin/bash
# 评测脚本（统一使用 SGLang 后端）
#
# 用法:
#   bash scripts/evaluate.sh                                    # 默认：LiveCodeBench multi_turn 评测
#   bash scripts/evaluate.sh one_shot                           # one_shot 评测
#   bash scripts/evaluate.sh multi_turn                         # multi_turn 评测
#   bash scripts/evaluate.sh one_shot --thinking                # 开启 thinking mode
#   bash scripts/evaluate.sh multi_turn "livecodebench" /path/model # 自定义数据集和模型
#
# 使用前需先启动 SGLang server:
#   bash scripts/start_sglang.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ---- 配置 ----
BASE_MODEL=/root/autodl-tmp/models/Qwen3-8B
DATA_DIR=/root/autodl-tmp/datasets
SGLANG_URL=http://localhost:30000/v1

# ---- 解析参数 ----
MODE=${1:-multi_turn}
THINKING=false
POSITIONAL=()

shift || true
for arg in "$@"; do
    if [ "$arg" = "--thinking" ]; then
        THINKING=true
    else
        POSITIONAL+=("$arg")
    fi
done

DATASETS=${POSITIONAL[0]:-"livecodebench"}
MODEL=${POSITIONAL[1]:-$BASE_MODEL}
MAX_TOOL_CALLS=${MAX_TOOL_CALLS:-32}
MAX_SUBMISSIONS=${MAX_SUBMISSIONS:-5}

if [ "$THINKING" = "true" ]; then
    MAX_TOKENS=4096
    THINKING_FLAG="--enable_thinking"
    OUTPUT_SUFFIX="${MODE}_thinking"
else
    MAX_TOKENS=1024
    THINKING_FLAG=""
    OUTPUT_SUFFIX="${MODE}"
fi

if [ "$THINKING" = "true" ]; then
    TEMPERATURE=0.6
elif [ "$MODE" = "one_shot" ]; then
    TEMPERATURE=0.0
else
    TEMPERATURE=0.7
fi

echo "==== 评测 (mode=$MODE, thinking=$THINKING, model=$MODEL, datasets=$DATASETS) ===="
python -m src.eval.evaluate \
    --model "$MODEL" \
    --datasets $DATASETS \
    --output_dir "./outputs/eval_${OUTPUT_SUFFIX}" \
    --mode "$MODE" \
    --sglang_url "$SGLANG_URL" \
    --temperature $TEMPERATURE \
    --max_tool_calls $MAX_TOOL_CALLS \
    --max_submissions $MAX_SUBMISSIONS \
    --max_new_tokens $MAX_TOKENS \
    --data_dir "$DATA_DIR" \
    $THINKING_FLAG
