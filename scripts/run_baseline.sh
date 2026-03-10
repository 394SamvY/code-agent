#!/bin/bash
# 裸模型 baseline 评估
# 用法: bash scripts/run_baseline.sh [model_name]

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:-"Qwen/Qwen2.5-Coder-1.5B-Instruct"}
OUTPUT_DIR="./outputs/baseline"

python -m src.eval.evaluate \
    --model "$MODEL" \
    --datasets mbpp_test humaneval \
    --output_dir "$OUTPUT_DIR" \
    --mode baseline \
    --temperature 0.0 \
    --max_new_tokens 512 \
    --data_dir ~/yangenhui/datasets
