#!/bin/bash
# 诊断评测：全量 few-shot multi-turn
# 用法: tmux new -s eval && bash scripts/run_diag.sh

cd /root/autodl-tmp/code-agent

MODEL=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct

echo "==== few-shot multi-turn 全量 ===="
python -m src.eval.evaluate \
    --model $MODEL \
    --datasets mbpp_test \
    --output_dir ./outputs/diag_fewshot \
    --mode multi_turn \
    --temperature 0.7 \
    --max_turns 10 \
    --data_dir /root/autodl-tmp/datasets \
    --few_shot

