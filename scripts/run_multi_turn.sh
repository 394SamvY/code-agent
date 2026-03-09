#!/bin/bash
# Phase 2: Multi-turn Agentic GRPO Training
# 用法: bash scripts/run_multi_turn.sh [config_path]

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-"configs/multi_turn.yaml"}

python -m src.train.multi_turn_grpo "$CONFIG"
