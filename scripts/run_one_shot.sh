#!/bin/bash
# Phase 1: One-shot GRPO Training
# 用法: bash scripts/run_one_shot.sh [config_path]

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-"configs/one_shot.yaml"}

python -m src.train.one_shot_grpo "$CONFIG"
