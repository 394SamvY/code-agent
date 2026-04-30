#!/bin/bash
# Focused 32-sample eval smoke for 2xA800-80GB.
#
# This script keeps the normal OJ-like verl validation path, but pins the
# currently useful A800 knobs, enables GPU monitoring, and applies the opt-in
# short-thinking controls used for debugging response-budget waste.
#
# Usage:
#   bash scripts/evaluate_2xa800_32_debug.sh codecontests_test
#   FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET=1536 bash scripts/evaluate_2xa800_32_debug.sh codecontests_test
#   GPU_MEMORY_UTILIZATION=0.94 VAL_BATCH_SIZE=24 AGENT_WORKERS=24 MAX_NUM_SEQS=48 \
#     MAX_NUM_BATCHED_TOKENS=49152 bash scripts/evaluate_2xa800_32_debug.sh codecontests_test

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

DATASET_ARG="${1:-codecontests_test}"
MODEL_PATH="${2:-/root/autodl-tmp/models/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/verl_baseline_eval}"

dataset_tag="$(basename "$DATASET_ARG" .parquet | tr -c 'A-Za-z0-9_.-' '_')"
timestamp="$(date +%Y%m%d_%H%M%S)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS="${NUM_GPUS:-2}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-32}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
export AGENT_WORKERS="${AGENT_WORKERS:-16}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
export VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"
export VAL_TOP_P="${VAL_TOP_P:-0.95}"
export VAL_TOP_K="${VAL_TOP_K:-20}"
export VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-true}"
export LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-1}"

# Short-thinking controls.
#
# CODE_AGENT_PROMPT_STYLE=short_thinking:
#   Soft control. src/verl_dataset_adapter.py dynamically appends a short
#   instruction to the decoded system prompt. It does not rewrite parquet files
#   and does not disable Qwen3 thinking mode.
#
# CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET / CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET:
#   Hard controls implemented by src.verl_agent_loop.CodeAgentToolAgentLoop
#   inside verl AgentLoopWorker processes. They cap one assistant generation
#   call, not the whole trajectory. MAX_RESPONSE_LENGTH stays 8192, so
#   multi-turn repair can still use remaining budget after tools.
export CODE_AGENT_PROMPT_STYLE="${CODE_AGENT_PROMPT_STYLE:-short_thinking}"
export FIRST_ASSISTANT_TURN_TOKEN_BUDGET="${FIRST_ASSISTANT_TURN_TOKEN_BUDGET:-3072}"
export FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET="${FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET:-2048}"
export CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET="${CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET:-$FIRST_ASSISTANT_TURN_TOKEN_BUDGET}"
export CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET="${CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET:-$FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET}"

export RUN_NAME="${RUN_NAME:-debug32_${dataset_tag}_f${CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET}_u${CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET}_${timestamp}}"
run_dir="$OUTPUT_ROOT/$RUN_NAME"
gpu_csv="$run_dir/gpu_monitor.csv"
summary_file="$run_dir/summary.txt"

mkdir -p "$run_dir"

monitor_pid=""
cleanup() {
    if [ -n "$monitor_pid" ] && kill -0 "$monitor_pid" >/dev/null 2>&1; then
        kill "$monitor_pid" >/dev/null 2>&1 || true
        wait "$monitor_pid" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

bash scripts/monitor_gpu.sh 2 "$gpu_csv" &
monitor_pid="$!"

bash scripts/evaluate_baseline_with_verl.sh "$DATASET_ARG" "$MODEL_PATH"
cleanup
monitor_pid=""

generation_jsonl="$run_dir/generations/0.jsonl"
if [ ! -f "$generation_jsonl" ]; then
    generation_jsonl="$run_dir/generations/partial_0.jsonl"
fi

python3 scripts/analyze_eval_generations.py "$generation_jsonl" --gpu-csv "$gpu_csv" | tee "$summary_file"

echo ""
echo "Run directory: $run_dir"
echo "Summary: $summary_file"
