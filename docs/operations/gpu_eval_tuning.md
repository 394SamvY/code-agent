# Baseline Eval GPU Tuning Notes

Date: 2026-04-29

Scope: OJ-like baseline evaluation through `scripts/evaluate_baseline_with_verl.sh`
on 2xA800 80GB. This note is for validation/eval throughput, not GRPO training
quality.

## Main Throughput Knobs

The most important variables are:

| Variable | Current high-util smoke | Effect |
| --- | ---: | --- |
| `GPU_MEMORY_UTILIZATION` | `0.82` | Reserves rollout-engine GPU memory, mainly KV-cache headroom. Higher values allow more concurrent long-context decoding, but do not linearly speed up eval by themselves. |
| `MAX_NUM_SEQS` | `32` | Max concurrent sequences per rollout server scheduler. If too low, GPUs sit idle even with free memory. |
| `MAX_NUM_BATCHED_TOKENS` | `32768` | Scheduler token budget for batching prefill/decode work. If too low, it throttles long prompts and long thinking trajectories. |
| `AGENT_WORKERS` | `16` | Number of async agent-loop workers. This must be high enough to keep rollout servers fed, especially because tool calls and sandbox execution add CPU/wall-clock stalls. |
| `VAL_BATCH_SIZE` | `16` | Validation dataloader batch size in this patched eval path. It should be large enough to feed `AGENT_WORKERS`; in recent verl logs this is marked deprecated because inference engines self-schedule, but it still controls our validation loop batch granularity. |
| `MAX_PROMPT_LENGTH` | `4096` | Prompt budget. Higher values increase accepted dataset coverage but consume KV-cache and batching capacity. |
| `MAX_RESPONSE_LENGTH` | `8192` | Whole trajectory response budget. Long Qwen3 thinking can consume most of this; it strongly affects per-sample runtime and KV-cache pressure. |
| `MAX_MODEL_LEN` | `12288` | Prompt + response capacity seen by the rollout engine. Must cover `4096 + 8192`. |
| `NUM_GPUS` / `ROLLOUT_TP` | `2` / `1` | Current eval uses 2 GPUs with rollout tensor parallel size 1, effectively using data-parallel rollout servers. |

Sampling variables such as `VAL_TEMPERATURE`, `VAL_TOP_P`, and `VAL_TOP_K` do not
directly increase GPU utilization. They matter indirectly because they change output
length, tool-call frequency, and accepted rate. Current Qwen3 thinking smoke uses:

```bash
VAL_TEMPERATURE=0.6
VAL_TOP_P=0.95
VAL_TOP_K=20
VAL_DO_SAMPLE=true
```

## Current Configurations Compared

The interrupted `smoke_structured_2gpu_` run completed 264 partial samples before a
shape mismatch. Its relevant config was:

```bash
VAL_BATCH_SIZE=8
AGENT_WORKERS=8
MAX_NUM_SEQS=16
GPU_MEMORY_UTILIZATION=0.55
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=8192
```

The focused high-util smoke used:

```bash
VAL_BATCH_SIZE=16
AGENT_WORKERS=16
MAX_NUM_SEQS=32
GPU_MEMORY_UTILIZATION=0.82
MAX_NUM_BATCHED_TOKENS=32768
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=8192
```

Observed live GPU memory in the high-util smoke was about `67.4GB` per A800, with
utilization often in the `90-100%` range during decode-heavy periods.

## Speedup Estimate

Memory moved from roughly `47GB` to `67GB` per GPU, about `1.43x` more resident GPU
memory. However, eval speed does not scale linearly with memory because:

- A large part of runtime is autoregressive decoding, where token generation is
  sequential inside each trajectory.
- Tool calls add CPU sandbox time and agent-loop scheduling overhead.
- Qwen3 over-thinking creates very long outputs; reducing output length can help more
  than reserving additional KV cache.
- `GPU_MEMORY_UTILIZATION` mostly enables more in-flight sequences; it is only useful
  if `AGENT_WORKERS`, `MAX_NUM_SEQS`, and batch size keep the server fed.

For the old 264-sample run, the practical expected improvement from the high-util
setting is roughly `1.2x-1.6x`, not `1.43x` guaranteed. Because the new config also
doubles `AGENT_WORKERS` and `MAX_NUM_SEQS`, it should be closer to the upper side when
GPU decode is the bottleneck, and closer to the lower side when tool/CPU or very long
single trajectories dominate.

For a full 500-sample eval, use the same estimate: a run that took `T` hours at the
old 47GB-ish setting would likely take about `T / 1.2` to `T / 1.6` hours at the
67GB setting, assuming no shape mismatch and similar output lengths.

Pushing from `67GB` to about `77GB` is only another `1.15x` memory increase. Expected
wall-clock improvement is likely smaller, around `1.03x-1.12x`, unless the current
run is clearly scheduler/KV-cache limited. It also reduces OOM safety margin on 80GB
A800s, so it should be validated with 32 samples before any 500-sample run.

## 77GB Probe Plan

Focused 32-sample smoke command:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NUM_GPUS=2 \
VAL_MAX_SAMPLES=32 \
VAL_BATCH_SIZE=24 \
AGENT_WORKERS=24 \
MAX_NUM_SEQS=48 \
GPU_MEMORY_UTILIZATION=0.94 \
MAX_NUM_BATCHED_TOKENS=49152 \
RUN_NAME=smoke_sampling_t06_topk20_vbs24_32_mem094 \
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

Watch for:

- OOM or allocator fragmentation during rollout server startup.
- More stable `90-100%` GPU utilization, not just higher reserved memory.
- Per-sample runtime improvement versus the 67GB run.
- Shape mismatch recurrence.
- Accepted trajectories stopping immediately after `submit_solution: accepted`.

If `0.94` is unstable, back off to `GPU_MEMORY_UTILIZATION=0.90` with
`VAL_BATCH_SIZE=20`, `AGENT_WORKERS=20`, and `MAX_NUM_SEQS=40`.

## Observed 77GB Probe Result

The `0.94` probe completed without OOM or shape mismatch:

```text
outputs/verl_baseline_eval/smoke_sampling_t06_topk20_vbs24_32_mem094
```

Comparison against the 67GB smoke:

| Run | GPU memory | Key concurrency | high-memory seconds/sample | Result |
| --- | ---: | --- | ---: | --- |
| `smoke_sampling_t06_topk20_vbs16_32_v3` | 67.5GB | `16/16/32` | 12.3s | Completed, no shape mismatch |
| `smoke_sampling_t06_topk20_vbs24_32_mem094` | 77.8GB | `24/24/48` | 14.2s | Completed, no shape mismatch |

The 77GB setting filled memory successfully but was slower on this 32-sample smoke.
The likely reason is long-tail autoregressive decoding from overlong thinking: once
the batch is waiting on a small number of very long trajectories, extra KV cache and
more agent workers do not improve wall-clock time much and may add scheduling
overhead.

Current default recommendation:

```bash
VAL_BATCH_SIZE=16
AGENT_WORKERS=16
MAX_NUM_SEQS=32
GPU_MEMORY_UTILIZATION=0.82
MAX_NUM_BATCHED_TOKENS=32768
```

Next tuning priority is reducing overlong thinking and truncating text generated
after a tool call in the same assistant turn, not pushing memory closer to 80GB.
