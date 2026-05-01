# verl baseline eval 调试记录 2026-04-28

本文记录 2*A800 80GB 上继续调试 OJ-like baseline validation 的过程。本次结论修正了
`docs/debug/verl_agent_loop_batch_failure_analysis_2026-04-25.md` 中“主要是多轮变长
trajectory batch 不安全”的早期判断。

## 目标

- 使用单个 `verl` 进程统一调度两张 A800。
- 让每张卡启动一个 SGLang server，并尽量提高 validation 并行度。
- `VAL_BATCH_SIZE > 1` 必须稳定通过，不能退回到“两张卡各跑一个 verl 进程”的数据分片 fallback。
- validation 过程中要增量落盘，避免长评测被中断后完全丢失已生成序列。

## 关键问题

最初 `VAL_BATCH_SIZE=1` 可以跑，`VAL_BATCH_SIZE>1` 会在 agent loop postprocess
阶段报 shape mismatch，例如一个 prompt tensor 宽度是 3475，另一个是 2048。

真实根因不是普通工具调用场景天然不能 batch，也不是 response 变长。verl 的
multi-turn response 会按 `response_length` pad/truncate，工具 observation 也在
response 侧通过 mask 处理。

问题出在数据过滤：

- 当前 parquet 的 `prompt` 是 JSON string。
- verl 默认 `RLHFDataset.maybe_filter_out_long_prompts` 在过滤时直接读取
  `doc[prompt_key]`，没有走自定义 dataset 的 JSON 解码和 chat template 路径。
- 因此过长的真实 chat prompt 被低估，`filter_overlong_prompts=true` 也没有过滤掉。
- 后续 `AgentLoopWorker._agent_loop_postprocess` 调用 tokenizer pad 时，`max_length`
  不会截断已经超过上限的 prompt，于是 batch 内 tensor 宽度不一致，`torch.cat`
  失败。

修复点是 `src/verl_dataset_adapter.py`：在 `OJLikeRLHFDataset` 中覆盖
`maybe_filter_out_long_prompts`，过滤前先把 JSON string 还原为 messages，并用真实
chat template token 长度判断。

## 失败尝试

为了让 validation 中途写出结果，曾尝试通过 `sitecustomize.py` 在所有 Python 进程
启动时自动导入 `src.verl_runtime_patch`。

这个方案不可用：Ray GPU worker 在 per-worker `CUDA_VISIBLE_DEVICES` 最终确定前就会
导入 `sitecustomize.py`，进而过早 import verl/torch，导致 NCCL 报
`Duplicate GPU detected`。因此不要再使用 `sitecustomize.py` 或
`CODE_AGENT_PATCH_VERL` 这类全局启动钩子。

最终方案是 `scripts/verl_main_wrapper.py` 自定义 `CodeAgentTaskRunner`，只在 CPU
TaskRunner Ray actor 内调用 `src.verl_runtime_patch.apply_patches()`。这样 patch 覆盖
validation 主流程，不影响 GPU worker 的 CUDA 绑定。

## 最终方案

当前主入口是：

```bash
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

默认资源配置：

```text
NUM_GPUS=全部可见 GPU
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=8192
MAX_MODEL_LEN=MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH
VAL_BATCH_SIZE=8
AGENT_WORKERS=8
GPU_MEMORY_UTILIZATION=0.70
MAX_NUM_BATCHED_TOKENS=16384
MAX_NUM_SEQS=16
FILTER_OVERLONG_PROMPTS=true
LOG_VAL_GENERATIONS=1
```

这个分配依据是 `docs/specs/verl_parquet_dataset_analysis.md` 的 prompt 分布：`4096`
已经覆盖 LiveCodeBench test 全部样本，CodeContests test 也只剩 1/500 条超过上限；
把同样的 12288 总上下文预算更多留给 multi-turn response 更合理。

`src/verl_runtime_patch.py` 做两件事：

- 让 stdlib `json` 可以序列化 numpy scalar / ndarray，避免 validation dump 失败。
- patch `RayPPOTrainer._validate`，每完成一个 validation batch 就追加写
  `generations/partial_0.jsonl`；整轮 validation 结束后仍写 verl 原始的
  `generations/0.jsonl`。

`MAX_NUM_SEQS` 是每个 SGLang server 允许同时驻留的最大请求数，不等于总 rollout
数量。实际并发还受 `VAL_BATCH_SIZE`、`AGENT_WORKERS`、`NUM_GPUS / ROLLOUT_TP` 和
SGLang token budget 共同限制。

## 验证记录

1GPU partial dump smoke 已通过：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 VAL_MAX_SAMPLES=2 VAL_BATCH_SIZE=1 \
AGENT_WORKERS=1 MAX_RESPONSE_LENGTH=128 LOG_VAL_GENERATIONS=1 \
RUN_NAME=smoke_partial_taskrunner_20260428_195004 \
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

结果：

- `generations/partial_0.jsonl` 写出 2 条。
- `generations/0.jsonl` 写出 2 条。

2GPU 统一调度 smoke 已通过：

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 VAL_MAX_SAMPLES=4 VAL_BATCH_SIZE=4 \
AGENT_WORKERS=4 MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=128 \
MAX_MODEL_LEN=8320 MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=16384 \
GPU_MEMORY_UTILIZATION=0.55 LOG_VAL_GENERATIONS=1 \
RUN_NAME=smoke_unified_2gpu_batch4_taskrunner_20260428_195208 \
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

结果：

- 单个 `verl` 进程完成 validation。
- 两个 SGLang server 分别绑定 GPU0 和 GPU1。
- `generations/partial_0.jsonl` 写出 4 条。
- `generations/0.jsonl` 写出 4 条。

清理 sharded fallback 和 `sitecustomize.py` 后，又用同配置跑过一次：

```bash
RUN_NAME=smoke_unified_2gpu_cleanup_20260428_200331 \
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

结果同样通过，`partial_0.jsonl` 和 `0.jsonl` 都写出 4 条，日志中两个 SGLang
server 分别为 `cuda_visible_devices='0'` 和 `cuda_visible_devices='1'`。

## 推荐正式命令

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VAL_MAX_SAMPLES=500 \
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

后续如果显存仍有明显余量，优先尝试提高 `VAL_BATCH_SIZE` 和 `AGENT_WORKERS`；如果
SGLang 侧排队明显但显存稳定，再尝试提高 `MAX_NUM_SEQS`。不要恢复 sharded fallback
作为默认方案，否则会回到数据层并行，无法验证 verl/Ray 的统一调度能力。
