# Baseline Eval GPU 调参记录

日期：2026-04-29

范围：通过 `scripts/evaluate_baseline_with_verl.sh` 在 2xA800 80GB 上运行 OJ-like baseline 评测。本文针对 validation/eval 吞吐，而非 GRPO 训练质量。

## 主要吞吐调节参数

最重要的变量：

| 变量 | 当前高利用 smoke 值 | 作用 |
| --- | ---: | --- |
| `GPU_MEMORY_UTILIZATION` | `0.82` | rollout 引擎 GPU 显存预留，主要是 KV-cache 余量。值越高允许更多并发的长上下文解码，但本身不线性提升 eval 速度。 |
| `MAX_NUM_SEQS` | `32` | 每个 rollout server scheduler 的最大并发序列数。过低则即使有空闲显存，GPU 也会闲置。 |
| `MAX_NUM_BATCHED_TOKENS` | `32768` | Scheduler 的 batch prefill/decode token 预算。过低会限制长 prompt 和长 thinking trajectory。 |
| `AGENT_WORKERS` | `16` | 异步 agent-loop worker 数量。必须足够多才能喂饱 rollout server，尤其是 tool call 和沙箱执行会引入 CPU/wall-clock 延迟。 |
| `VAL_BATCH_SIZE` | `16` | 此 patch 后 eval 路径中的 validation dataloader batch size。需足够大以喂饱 `AGENT_WORKERS`；在近期 verl 日志中标记为 deprecated（因为推理引擎自行调度），但仍控制着 validation 循环的 batch 粒度。 |
| `MAX_PROMPT_LENGTH` | `4096` | Prompt 预算。更大的值能覆盖更多数据集样本，但消耗 KV-cache 和 batching 容量。 |
| `MAX_RESPONSE_LENGTH` | `8192` | 整条 trajectory 的 response 预算。Qwen3 长 thinking 会消耗大部分；对单样本耗时和 KV-cache 压力影响很大。 |
| `MAX_MODEL_LEN` | `12288` | rollout 引擎看到的 prompt + response 容量。必须覆盖 `4096 + 8192`。 |
| `NUM_GPUS` / `ROLLOUT_TP` | `2` / `1` | 当前 eval 使用 2 GPU，rollout tensor parallel size=1，本质上是数据并行的 rollout server。 |

采样变量如 `VAL_TEMPERATURE`、`VAL_TOP_P`、`VAL_TOP_K` 不直接提高 GPU 利用率。它们通过改变输出长度、tool-call 频率和 accepted rate 间接影响。当前 Qwen3 thinking smoke 使用：

```bash
VAL_TEMPERATURE=0.6
VAL_TOP_P=0.95
VAL_TOP_K=20
VAL_DO_SAMPLE=true
```

## 已对比的配置

中断的 `smoke_structured_2gpu_` 运行在 shape mismatch 前完成了 264 条 partial 样本。相关配置：

```bash
VAL_BATCH_SIZE=8
AGENT_WORKERS=8
MAX_NUM_SEQS=16
GPU_MEMORY_UTILIZATION=0.55
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=8192
```

聚焦的高利用 smoke 使用：

```bash
VAL_BATCH_SIZE=16
AGENT_WORKERS=16
MAX_NUM_SEQS=32
GPU_MEMORY_UTILIZATION=0.82
MAX_NUM_BATCHED_TOKENS=32768
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=8192
```

高利用 smoke 中观察到的 GPU 实时显存约 `67.4GB`/A800，decode 密集型时段利用率常在 `90-100%`。

## 加速预估

显存从约 `47GB` 提升到 `67GB`/GPU，约 `1.43x` 更多驻留显存。但 eval 速度不与显存线性相关，因为：

- 大部分运行时间是自回归解码，每条 trajectory 内部是串行的。
- Tool call 增加了 CPU 沙箱时间和 agent-loop 调度开销。
- Qwen3 过度思考产生极长输出；缩短输出长度比预留额外 KV-cache 更有效。
- `GPU_MEMORY_UTILIZATION` 主要增加并发序列数；只有在 `AGENT_WORKERS`、`MAX_NUM_SEQS` 和 batch size 能喂饱 server 时才有效。

对于旧的 264 样本运行，高利用设置的实际预期提升约 `1.2x-1.6x`，不保证达到 `1.43x`。因为新配置同时翻倍了 `AGENT_WORKERS` 和 `MAX_NUM_SEQS`，当 GPU decode 为瓶颈时应接近上限，当工具/CPU 或单条超长 trajectory 主导时应接近下限。

对于完整的 500 样本 eval，使用相同的预估：在旧 47GB 档位运行 `T` 小时的任务，在 67GB 档位下大约需要 `T / 1.2` 到 `T / 1.6` 小时，前提是无 shape mismatch 且输出长度相近。

从 `67GB` 推到约 `77GB` 只增加 `1.15x` 显存。预期的 wall-clock 改进更小，约 `1.03x-1.12x`，除非当前运行明显受 scheduler/KV-cache 限制。在 80GB A800 上也会减少 OOM 安全余量，因此在任何 500 样本运行前应先 32 样本验证。

## 77GB 探测计划

聚焦的 32 样本 smoke 命令：

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

关注点：

- rollout server 启动期间的 OOM 或分配器碎片化。
- 更稳定的 `90-100%` GPU 利用率，而非仅更高的预留显存。
- 相对于 67GB 运行的每样本耗时改进。
- Shape mismatch 是否复现。
- Accepted trajectory 在 `submit_solution: accepted` 后是否立即停止。

如果 `0.94` 不稳定，回退到 `GPU_MEMORY_UTILIZATION=0.90`，配合 `VAL_BATCH_SIZE=20`、`AGENT_WORKERS=20`、`MAX_NUM_SEQS=40`。

## 77GB 探测结果观察

`0.94` 探测无 OOM、无 shape mismatch 完成：

```text
outputs/verl_baseline_eval/smoke_sampling_t06_topk20_vbs24_32_mem094
```

与 67GB smoke 的对比：

| 运行 | GPU 显存 | 关键并发参数 | 秒/样本 | 结果 |
| --- | ---: | --- | ---: | --- |
| `smoke_sampling_t06_topk20_vbs16_32_v3` | 67.5GB | `16/16/32` | 12.3s | 完成，无 shape mismatch |
| `smoke_sampling_t06_topk20_vbs24_32_mem094` | 77.8GB | `24/24/48` | 14.2s | 完成，无 shape mismatch |

77GB 设置成功填满显存，但在此 32 样本 smoke 上更慢。可能原因是过度思考导致的长尾自回归解码：一旦 batch 在等待少量超长 trajectory，额外的 KV-cache 和更多的 agent worker 并不能明显改善 wall-clock 时间，反而可能增加调度开销。

当前默认推荐：

```bash
VAL_BATCH_SIZE=16
AGENT_WORKERS=16
MAX_NUM_SEQS=32
GPU_MEMORY_UTILIZATION=0.82
MAX_NUM_BATCHED_TOKENS=32768
```

下一步调参优先级是减少过度思考和截断同一 assistant turn 中 tool call 之后的多余文本，而非继续把显存推到接近 80GB。

## 2026-04-30 Short-Thinking 探测

> **2026-05-01 更新**：本文描述的 short-thinking 和 per-turn token budget 方案已全部废弃。详细分析见 `docs/debug/2026-05-01-thinking-budget-detour.md`。以下原文保留为历史上下文。

eval 的主要浪费是 Qwen3 在未闭合的 `<think>` 块中一直思考到用完 `MAX_RESPONSE_LENGTH=8192`。当时的 focused debug 脚本为：

```bash
bash scripts/evaluate_2xa800_32_debug.sh codecontests_test
```

保持 `enable_thinking=true` 和 `MAX_RESPONSE_LENGTH=8192`，但增加了 per-assistant-turn 硬预算：

```bash
CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET=3072
CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET=2048
```

token budget 限制每轮 assistant 生成，而非整条 trajectory。这让有用的多轮修复仍能使用完整的 response budget，而超长的首轮 thinking 在约 3072 token 处停止，修复轮被压缩到 2048 token。

当时 32 样本结果：

| 运行 | GPU 显存 | 关键并发 | 秒/样本 | Accepted | no-tool rate | 平均输出字符 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `smoke_sampling_t06_topk20_vbs16_32_v3` | 67.5GB | `16/16/32` | ~12.3s | 18.8% | 78.1% | 28,192 |
| `debug32_shortthink_tb3072_v1` | 66.0GB | `16/16/32` | 13.3s | 31.2% | 31.2% | 18,220 |
| `debug32_shortthink_tb3072_mem094_v1` | 75.8GB | `24/24/48` | 15.2s | 28.1% | 31.2% | 20,302 |

重要修正：早期 `tb3072` 运行使用了 TaskRunner 侧的 monkey patch，后续 499 样本 debug run 显示硬预算和 terminal stop 并未进入实际的 `AgentLoopWorker` 进程。这些结果仅作为调试对照，不作为正式 baseline。

当时默认推荐保持 67GB 档位；77GB 档位虽然填满显存但在该 32 样本 smoke 中更慢。当时的下一步 eval 调参方向是 budget/prompt 行为和 malformed tool call，而非继续提高 `GPU_MEMORY_UTILIZATION`。
