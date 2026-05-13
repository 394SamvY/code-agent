# 评测效率实验计划

更新日期：2026-05-13

## 目标

当前 `scripts/evaluate_baseline_with_verl.sh` 的完整评测目标是：

- `data/verl/codecontests_test.parquet`：500 题，2 小时内完成。
- `data/verl/livecodebench_test.parquet`：611 题，2 小时内完成。

当前结论：目标已达成。最终默认参数已写入 `scripts/evaluate_baseline_with_verl.sh`，两次正式 full eval 分别为 CodeContests 499 条 45m02s、LiveCodeBench 611 条 55m33s。CodeContests 少 1 条是因为 `MAX_PROMPT_LENGTH=4096` 下 `FILTER_OVERLONG_PROMPTS=true` 过滤了一条 overlong prompt。

第一阶段实验不直接追求一次调到目标，而是先回答两个问题：

1. 当前 16 样本 smoke 是否没有让评测链路满载。
2. 评测慢主要来自 SGLang 吞吐不足，还是单题 assistant 生成 token 过多。

## 最终默认参数

当前 baseline / full eval 默认使用：

```text
MAX_RESPONSE_LENGTH=8192
MAX_MODEL_LEN=12288
VAL_BATCH_SIZE=64
AGENT_WORKERS=64
MAX_NUM_SEQS=64
GPU_MEMORY_UTILIZATION=0.88
VAL_TEMPERATURE=0
VAL_DO_SAMPLE=false
ENABLE_THINKING=true
ROLLOUT_TP=1
```

这些默认值已经同步到 `scripts/evaluate_baseline_with_verl.sh`。后续如需探索更激进并发或更高 `GPU_MEMORY_UTILIZATION`，应通过命令行显式覆盖，而不是改默认值。

## 统计口径

后续所有实验统一记录以下指标：

| 指标 | 说明 |
| --- | --- |
| wall time | 脚本 `start` 到 `end` 的总耗时 |
| ready-to-end time | SGLang HTTP server ready 后到 `end` 的耗时 |
| assistant tokens/problem | 只统计 `messages` 中 `role == "assistant"` 的 `content` 和 `tool_calls`，不统计 `role == "tool"` 的 judge 返回 |
| assistant tok/s | `assistant tokens / ready-to-end time` |
| tool response tokens/problem | 只统计 `role == "tool"` 的 observation token，用于确认环境反馈占比 |
| score / acc | verl validation 输出的效果指标；当前新轨迹额外记录 `acc_any`、`best_submit_pass_rate`、`last_submit_pass_rate` 和 `reward_breakdown` |
| tool calls/problem | 平均工具调用次数 |
| terminal_reason | `no_tool_call`、`accepted`、`tool_call_limit_exhausted` 等终止原因分布 |
| cap hit rate | assistant tokens 打满 `MAX_RESPONSE_LENGTH` 的比例 |

token 数使用当前评测模型目录中的真实 Hugging Face tokenizer 计算，不用字符估计：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
num_tokens = len(tok.encode(text, add_special_tokens=False))
```

## 已有 smoke 快照

最近三次 16 样本 smoke 的 assistant-only token 统计如下。这里已经排除了 tool 返回。

| run | 样本 | assistant tokens | 平均/题 | median | 打满 28672 数 | ready 后 assistant tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `smoke_sft_qwen3_8b_toolagent_think_2gpu_16_fast1` | 16 | 270,562 | 16,910 | 19,513 | 7 | 260 |
| `smoke_sft_qwen3_8b_toolagent_think_2gpu_16_test2` | 16 | 405,015 | 25,313 | 28,672 | 10 | 317 |
| `smoke_sft_qwen3_8b_toolagent_think_2gpu_16_test3` | 16 | 360,403 | 22,525 | 28,672 | 8 | 312 |

48 条合计：

- assistant tokens 总数：1,035,980。
- 平均 assistant tokens/problem：21,583。
- 打满 `MAX_RESPONSE_LENGTH=28672`：25/48。
- tool response tokens/problem 只有几百 token，主量来自 assistant 自身的 `<think>`、代码和 tool call。

初步结论：当前主要瓶颈更像是单题生成过长，而不是 tool observation 或日志落盘。

## 固定基线参数

除实验变量外，第一轮实验先固定以下参数：

```bash
CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
ROLLOUT_TP=1
VAL_TEMPERATURE=0
VAL_DO_SAMPLE=false
MAX_PROMPT_LENGTH=4096
GPU_MEMORY_UTILIZATION=0.88
MAX_NUM_SEQS=48
AGENT_WORKERS=32
ENABLE_THINKING=true
```

这组是第一轮效率实验的历史基线。当前 full eval 默认已改为 `MAX_RESPONSE_LENGTH=8192`、`VAL_BATCH_SIZE=64`、`AGENT_WORKERS=64`、`MAX_NUM_SEQS=64`；详见上面的“最终默认参数”。

`ROLLOUT_TP=1` 表示 2 卡上有 2 个独立 SGLang rollout replica。对 8B 模型和 2xA800，这通常比 `ROLLOUT_TP=2` 更适合吞吐评测。

当前 verl + SGLang async server 路径下，真正需要重点观察的并发参数是：

| eval 环境变量 | 实际 SGLang / verl 含义 |
| --- | --- |
| `GPU_MEMORY_UTILIZATION` | 传给 SGLang `mem_fraction_static`，控制模型权重 + KV cache pool 的静态显存比例 |
| `MAX_NUM_SEQS` | 传给 SGLang `max_running_requests`，限制每个 SGLang replica 同时 running 的 request 数 |
| `AGENT_WORKERS` | verl agent loop 并发 worker 数，决定同时驱动多少条 multi-turn agent trajectory |

`MAX_NUM_BATCHED_TOKENS` 虽然会写入 verl rollout config，但当前 SGLang async server 启动参数没有把它传给 SGLang；它主要是 vLLM 路径的有效参数。因此第一轮 SGLang 并发实验不把 `MAX_NUM_BATCHED_TOKENS` 作为判断依据。若后续要调 SGLang token batching / prefill，需要通过 `actor_rollout_ref.rollout.engine_kwargs.sglang.*` 显式传 SGLang 原生参数。

## 实验 A：样本数和满载程度

目的：确认 16 样本 smoke 是否低估了真实吞吐。

保持当前大 response budget，只改样本数：

| run | VAL_MAX_SAMPLES | MAX_RESPONSE_LENGTH | 观察点 |
| --- | ---: | ---: | --- |
| A1 | 16 | 28672 | 对齐已有 smoke |
| A2 | 64 | 28672 | 看 assistant tok/s 是否明显上升 |
| A3 | 128 | 28672 | 看是否接近稳定满载吞吐 |

命令示例：

```bash
RUN_NAME=A2_n64_mr28672 VAL_MAX_SAMPLES=64 \
bash scripts/evaluate_baseline_with_verl.sh codecontests_test

RUN_NAME=A3_n128_mr28672 VAL_MAX_SAMPLES=128 \
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

判断：

- 如果 `16 -> 64 -> 128` 的 assistant tok/s 明显上升，说明 16 样本没有打满链路。
- 如果 tok/s 基本不变，说明当前慢主要不是 batch 太小，而是单题 trajectory 太长或 agent loop 长尾。

### A3 已完成结果

run：`outputs/verl_baseline_eval/A3_n128_mr28672`

配置：

```bash
RUN_NAME=A3_n128_mr28672 VAL_MAX_SAMPLES=128 MAX_RESPONSE_LENGTH=28672 \
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

时间：

| 指标 | 数值 |
| --- | ---: |
| start | 2026-05-07 23:38:37 |
| SGLang ready / validation_start | 2026-05-07 23:39:58 |
| end | 2026-05-08 00:59:25 |
| wall time | 80m48s |
| ready-to-end / validation elapsed | 79m19s |
| 启动和准备开销 | 约 81s |

整体结果：

| 指标 | 数值 |
| --- | ---: |
| samples | 128 |
| score mean | 0.2125 |
| acc | 21.09% |
| assistant tokens | 2,615,080 |
| assistant tokens/problem | 20,430 |
| assistant token p50 | 28,540 |
| assistant token p90 | 28,672 |
| assistant tok/s | 549.5 |
| response cap hit | 75/128 |
| assistant cap hit | 50/128 |
| tool response tokens/problem | 404.9 |
| tool calls/problem | 6.15 |
| max tool calls | 22 |
| tool wall time | 517.8s |
| judge runtime | 512.5s |

按 batch 的 summary：

| batch | elapsed | assistant tok/s | assistant tokens/problem | cap hit | score | acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1446.5s | 557.7 | 25,211 | 25/32 | 0.0627 | 6.25% |
| 1 | 1065.2s | 513.5 | 17,093 | 14/32 | 0.3154 | 31.25% |
| 2 | 1080.0s | 543.9 | 18,358 | 16/32 | 0.2838 | 28.12% |
| 3 | 1166.6s | 577.7 | 21,060 | 20/32 | 0.1879 | 18.75% |

terminal / tool 分布：

| 分布项 | 数值 |
| --- | ---: |
| `no_tool_call` | 96 |
| `tool_call_limit_exhausted` | 31 |
| `unknown` / missing | 1 |
| `num_tool_calls=0` | 54 |
| `num_tool_calls=2` | 28 |
| `num_tool_calls=22` | 31 |
| parse failures | 4 |

按终止原因统计 assistant-only token：

| terminal_reason | n | assistant tokens | 平均/题 | p50 | cap hit | acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_tool_call` | 96 | 2,221,620 | 23,142 | 28,672 | 50/96 | 25.0% |
| `tool_call_limit_exhausted` | 31 | 366,392 | 11,819 | 9,590 | 0/31 | 9.68% |
| missing | 1 | 27,068 | 27,068 | 27,068 | 0/1 | 0.0% |

关键结论：

- 16 样本 smoke 明显低估吞吐。之前 16 样本约 `260-317 assistant tok/s`，A3 到 `549.5 assistant tok/s`，说明 128 样本能更充分打满 SGLang/agent 链路。
- 当前主要瓶颈仍是单题 assistant 输出过长，不是 tool observation 或 judge。`tool_wall_time=517.8s` 只占 ready-to-end 的约 10.9%，主耗时是 SGLang decode。
- 最大 token 来源是 `no_tool_call` 长输出，尤其 `num_tool_calls=0` 的 54 条，平均接近打满 `MAX_RESPONSE_LENGTH=28672`。这说明模型经常长时间思考但没有进入工具调用。
- `tool_call_limit_exhausted` 不是最大 token 来源，但暴露出 accepted 后继续调用工具的行为问题。当前 reward 只消费 agent loop 记录的结构化 `code_agent_tool_events`；`acc_final` 按最后一次 `submit_solution` 计算，`acc_any` 记录任意一次正式提交 AC。历史 A3 运行早于这些额外诊断字段，因此这里只能把 `acc` 当作 last-submit 口径解读。
- 这符合后续 RL 可优化方向：通过 `R_debug_prm` 奖励有效修复，通过 `R_bad_pattern` 惩罚 accepted 后继续工具调用、无工具长思考、撞工具上限等行为。

按 A3 结果外推：

```text
CodeContests 500 题: 500 * 20,430 / 549.5 / 3600 ≈ 5.16h
LiveCodeBench 611 题: 611 * 20,430 / 549.5 / 3600 ≈ 6.31h
```

若吞吐维持 `~550 assistant tok/s`，2 小时内完成的 token budget 约为：

```text
500 题: <= 7,920 assistant tokens/problem
611 题: <= 6,480 assistant tokens/problem
```

因此下一步优先做实验 B。并发参数可以继续测，但当前更大的收益来自压低单题输出长度和通过 RL 改善 tool / stop 行为。

## 实验 B：response budget

目的：量化输出长度对速度和效果的影响。

固定 `VAL_MAX_SAMPLES=128`，只改 `MAX_RESPONSE_LENGTH`：

| run | MAX_RESPONSE_LENGTH | 观察点 |
| --- | ---: | --- |
| B1 | 4096 | 接近 500 题 2 小时所需 token/题量级 |
| B2 | 8192 | 可能的效率/效果折中点 |
| B3 | 16384 | 保留较长思考，但比当前少约一半 |
| B4 | 28672 | 当前基线 |

命令示例：

```bash
RUN_NAME=B1_n128_mr4096 VAL_MAX_SAMPLES=128 MAX_RESPONSE_LENGTH=4096 \
bash scripts/evaluate_baseline_with_verl.sh codecontests_test

RUN_NAME=B2_n128_mr8192 VAL_MAX_SAMPLES=128 MAX_RESPONSE_LENGTH=8192 \
bash scripts/evaluate_baseline_with_verl.sh codecontests_test

RUN_NAME=B3_n128_mr16384 VAL_MAX_SAMPLES=128 MAX_RESPONSE_LENGTH=16384 \
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

判断：

- 如果 `acc` 损失小，但 assistant tokens/problem 大幅下降，优先采用较短 response budget。
- 如果 `cap hit rate` 仍很高，说明模型仍在打满预算，需要继续做 thinking/tool 行为控制。
- 降低 `MAX_RESPONSE_LENGTH` 会同步降低默认 `MAX_MODEL_LEN`，提升 KV cache 可容纳并发，这可能带来额外吞吐收益。

### B3 已完成结果

run：`outputs/verl_baseline_eval/B3_A3same_n128_mr16384`

为保证和 A3 可比，B3 使用从 A3 `generations/0.jsonl` 中抽出的同一 128 条样本：

```bash
RUN_NAME=B3_A3same_n128_mr16384 MAX_RESPONSE_LENGTH=16384 VAL_MAX_SAMPLES=1000 \
bash scripts/evaluate_baseline_with_verl.sh data/verl/codecontests_test_A3_n128.parquet
```

日志里的 `Generating train split: 128 examples` 是 Hugging Face datasets 加载单个 parquet 时给出的默认 split 名称；本次实际 validation file 是 `data/verl/codecontests_test_A3_n128.parquet`，数据集长度为 128，没有重新随机抽样。

时间：

| 指标 | A3 `28672` | B3 `16384` | 变化 |
| --- | ---: | ---: | ---: |
| wall time | 80m48s | 40m19s | 2.00x faster |
| validation elapsed | 4758.9s | 2329.8s | 2.04x faster |
| assistant tok/s | 549.5 | 705.5 | +28.4% |

整体结果：

| 指标 | A3 `28672` | B3 `16384` | 变化 |
| --- | ---: | ---: | ---: |
| samples | 128 | 128 | - |
| score mean | 0.2125 | 0.1878 | -0.0246 |
| acc | 21.09% | 18.75% | -2.34pp |
| accepted 数 | 27/128 | 24/128 | -3 |
| assistant tokens | 2,615,080 | 1,643,692 | -37.1% |
| assistant tokens/problem | 20,430 | 12,841 | -37.1% |
| assistant token p50 | 28,540 | 16,280 | -42.9% |
| assistant token p90 | 28,672 | 16,384 | -42.9% |
| response cap hit | 75/128 | 88/128 | 更高 |
| assistant cap hit | 50/128 | 49/128 | 基本持平 |
| tool response tokens/problem | 404.9 | 376.2 | -7.1% |
| tool calls/problem | 6.15 | 4.91 | -20.1% |
| tool wall time | 517.8s | 298.1s | -42.4% |
| judge runtime | 512.5s | 295.4s | -42.4% |
| num turns mean | 14.05 | 11.68 | -16.9% |

按 batch 的 B3 summary：

| batch | elapsed | assistant tok/s | assistant tokens/problem | response cap hit | assistant cap hit | score | acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 680.0s | 686.1 | 14,580 | 26/32 | 14/32 | 0.0628 | 6.25% |
| 1 | 527.4s | 702.6 | 11,580 | 20/32 | 12/32 | 0.3131 | 31.25% |
| 2 | 604.6s | 660.0 | 12,470 | 22/32 | 11/32 | 0.2189 | 18.75% |
| 3 | 517.2s | 787.9 | 12,735 | 20/32 | 12/32 | 0.1565 | 18.75% |

terminal / tool 分布：

| 分布项 | A3 `28672` | B3 `16384` |
| --- | ---: | ---: |
| `no_tool_call` | 96 | 100 |
| `tool_call_limit_exhausted` | 31 | 19 |
| missing / `unknown` | 1 | 9 |
| `num_tool_calls=0` | 54 | 54 |
| `num_tool_calls=22` | 31 | 19 |
| parse failures | 4 | 6 |

同一 128 条样本上，正确性变化不是单向的：

| 变化类型 | 数量 |
| --- | ---: |
| A3 正确，B3 错误 | 9 |
| A3 错误，B3 正确 | 6 |
| 总 accepted 变化 | 27 -> 24 |

关键结论：

- `MAX_RESPONSE_LENGTH=16384` 是明显有效的效率改动：同样 128 条从 79m19s 降到 38m50s，约 2.04x faster。
- 加速来自两部分：assistant token/problem 降低 37.1%，同时由于较短上下文和更少长尾，assistant tok/s 从 549.5 提升到 705.5。
- 效果损失目前可接受但不是免费：acc 从 21.09% 降到 18.75%，少 3 条 accepted；样本级有 15 条发生正确性翻转。
- 16k 仍然没有解决“打满预算”问题。B3 的 response cap hit 是 88/128，assistant cap hit 仍有 49/128，说明模型行为仍是主要问题，不只是 budget 太大。
- B3 估算 500 题仍约 2.53h，611 题约 3.09h，距离 2 小时目标还差一截。

按 B3 结果外推：

```text
CodeContests 500 题: 500 * 12,841 / 705.5 / 3600 ≈ 2.53h
LiveCodeBench 611 题: 611 * 12,841 / 705.5 / 3600 ≈ 3.09h
```

若吞吐维持 `~705 assistant tok/s`，2 小时内完成的 token budget 约为：

```text
500 题: <= 10,159 assistant tokens/problem
611 题: <= 8,313 assistant tokens/problem
```

因此 16k 是一个好的中间点，但还不足以达成完整集 2 小时目标。下一步可以做两条线：一是继续测 `MAX_RESPONSE_LENGTH=12288` 或 `8192` 的边界；二是在 16k 下调 `AGENT_WORKERS/MAX_NUM_SEQS`，确认还有多少纯并发吞吐空间。

### B2 已完成结果

run：`outputs/verl_baseline_eval/B2_A3same_n128_mr8192`

同样使用 A3 抽出的固定 128 条样本：

```bash
RUN_NAME=B2_A3same_n128_mr8192 MAX_RESPONSE_LENGTH=8192 VAL_MAX_SAMPLES=1000 \
bash scripts/evaluate_baseline_with_verl.sh data/verl/codecontests_test_A3_n128.parquet
```

时间和吞吐：

| 指标 | A3 `28672` | B3 `16384` | B2 `8192` |
| --- | ---: | ---: | ---: |
| wall time | 80m48s | 40m19s | 17m04s |
| validation elapsed | 4758.9s | 2329.8s | 940.4s |
| assistant tok/s | 549.5 | 705.5 | 928.9 |
| assistant tokens/problem | 20,430 | 12,841 | 6,824 |
| assistant tokens total | 2,615,080 | 1,643,692 | 873,485 |

效果和行为：

| 指标 | A3 `28672` | B3 `16384` | B2 `8192` |
| --- | ---: | ---: | ---: |
| score mean | 0.2125 | 0.1878 | 0.2115 |
| acc | 27/128 | 24/128 | 27/128 |
| response cap hit | 75/128 | 88/128 | 98/128 |
| assistant cap hit | 50/128 | 49/128 | 54/128 |
| tool calls/problem | 6.15 | 4.91 | 3.38 |
| max tool calls | 22 | 22 | 22 |
| tool wall time | 517.8s | 298.1s | 354.7s |
| num turns mean | 14.05 | 11.68 | 8.68 |
| `no_tool_call` | 96 | 100 | 101 |
| `tool_call_limit_exhausted` | 31 | 19 | 7 |
| missing / `unknown` | 1 | 9 | 20 |

样本级正确性变化：

| 对比 | 前者正确、B2 错误 | 前者错误、B2 正确 | 总变化 |
| --- | ---: | ---: | ---: |
| A3 vs B2 | 7 | 7 | 14 |
| B3 vs B2 | 4 | 7 | 11 |

关键结论：

- B2 在同一 128 条上 acc 回到 A3 的 `27/128`，高于 B3 的 `24/128`，同时 validation elapsed 只有 `940.4s`。
- B2 相比 A3 快 `5.06x`，相比 B3 快 `2.48x`。这对 RL rollout 很关键，因为多采样时 token 成本会随 `n` 近似线性放大。
- `8192` 并没有在这组样本上降低总体 acc，但会改变具体做对的题：B2 相比 A3 有 7 条丢失、7 条新增；相比 B3 有 4 条丢失、7 条新增。
- B2 的 `response_cap_hit=98/128`、missing/unknown 上升到 20，说明 8192 是强约束，不是行为问题的根治。它把无效长思考更早截断，但仍需要 RL reward 去训练主动短路径和 accepted 后停止。
- 从 rollout 成本看，若正式训练需要同题多采样，`8192` 比 `12288/16384` 更适合作为主线预算；12k 只有在后续发现 8192 明显损害有效轨迹率时才有必要补测。

按 B2 结果外推：

```text
CodeContests 500 题: 500 * 6,824 / 928.9 / 3600 ≈ 1.02h
LiveCodeBench 611 题: 611 * 6,824 / 928.9 / 3600 ≈ 1.25h
```

### 第一次 accepted submit 前 token

为了判断 `8192` 是否过短，对 A3/B3 的正确样本按 `messages` 还原，统计到第一次 `submit_solution` 返回 accepted 为止的 assistant-only token。这里包含发起 accepted submit 的 assistant turn 和 tool call，不包含后续 assistant 收尾；accepted observation 约 20 token，单独看对结论影响很小。

| run | correct n | mean | p50 | p75 | p90 | max | `<=8192` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A3 `28672` | 27 | 2,010 | 1,371 | 2,625 | 3,214 | 8,394 | 26/27 |
| B3 `16384` | 24 | 1,551 | 1,359 | 2,072 | 2,703 | 3,730 | 24/24 |
| B2 `8192` | 27 | 1,762 | 1,377 | 2,258 | 2,772 | 4,600 | 27/27 |

补充观察：

- A3 正确样本里，accepted submit 前超过 `4096` 的只有 2/27，超过 `8192` 的只有 1/27。
- B3 的所有正确样本都在 `4096` 内完成第一次 accepted submit。
- B2 的所有正确样本都在 `8192` 内完成第一次 accepted submit，max 只有 4,600。
- A3 正确样本在第一次 accepted submit 之后仍平均继续生成 4,706 assistant token，p90 19,659，说明 accepted 后停止行为仍有明显浪费。
- 这支持继续尝试 `MAX_RESPONSE_LENGTH=8192`：正确代码和 submit 本身通常不需要 16k；风险主要在于未做对样本中的长思考是否能在 8192 内转化成有效工具调用。


## 实验 C：SGLang / agent 并发参数

目的：在较合理 token budget 下测 SGLang/agent loop 的吞吐上限。

建议在 `MAX_RESPONSE_LENGTH=8192`、固定 A3 128 条样本下测试。当前 B2 基线是：

```text
VAL_BATCH_SIZE=32
AGENT_WORKERS=32
MAX_NUM_SEQS=48
GPU_MEMORY_UTILIZATION=0.88
assistant_tok_s=928.9
validation_elapsed=940.4s
```

有效并发近似受以下下限控制：

```text
effective_concurrency ~= min(VAL_BATCH_SIZE, AGENT_WORKERS, NUM_SGLANG_REPLICAS * MAX_NUM_SEQS)
```

当前 `NUM_GPUS=2, ROLLOUT_TP=1`，即 2 个 SGLang replica。因此提高 `AGENT_WORKERS` 时，也要同步提高 `VAL_BATCH_SIZE`，否则每个 validation batch 仍只有 32 条样本。

建议按顺序测试：

| run | VAL_BATCH_SIZE | AGENT_WORKERS | MAX_NUM_SEQS | GPU_MEMORY_UTILIZATION | 观察点 |
| --- | ---: | ---: | ---: | ---: | --- |
| C1 | 32 | 32 | 48 | 0.88 | B2 当前基线 |
| C2 | 64 | 64 | 64 | 0.88 | 首选并发测试，看能否继续提升 tok/s、缩短 elapsed |
| C3 | 96 | 96 | 96 | 0.88 | 激进并发测试，观察 Ray/CPU 调度、长尾和稳定性 |
| C4 | 64 | 64 | 64 | 0.92 | 在 C2 稳定后再提高 KV cache 显存比例 |
| C5 | 64 | 64 | 64 | 0.84 | 若 C2/C4 无收益，测试较低显存预留是否不影响吞吐 |

命令示例：

```bash
RUN_NAME=C2_A3same_n128_mr8192_vb64_w64_s64_mem088 \
VAL_BATCH_SIZE=64 AGENT_WORKERS=64 MAX_NUM_SEQS=64 \
MAX_RESPONSE_LENGTH=8192 VAL_MAX_SAMPLES=1000 GPU_MEMORY_UTILIZATION=0.88 \
bash scripts/evaluate_baseline_with_verl.sh data/verl/codecontests_test_A3_n128.parquet

RUN_NAME=C4_A3same_n128_mr8192_vb64_w64_s64_mem092 \
VAL_BATCH_SIZE=64 AGENT_WORKERS=64 MAX_NUM_SEQS=64 \
MAX_RESPONSE_LENGTH=8192 VAL_MAX_SAMPLES=1000 GPU_MEMORY_UTILIZATION=0.92 \
bash scripts/evaluate_baseline_with_verl.sh data/verl/codecontests_test_A3_n128.parquet
```

判断：

- 如果 C2/C3 tok/s 明显提高且不 OOM，可以把 `AGENT_WORKERS` / `MAX_NUM_SEQS` 上调。
- 如果 C2 提升有限，说明 B2 已接近当前 2 卡 decode 上限，后续主要靠减少无效 token 和 accepted 后停止。
- 如果 C4 更快且稳定，可以考虑提高 `GPU_MEMORY_UTILIZATION`；如果 C4 无收益或不稳定，就维持 0.88。
- 如果 C5 不变慢，说明 8k 下 KV cache 容量不是瓶颈，可以考虑降低显存预留，为训练时其他组件留余量。
- 如果并发上调后 tok/s 不升，说明瓶颈在长序列 decode、agent loop 长尾或工具调用同步开销。
- 16 样本不适合测并发参数，因为样本数低于 `AGENT_WORKERS` 和 `MAX_NUM_SEQS` 的默认容量。

### C2 已完成结果

run：`outputs/verl_baseline_eval/C2_A3same_n128_mr8192_greedy_vb64_w64_s64_mem088`

配置：

```text
VAL_BATCH_SIZE=64
AGENT_WORKERS=64
MAX_NUM_SEQS=64
GPU_MEMORY_UTILIZATION=0.88
MAX_RESPONSE_LENGTH=8192
VAL_TEMPERATURE=0
VAL_DO_SAMPLE=false
```

对比 B2：

| 指标 | B2 `vb32/w32/s48` | C2 `vb64/w64/s64` | 变化 |
| --- | ---: | ---: | ---: |
| val batches | 4 | 2 | - |
| wall time | 17m04s | 12m33s | 1.36x faster |
| validation elapsed | 940.4s | 668.7s | 1.41x faster |
| assistant tok/s | 928.9 | 1304.7 | +40.5% |
| assistant tokens/problem | 6,824 | 6,817 | 基本不变 |
| score mean | 0.2115 | 0.1878 | -0.0237 |
| acc | 27/128 | 24/128 | -3 |
| response cap hit | 98/128 | 100/128 | +2 |
| assistant cap hit | 54/128 | 48/128 | -6 |
| tool calls/problem | 3.38 | 4.04 | +19.5% |
| tool wall time | 354.7s | 448.9s | +26.5% |
| num turns mean | 8.68 | 9.98 | +15.0% |
| `no_tool_call` | 101 | 97 | -4 |
| `tool_call_limit_exhausted` | 7 | 8 | +1 |
| missing / `unknown` | 20 | 23 | +3 |

样本级变化：

| 对比 | 数量 |
| --- | ---: |
| B2 正确，C2 错误 | 10 |
| B2 错误，C2 正确 | 7 |
| B2/C2 都正确 | 17 |

结论：

- C2 的吞吐收益很明显，`assistant_tok_s` 提升 40.5%，validation elapsed 缩短到 668.7s。
- 但 C2 的 acc 从 27/128 降到 24/128，且 missing/unknown 和 tool wall time 都上升。并发调高会改变 SGLang batching / agent 调度轨迹，即使 greedy eval 下也不应默认认为质量完全等价。
- 当前目标已经转向提高 rollout 效率，并且 B2 也只有一次 greedy 运行，不能把 3 条 acc 差异视为稳定质量差距。因此 full eval baseline 默认采用 C2 的高吞吐配置。
- 后续若需要“质量敏感”的复核，可以单独跑 B2 配置；训练和大规模评测优先使用 C2，避免在 rollout 阶段浪费主要时延。

### C6/C7 高显存预留尝试

C6 尝试在 C2 并发上把 `GPU_MEMORY_UTILIZATION` 提到 0.95：

```text
VAL_BATCH_SIZE=64
AGENT_WORKERS=64
MAX_NUM_SEQS=64
GPU_MEMORY_UTILIZATION=0.95
MAX_RESPONSE_LENGTH=8192
```

运行在第一批 64 条完成前失败，没有任何 generation 落盘。关键错误是：

```text
libgomp: Thread creation failed: Resource temporarily unavailable
ray.exceptions.ActorDiedError
```

失败后两张 GPU 显存完全释放，日志里没有明确 CUDA OOM。因此这次不是典型显存不足，而是 64 个 agent worker 加上 OpenMP/BLAS/tokenizer 等线程的瞬时峰值触发了系统线程资源上限。`mem0.95` 不是直接根因，但会改变 SGLang backpressure 和请求推进节奏，使 CPU/Ray 侧峰值更容易撞线。

C7 降低 agent/data 并发到 50，同时保留 `MAX_NUM_SEQS=64` 和 `GPU_MEMORY_UTILIZATION=0.95`：

```text
VAL_BATCH_SIZE=50
AGENT_WORKERS=50
MAX_NUM_SEQS=64
GPU_MEMORY_UTILIZATION=0.95
MAX_RESPONSE_LENGTH=8192
```

结果：

| 指标 | C2 `vb64/w64/s64/mem0.88` | C7 `vb50/w50/s64/mem0.95` |
| --- | ---: | ---: |
| wall time | 12m33s | 15m52s |
| validation elapsed | 668.7s | 867.1s |
| assistant tok/s | 1304.7 | 1024.8 |
| score mean | 0.1878 | 0.2037 |
| acc | 24/128 | 26/128 |
| response cap hit | 100/128 | 98/128 |
| tool wall time | 448.9s | 412.3s |

C7 稳定完成，但因为 128 条被切成 `50 + 50 + 28` 三个 batch，最后一批吞吐只有 `668.9 assistant tok/s`，整体明显慢于 C2。`mem0.95` 没有证明能带来收益，反而增加稳定性风险。

当前默认评测配置定为：

```text
MAX_RESPONSE_LENGTH=8192
VAL_BATCH_SIZE=64
AGENT_WORKERS=64
MAX_NUM_SEQS=64
GPU_MEMORY_UTILIZATION=0.88
VAL_TEMPERATURE=0
VAL_DO_SAMPLE=false
```

这组参数已经写入 `scripts/evaluate_baseline_with_verl.sh` 默认值。脚本不新增 `threads1` 类限制；如果后续再次探索更激进并发或 `mem0.95`，再由命令行显式传入线程限制做单独实验。

## 正式 full eval 结果

两次正式评测都使用当前默认参数：

```text
MAX_RESPONSE_LENGTH=8192
VAL_BATCH_SIZE=64
AGENT_WORKERS=64
MAX_NUM_SEQS=64
GPU_MEMORY_UTILIZATION=0.88
VAL_TEMPERATURE=0
VAL_DO_SAMPLE=false
ENABLE_THINKING=true
```

模型 checkpoint：

```text
/root/autodl-tmp/code-agent/outputs/verl_sft/qwen3_8b_oj_sft_20260505_032710/global_step_234/huggingface
```

### CodeContests full eval

run：`outputs/verl_baseline_eval/full_codecontests_test_mr8192_vb64_w64_s64_mem088_greedy`

时间：

| 指标 | 数值 |
| --- | ---: |
| start | 2026-05-08 04:01:40 |
| validation_start | 2026-05-08 04:03:03 |
| end | 2026-05-08 04:46:42 |
| wall time | 45m02s |
| validation elapsed | 2605.7s / 43m25.7s |

结果：

| 指标 | 数值 |
| --- | ---: |
| samples | 499 |
| score mean | 0.2213 |
| acc | 22.04% |
| assistant tokens | 3,386,657 |
| assistant tokens/problem | 6,786.9 |
| assistant token p50 | 8,093 |
| assistant token p90 | 8,192 |
| assistant tok/s | 1,299.7 |
| response cap hit | 386/499 |
| assistant cap hit | 214/499 |
| tool response tokens/problem | 178.4 |
| tool calls/problem | 3.03 |
| max tool calls | 22 |
| tool wall time | 1,647.0s |
| judge runtime | 1,611.3s |
| num turns mean | 7.96 |

terminal 分布：

| terminal_reason | 数量 |
| --- | ---: |
| `no_tool_call` | 394 |
| `tool_call_limit_exhausted` | 10 |
| `unknown` | 95 |

说明：

- `codecontests_test` 原始 500 条，validation 实际 499 条；日志显示 train stub 过滤后 500 条、validation file 过滤后 499 条。
- `generations/0.jsonl` 和 `generations/partial_0.jsonl` 都是 499 行，已完整落盘。
- 本次 full eval 已满足 2 小时目标，且比 B2/C2 的 128 条外推更可信。

### LiveCodeBench full eval

run：`outputs/verl_baseline_eval/full_livecodebench_test_mr8192_vb64_w64_s64_mem088_greedy`

时间：

| 指标 | 数值 |
| --- | ---: |
| start | 2026-05-08 22:20:10 |
| validation_start | 2026-05-08 22:22:40 |
| end | 2026-05-08 23:15:43 |
| wall time | 55m33s |
| validation elapsed | 3168.5s / 52m48.5s |

结果：

| 指标 | 数值 |
| --- | ---: |
| samples | 611 |
| score mean | 0.3550 |
| acc | 35.19% |
| assistant tokens | 3,879,277 |
| assistant tokens/problem | 6,349.1 |
| assistant token p50 | 8,075 |
| assistant token p90 | 8,192 |
| assistant tok/s | 1,224.3 |
| response cap hit | 443/611 |
| assistant cap hit | 221/611 |
| tool response tokens/problem | 146.6 |
| tool calls/problem | 3.08 |
| max tool calls | 22 |
| tool wall time | 1,354.5s |
| judge runtime | 1,343.0s |
| num turns mean | 8.08 |

terminal 分布：

| terminal_reason | 数量 |
| --- | ---: |
| `no_tool_call` | 509 |
| `tool_call_limit_exhausted` | 14 |
| `unknown` | 88 |

说明：

- `generations/0.jsonl` 和 `generations/partial_0.jsonl` 都是 611 行，已完整落盘。
- 本次 full eval 已满足 2 小时目标；当前评测效率 blocker 可以关闭。

### full eval 结论

- 正式评测耗时已经远低于目标：CodeContests 45m02s，LiveCodeBench 55m33s。
- full eval 吞吐约 `1224-1300 assistant tok/s`，与 C2 固定 128 条的 `1304.7 assistant tok/s` 同量级，说明 C2 参数在完整集上稳定。
- 两个 full run 的 assistant tokens/problem 都在 `6.3k-6.8k`，符合 B2 对 `8192` budget 的预期。
- `response_cap_hit` 仍很高，说明效率达标主要来自硬截断和并发优化；模型行为仍需要通过 RL 奖励和 stop 约束改善。
- 后续大规模 baseline、RL validation 和 rollout 默认复用脚本内参数；只有做质量敏感复核或并发消融时再显式覆盖。


## 估算公式

完整 500 题耗时可用以下公式估算：

```text
estimated_hours = 500 * assistant_tokens_per_problem / assistant_tokens_per_second / 3600
```

500 题 2 小时目标等价于：

```text
assistant_tokens_per_problem <= assistant_tokens_per_second * 7200 / 500
```

不同吞吐下的单题 token 上限：

| assistant tok/s | 2 小时内每题 token 上限 |
| ---: | ---: |
| 300 | 4,320 |
| 500 | 7,200 |
| 800 | 11,520 |
| 1000 | 14,400 |

历史 smoke 平均约 `21,583 assistant tokens/problem`。正式 full eval 已经通过 `8192` response budget 和 C2 并发参数把 assistant tokens/problem 降到 `6.3k-6.8k`，并把吞吐提升到 `1224-1300 assistant tok/s`，因此完整评测 2 小时目标已经达成。
