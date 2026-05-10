# 项目状态

更新日期：2026-05-09

## 项目背景

- 项目目标：训练 OJ-like code agent，解完整 stdin/stdout 竞赛编程题。
- 训练数据：`CodeContests`。
- 最终测试：`LiveCodeBench`。
- 环境动作：`run_public_tests` 和 `submit_solution`。
- 评测入口：`scripts/evaluate_baseline_with_verl.sh`，复用 verl 原生 validation / `ToolAgentLoop` 路径。
- baseline 评测输出统一保存在 `outputs/verl_baseline_eval/`。
- RL链路待讨论

## 当前状态

OJ-like v1 的数据、tool、reward 和 verl validation 评测链路已经接通。此前主要 blocker 是评测效率过低；经过 response budget 和并发参数实验后，当前默认评测参数已经写入 `scripts/evaluate_baseline_with_verl.sh`，两次正式 full eval 都在 2 小时目标内完成。

当前默认评测参数：

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

评测效率实验细节、A/B/C 对比和 full eval 结果见 [`docs/eval_efficiency_experiment_plan.md`](eval_efficiency_experiment_plan.md)。

## 正式评测结果

两次正式评测使用相同默认参数和同一个 SFT checkpoint：

```text
/root/autodl-tmp/code-agent/outputs/verl_sft/qwen3_8b_oj_sft_20260505_032710/global_step_234/huggingface
```

| run | dataset | 样本数 | wall time | validation elapsed | score | acc | assistant tok/s | assistant tokens/problem | 平均 tool calls | 平均 turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_codecontests_test_mr8192_vb64_w64_s64_mem088_greedy` | `codecontests_test` | 499 | 45m02s | 43m25.7s | 0.2213 | 22.04% | 1299.7 | 6786.9 | 3.03 | 7.96 |
| `full_livecodebench_test_mr8192_vb64_w64_s64_mem088_greedy` | `livecodebench_test` | 611 | 55m33s | 52m48.5s | 0.3550 | 35.19% | 1224.3 | 6349.1 | 3.08 | 8.08 |

补充说明：

- `codecontests_test` 原始 500 条中有 1 条因 `MAX_PROMPT_LENGTH=4096` 和 `FILTER_OVERLONG_PROMPTS=true` 被过滤，实际 validation 样本数为 499。
- 两次正式评测的 `generations/0.jsonl` 与 `generations/partial_0.jsonl` 都已完整写出。
- 旧的 16 样本 smoke 外推已不再代表当前效率；当前 full eval 实测比 2 小时目标有充足余量。

## 当前结论

- 评测链路效率 blocker 已解除。完整 CodeContests / LiveCodeBench 评测不再需要按 16 样本 smoke 外推，直接以 full eval 实测为准。
- `MAX_RESPONSE_LENGTH=8192` 是当前默认预算。它显著减少无效长思考，并在固定 128 条对比中保持和 28672 budget 相同的 accepted 数；但 `response_cap_hit` 仍高，说明模型行为还没有真正学会主动短路径。
- `VAL_BATCH_SIZE=64`、`AGENT_WORKERS=64`、`MAX_NUM_SEQS=64`、`GPU_MEMORY_UTILIZATION=0.88` 是当前稳定高吞吐配置。更激进的 `mem0.95` / 高并发尝试没有证明收益，且存在线程资源风险。
- 模型已经能产生 tool call；此前“tool-call rate = 0%”是因为脚本缺少 `actor_rollout_ref.rollout.agent.default_agent_loop=code_agent_tool_agent`，误走了 verl 默认 single-turn agent。
- 当前 reward / `acc` 口径按 `src/reward.py` 里的最后一次 `submit_solution` observation 计算，不是 `max(tool_rewards)`。如果某次 accepted 后只继续 `run_public_tests` 或普通文本、不再发生新的 submit，那么最后一次 submit 仍是 accepted；如果后续又 submit 失败，则以最后一次失败 submit 为准。RL 训练阶段仍需要通过调整reward来约束 accepted 后停止、无工具长思考和撞工具上限。

## 下一步

1. 进入 RL 方案设计：优先围绕 `submit accepted -> 简短收尾 -> 停止`、减少无工具长思考、避免 accepted 后继续工具调用来设计 reward / stop 行为约束。
2. 用当前默认评测参数量化基础模型与 SFT checkpoint 的差异，至少记录 `acc`、`score`、平均 tool calls、平均 turns、terminal_reason 分布和 assistant tokens/problem。
