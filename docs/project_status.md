# 项目状态

更新日期：2026-05-07

## 项目背景

- 项目目标：训练 OJ-like code agent，解完整 stdin/stdout 竞赛编程题。
- 训练数据：`CodeContests`。
- 最终测试：`LiveCodeBench`。
- 环境动作：`run_public_tests` 和 `submit_solution`。
- 评测入口：`scripts/evaluate_baseline_with_verl.sh`，复用 verl 原生 validation / `ToolAgentLoop` 路径。
- RL链路待讨论。

## 当前状态

OJ-like v1 的数据、tool、reward 和 verl validation 评测链路已经接通；当前主要 blocker 是评测效率太低，因此下一步应先把 ./scripts/evaluate_baseline_with_verl.sh 的评测链路效率提上去，要求完整评测 code-agent/data/verl/codecontests_test.parquet 和 code-agent/data/verl/livecodebench_test.parquet 各不超过两个小时！

TODO
1. 提升评测脚本的效率
2. 当前使用基础模型应该也能产生tool_call,所以sft给模型带来的提升有待量化，比如acc和平均tool_call的轮次等，待优化完成评测脚本后执行。

评测效率的第一轮对比实验计划见 [`docs/eval_efficiency_experiment_plan.md`](eval_efficiency_experiment_plan.md)。


## 当前评测链路效率分析

最近的有效评测测试在目录 `outputs/verl_baseline_eval/`下。当前保留了三次 16 样本 smoke：

| run | start | 耗时 | 样本数 | score/acc | 平均 tool calls | 平均 turns | 500 题外推 | 611 题外推 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `smoke_sft_qwen3_8b_toolagent_think_2gpu_16_fast1` | 2026-05-05 23:07:51 | 18m37s | 16 | 0.0% | 4.375 | 10.25 | 9.7h | 11.8h |
| `smoke_sft_qwen3_8b_toolagent_think_2gpu_16_test2` | 2026-05-05 23:40:30 | 22m38s | 16 | 0.0% acc / 0.825% score | 5.8125 | 13.375 | 11.8h | 14.4h |
| `smoke_sft_qwen3_8b_toolagent_think_2gpu_16_test3` | 2026-05-06 00:15:18 | 20m31s | 16 | 31.25% | 3.4375 | 8.75 | 10.7h | 13.1h |

这三次 smoke 的共同配置：

- dataset：`codecontests_test`
- model：`/root/autodl-tmp/code-agent/outputs/verl_sft/qwen3_8b_oj_sft_20260505_032710/global_step_234/huggingface`
- GPU：2 卡，`CUDA_VISIBLE_DEVICES=0,1`
- rollout backend：SGLang
- `VAL_MAX_SAMPLES=16`
- `MAX_PROMPT_LENGTH=4096`
- `MAX_RESPONSE_LENGTH=28672`
- `MAX_MODEL_LEN=32768`
- `VAL_TEMPERATURE=0`
- `VAL_DO_SAMPLE=false`
- `ENABLE_THINKING=true`
- `VAL_BATCH_SIZE=32`
- `AGENT_WORKERS=32`
- `MAX_NUM_SEQS=48`
- `MAX_NUM_BATCHED_TOKENS=49152`
- `GPU_MEMORY_UTILIZATION=0.88`

关键结论：

- 当前评测链路能完整跑完，`generations/partial_0.jsonl` 和 `generations/0.jsonl` 都能正常写出。
- 模型已经能产生 tool call，旧的“tool-call rate = 0%”结论是因为评测脚本中没有显式的配置`actor_rollout_ref.rollout.agent.default_agent_loop=code_agent_tool_agent ` 导致错误的使用了 verl 默认的single_turn_agent，所以没有一个工具调用。
- 评测效率远达不到目标：按最新 run 外推，500 条 CodeContests 约 10.7 小时，611 条 LiveCodeBench 约 13.1 小时。
- 当前首要 blocker 是吞吐和单条 trajectory 过长，而不是 eval 链路能不能跑通。
