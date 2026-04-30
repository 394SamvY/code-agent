# Baseline Eval 环境验收标准

更新日期：2026-05-01

本文定义 OJ-like baseline eval 链路的严格验收标准。只有通过本文的 P0 验收，才能说 eval 环境链路可信；否则不能把模型指标解释为模型能力结论。

本文关注 `scripts/evaluate_baseline_with_verl.sh` 及其复用的 verl validation / agent loop / tool / reward 链路。效率调优属于 P1，必须排在环境正确性之后。

## 验收原则

- P0 未通过前，不宣称 eval 环境已经调通。
- P0 未通过前，不用 accepted rate、score mean、no-tool rate 等指标判断模型能力。
- 验收必须基于真实 verl `AgentLoopWorker` 路径，不接受只在 TaskRunner patch、离线 parser 或本地环境中成立的证据。
- generation dump 必须足够复盘每条 trajectory 的真实交互：模型输出、tool call、tool response、reward、terminal reason 和解析失败。

## P0：环境正确性

### 1. Thinking 控制语义正确

目标不是简单截断输出，而是确保 Qwen3 thinking 结束后仍能继续进入 tool call。

验收要求：

- `enable_thinking=true` 保持开启。
- 模型可以输出 `<think>...</think>`。
- thinking 结束不等于停止整轮生成；模型必须仍有机会继续生成 `<tool_call>`。
- 如果实现 Qwen3 原生 thinking 早停，早停点应只结束 thinking 段，不应结束 assistant turn 或 trajectory。
- 如果模型在 thinking budget 内没有闭合 `<think>`，该样本可以被判为 no-action failure 或进入明确的 fallback 策略，但不能伪装成正常 tool trajectory。
- 当前 per-assistant-turn token budget 只能作为 hard cap，不能被当作完整 thinking 早停语义。

必须检查的证据：

- 至少抽查 3 条包含 `<think>...</think>` 后继续 tool call 的 trajectory。
- no-tool 样本的最大输出 token 不应吃满整条 `MAX_RESPONSE_LENGTH=8192`，除非明确关闭了 per-turn hard cap。
- generation dump 中不能出现大量“停止 thinking 后直接结束、没有 tool call，但被误计为正常完成”的样本。

### 2. Tool call 解析和执行一致

验收要求：

- 合法 `<tool_call>{...}</tool_call>` 会被执行。
- malformed JSON 或缺少 tool name / arguments 的 tool call 不执行工具，并记录为解析失败。
- 同一 assistant turn 中 `<tool_call>` 后的尾部文本不能影响 tool state；应被裁剪，或至少在 analysis 中单独统计为尾部浪费。
- `messages` 字段必须和真实 tool execution 对齐，不能只靠离线 parser 构造出看似合理但实际未执行的交互。

必须检查的证据：

- 抽查 tool call 的 `name`、`arguments.code`、tool response 与 `num_tool_calls` 一致。
- malformed tool call 数量单独统计。
- accepted 后额外文本长度单独统计，不混入新的有效行动。

### 3. 两个 OJ tool 的计数限制正确

验收要求：

- `run_public_tests` 默认最多 15 次。
- `run_public_tests` 达到上限后返回 `public_test_limit_exceeded`。
- public test 上限不消耗 `submit_solution` 次数，不终止 episode。
- `submit_solution` 默认最多 5 次。
- accepted 后 trajectory terminal。
- submit 次数耗尽后 trajectory terminal。
- verl 每次 tool call 即使走 `create -> execute -> release`，同一 trajectory 内 public/submission 计数也必须累计。

必须检查的证据：

- 单测覆盖 tool state 跨 `create -> execute -> release` 持久化。
- 真实 eval 输出中不存在 50/100+ 级别的 tool-call 循环。
- `submission_limit_exceeded` 后没有后续有效 tool call。

### 4. Terminal 语义进入真实 AgentLoopWorker

验收要求：

- accepted 后不再继续调用工具。
- `submission_limit_exceeded` 后不再继续调用工具。
- terminal stop 必须在 `CodeAgentToolAgentLoop` 或真实 AgentLoopWorker 进程内生效。
- 不接受只在 TaskRunner monkey patch 中生效的 terminal 证据。

必须检查的证据：

- `code_agent_terminal` metric 或等价 terminal reason 能在真实 eval output / metrics 中追踪。
- `max_tool_calls` 和 `num_turns max` 处于合理范围，不复现旧 run 的 100+ tool calls / 200+ turns。

### 5. Reward 与 observation 一致

验收要求：

- `run_public_tests` reward 永远是 `0.0`。
- `submit_solution` accepted reward 是 `1.0`。
- failed submit 的 shaped reward 只来自 `submit_solution`，不来自 public tests。
- public/private judge 都遇到首个失败 case 即停。
- observation text 和结构化 result 的 verdict、passed、total、first_failed 一致。
- rollout 结束时如果没有任何 valid submit，该 episode 视为失败。

必须检查的证据：

- 单测覆盖 public reward、accepted reward、failed submit shaped reward。
- 抽查 failed public / failed submit 的 observation 与结构化 result 一致。

### 6. 数据和 tool create kwargs 完整

验收要求：

- 每条 eval 样本都带 public tests 给 `run_public_tests`。
- 每条 eval 样本都带 private/full tests 给 `submit_solution`。
- 每条 eval 样本带题目级 time limit。
- 每条 eval 样本带 `max_submissions=5`。
- 每条 eval 样本带 `max_public_test_calls=15`。
- `task_id` / reward ground truth 可追踪。

必须检查的证据：

- Parquet schema / `extra_info.tools_kwargs` 抽样检查。
- `data/verl` 旧缓存不作为规范来源；必要时在训练服务器重新生成。

### 7. Prompt 不被 eval 脚本动态改写

验收要求：

- eval 不使用 `CODE_AGENT_PROMPT_STYLE` 或同类 soft prompt hook。
- dataset adapter 只 decode JSON-string parquet 字段，不动态追加 system prompt。
- thinking 长度控制只来自 agent loop / generation control，不来自 system prompt trick。

必须检查的证据：

- `rg` 检查当前运行时入口中不存在 `CODE_AGENT_PROMPT_STYLE`、`_SHORT_THINKING_DIRECTIVE`、`_apply_short_thinking_prompt`。
- 单测覆盖：即使环境里设置旧 `CODE_AGENT_PROMPT_STYLE=short_thinking`，adapter 也不会修改 decoded prompt。

### 8. Batch shape 和 prompt filtering 稳定

验收要求：

- `filter_overlong_prompts=true` 时按真实 chat template token 长度过滤。
- 过滤阶段必须 decode JSON-string prompt 后再应用 tokenizer chat template。
- `VAL_BATCH_SIZE > 1` 完整跑通。
- 不再出现 `DataProto.concat` shape mismatch。
- `partial_0.jsonl` 和最终 `0.jsonl` 行数一致，除非明确手动中断或崩溃。

必须检查的证据：

- 单测覆盖 JSON-string prompt decode 后再过滤。
- 32-sample strict smoke 完整写出 final `generations/0.jsonl`。

### 9. Generation dump 可审计

每条 generation 至少必须能复盘：

- `task_id`
- `score` / `acc` / `reward`
- `num_tool_calls`
- 标准 `messages`
- 每次 tool call 的 `name` 和 `arguments.code`
- 每次 tool response 的 observation
- terminal reason
- parse failure 计数或可推导证据

验收要求：

- 在线落盘和离线分析使用同一套 `messages` 语义。
- 不能只保存不可读的 decoded output，而缺少结构化消息。

## P0 Smoke 验收流程

先跑 32-sample strict smoke：

```bash
VAL_MAX_SAMPLES=32 bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

通过标准：

- final `generations/0.jsonl` 完整写出 32 条。
- 无 OOM、无 shape mismatch、无 Ray worker 异常。
- no-tool 样本最大 output tokens 接近 first-turn hard cap，而不是 8192。
- `max_tool_calls` 不出现 50/100+ 级别异常。
- accepted 后无后续有效 tool call。
- `submission_limit_exceeded` 后无后续有效 tool call。
- 至少人工抽查 3 条多轮 trajectory，确认 `messages` 与真实 tool execution 对齐。
- 至少人工抽查 3 条 failed public / failed submit，确认 observation 与结构化 result 一致。

32-sample 通过后，再跑 500-sample eval：

```bash
VAL_MAX_SAMPLES=500 bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

500-sample 通过同一组 P0 标准后，才可以把该 eval run 当作可信 baseline 输入。

## P1：2xA800 效率验收

P1 只在 P0 全部通过后进行。

目标：

- 单个 verl 进程统一调度两张 A800 80GB。
- 两张卡都有稳定 GPU memory 和 active utilization。
- 不被少数长 thinking / no-tool 样本拖死 batch。
- 在不破坏 P0 语义的前提下调 `VAL_BATCH_SIZE`、`AGENT_WORKERS`、`MAX_NUM_SEQS`、`GPU_MEMORY_UTILIZATION`、`MAX_NUM_BATCHED_TOKENS`。

必须记录：

- GPU monitor CSV 或等价 `nvidia-smi` 采样。
- seconds/sample。
- GPU max memory / avg active util。
- accepted rate、no-tool rate、avg tool calls、max tool calls、avg output length。
- 是否完整写出 final `0.jsonl`。

当前优先级：

1. P0 环境正确性。
2. 32-sample strict smoke。
3. 500-sample可信 baseline。
4. 2xA800 效率调优。
5. `livecodebench_test` 最终泛化评测。
