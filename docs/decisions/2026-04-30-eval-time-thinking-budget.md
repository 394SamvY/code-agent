# 决策：为 baseline eval 增加 eval-time thinking budget 控制

日期：2026-04-30

## 决策

baseline eval 默认启用两层 eval-time 控制，缓解 Qwen3 在 thinking mode 下长时间停留在未闭合 `<think>`、耗尽整条 trajectory `MAX_RESPONSE_LENGTH=8192` 的问题：

- soft control：`CODE_AGENT_PROMPT_STYLE=short_thinking`
- hard control：`CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET=3072` 与 `CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET=2048`

这两个控制只服务 baseline eval 的吞吐和行为稳定性，不改变 OJ-like 环境协议、reward 规则、工具 schema、`max_submissions=5` 语义，也不关闭 Qwen3 的 `enable_thinking=true`。

当前主评测入口 `scripts/evaluate_baseline_with_verl.sh` 已默认开启：

```bash
CODE_AGENT_PROMPT_STYLE=short_thinking
CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET=3072
CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET=2048
MAX_RESPONSE_LENGTH=8192
```

## 背景

本项目当前 baseline eval 复用 verl `main_ppo` validation 路径，模型在同一条 trajectory 中可以多轮调用：

- `run_public_tests`
- `submit_solution`

`MAX_RESPONSE_LENGTH=8192` 在 verl agent loop 中表示整条 multi-turn trajectory 的总 response token budget，而不是单轮 assistant 的输出上限。这个总预算需要保留，因为一个正常 OJ-like agent 可能需要：

1. 第一轮写代码并跑 public tests。
2. 根据 observation 修复代码。
3. 再跑 public tests 或直接 submit。
4. 根据 failed submit 的首错反馈继续修复。

问题在于 Qwen3-8B base model 开启 `enable_thinking=true` 后，很多样本会在第一轮 assistant 输出中长时间推导，甚至一直不闭合 `<think>`，也不发出任何 OJ tool call。这样会出现几个直接后果：

- 单个失败样本吃满 8192 response tokens。
- 没有产生 `run_public_tests` / `submit_solution`，因此没有有效 OJ 交互。
- GPU KV cache 和 wall-clock 被低价值长思考消耗。
- batch 内其他样本被慢样本拖住，500 条 eval 成本明显上升。

早期 32-sample smoke 中，这类无工具长思考占比很高：

- `num_tool_calls = 0` 达到约 78.1%。
- `unclosed_think_rate` 达到约 78.1%。
- 平均输出字符数约 28,192。
- accepted rate 约 18.8%。

因此需要限制“单轮 assistant generation”的最大长度，但不能把整条 trajectory 的 `MAX_RESPONSE_LENGTH` 直接改小。

## 设计目标

本控制的目标是：

- 保留整条 trajectory 的 8192 token 总预算，允许真正发生多轮调试。
- 防止第一轮或某一轮 assistant 长思考独占全部 8192。
- 尽早让无工具、无提交的失败样本结束，节省 eval 时间。
- 提高模型实际调用 OJ tool 的比例。
- 不改 parquet，不重新导出数据。
- 不改 reward，不给 public tests reward。
- 不把 eval 工程保护写成 OJ 任务规则。
- 不依赖解析 `<think>` 内容，避免和模型模板、tool parser 强耦合。

非目标：

- 不试图精确控制 `<think>` 内 token 数。
- 不保证每个样本都会产生 tool call。
- 不解决 base model 代码修复能力不足的问题。
- 不把该策略直接定义为训练环境语义；训练是否使用需要单独决策。

## 做法一：short-thinking prompt 软约束

实现位置：

- `src/verl_dataset_adapter.py`

开关：

```bash
CODE_AGENT_PROMPT_STYLE=short_thinking
```

实现方式：

`OJLikeRLHFDataset` 在读取 parquet 时，本来就需要把 JSON string 格式的 `prompt` decode 成 Python chat messages。short-thinking 控制复用这个 adapter，在 decode 后动态修改第一条 system message：

1. 检查 `CODE_AGENT_PROMPT_STYLE`。
2. 只有值为 `short_thinking`、`brief_thinking` 或 `tool_first` 时才启用。
3. 确认 prompt 是 list，且第一条 message 是 `role=system`。
4. 在原 system prompt 末尾追加 eval-time 约束。
5. 不写回 parquet，不修改磁盘数据。

追加的约束表达的是：

- private reasoning 要简短、面向行动。
- 第一轮 assistant 最多用几行 concise thinking。
- 尽快闭合 thinking block。
- 调用 `run_public_tests` 或 `submit_solution`。
- 不要在 OJ tool call 前把 response budget 花在大量 case-by-case derivation 上。

这是 soft control，因为它只是在 prompt 层引导模型。模型仍然可能不遵守，所以它必须和 hard control 配合。

选择在 dataset adapter 做，而不是重新导出 parquet，原因是：

- 当前四个标准 parquet 是 v1 baseline 输入，不应为了 eval 调参频繁重写。
- prompt style 属于运行时评测策略，不属于数据 schema。
- 同一份 parquet 可以用不同 prompt style 做可比实验。
- 可以通过环境变量关闭或切换，不影响训练数据源。

## 做法二：per-assistant-turn token budget 硬约束

实现位置：

- `src/verl_agent_loop.py`
- `configs/verl/code_agent_loop.yaml`

开关：

```bash
CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET=3072
CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET=2048
```

verl 配置入口：

```text
actor_rollout_ref.rollout.agent.default_agent_loop=code_agent_tool_agent
actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/verl/code_agent_loop.yaml
```

`CodeAgentToolAgentLoop` 继承 verl 原生 `ToolAgentLoop`。它运行在真正执行样本的 `AgentLoopWorker` Ray actor 内，而不是 TaskRunner actor 内。2026-04-30 的 500 条 debug run 已证明：只在 `CodeAgentTaskRunner.run()` 里 monkey patch `ToolAgentLoop`，不会自动影响独立的 `AgentLoopWorker` 进程。

该子类覆盖两个方法：

- `_handle_generating_state()`：在每轮 assistant generation 前修改 `sampling_params["max_new_tokens"]`。
- `_handle_processing_tools_state()`：父类完成 tool execution 后，如果 OJ tool 已设置 `agent_data.code_agent_terminal=True`，立刻返回 `AgentState.TERMINATED`。

原始 generation 流程大致是：

1. 根据当前 prompt/tool observation 调用 rollout backend 生成 assistant tokens。
2. 把生成 token 追加到当前 trajectory。
3. 用 tool parser 从本轮 assistant 输出中解析 tool call。
4. 如果解析到 tool call，进入 tool execution。
5. 如果没有 tool call，当前 trajectory 结束。

本项目只在第 1 步之前修改 `sampling_params`，不改后面的 tool parser、tool execution、reward 或 OJ 工具语义。

核心逻辑：

```python
if agent_data.assistant_turns == 0:
    turn_budget = int(os.getenv("CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET"))
else:
    turn_budget = int(os.getenv("CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET"))
remaining_budget = self.response_length - len(agent_data.response_mask)
max_new_tokens = min(turn_budget, remaining_budget)
sampling_params["max_new_tokens"] = max_new_tokens
```

含义：

- `self.response_length` 是整条 trajectory 的总 response budget，当前 eval 默认是 8192。
- `agent_data.response_mask` 记录这条 trajectory 到目前为止的 response 区域 token。
- `remaining_budget` 表示这条 trajectory 还剩多少 response token 可用。
- `turn_budget` 是单次 assistant generation call 的上限；第一轮默认 3072，后续轮默认 2048。
- 实际传给 backend 的 `max_new_tokens` 取两者较小值。

这样做的效果是：

- 第一轮最多生成 3072 tokens。
- 如果第一轮成功发出 tool call，后续 observation 之后还能继续生成。
- 第二轮及之后每轮最多生成 2048 tokens，但整条 trajectory 总量仍不超过 8192。
- 如果前面已经用掉较多预算，后续单轮上限会自动被 `remaining_budget` 截到更小。

这和直接设置 `MAX_RESPONSE_LENGTH=3072` 不同。直接缩小 `MAX_RESPONSE_LENGTH` 会让整条 trajectory 只能用 3072，正常多轮 repair 也会被压死；per-turn budget 只限制某一轮不要独占总预算。

## sampling 参数兼容

不同 verl / rollout backend 路径可能使用不同字段名：

- `max_new_tokens`
- `max_tokens`

patch 会同时检查这两个 key：

1. 如果上游已经传了更小的值，尊重上游更小值。
2. 移除旧的 `max_tokens` / `max_new_tokens`。
3. 最后统一写入 `max_new_tokens`。

这样避免同一个请求同时带两个近义参数，导致 SGLang / vLLM backend 对优先级解释不一致。

## 为什么不解析 `<think>`

本决策故意不做 `<think>` 级别截断，原因是：

- verl tool parser 负责解析 tool call，本项目不应再实现一套模型模板解析器。
- `<think>` token 格式依赖模型模板，未来换模型或 tokenizer 时容易失效。
- 截断 `<think>` 内部文本后强行拼 tool call，会改变模型真实行为，不适合 baseline eval。
- 当前目标是节省无效长输出，而不是构造一个更会调用工具的 synthetic assistant。

因此 hard control 只限制单次 generate 的长度。如果模型在 first-turn 3072 tokens 内没有发出 tool call，verl 原始逻辑会把该 trajectory 作为无工具失败样本结束；后续修复轮默认限制为 2048 tokens。

## 为什么不关闭 thinking

不直接关闭 `enable_thinking`，原因是：

- Qwen3 的 thinking 模式可能对复杂竞赛题有帮助。
- 当前目标不是把模型变成 one-shot 代码生成器，而是 OJ-like multi-turn agent。
- 完全关闭 thinking 会引入另一套 prompt / 模型行为分布，和后续 RL 训练目标不一定一致。

当前策略是保留 thinking，但要求它短、面向行动，并用 per-turn budget 防止失控。

## 为什么不直接缩小 MAX_RESPONSE_LENGTH

`MAX_RESPONSE_LENGTH` 是整条 trajectory 的预算。把它从 8192 降到 3072 会同时伤害：

- 第一轮代码生成。
- public test 后的修复。
- failed submit 后的修复。
- 多轮 tool call 的 observation 后续回答。

OJ-like agent 的关键能力是交互式修复，所以总预算应保留给真正发生工具调用的样本。当前问题是“单轮长思考独占总预算”，因此控制点应放在 assistant turn，而不是 whole trajectory。

## 为什么不把它放进环境协议

thinking budget 是 eval 运行时策略，不是 OJ 规则。环境协议仍然只关心：

- `run_public_tests`
- `submit_solution`
- `max_submissions=5`
- public tests 不给 reward
- submit 是主 reward 来源

单轮 assistant 生成上限属于 rollout/backend 工程保护，类似 `max_tool_calls`，不应写成题目规则，也不应影响 reward 解释。

## 运行默认值

`scripts/evaluate_baseline_with_verl.sh` 当前默认面向 2xA800-80GB：

```bash
CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
VAL_MAX_SAMPLES=500
VAL_BATCH_SIZE=16
AGENT_WORKERS=16
MAX_NUM_SEQS=32
GPU_MEMORY_UTILIZATION=0.82
MAX_NUM_BATCHED_TOKENS=32768
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=8192
CODE_AGENT_PROMPT_STYLE=short_thinking
CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET=3072
CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET=2048
```

常用命令：

```bash
bash scripts/evaluate_baseline_with_verl.sh codecontests_test.parquet
```

指定 run name：

```bash
RUN_NAME=codecontests_test_500_shortthink_f3072_u2048 \
  bash scripts/evaluate_baseline_with_verl.sh codecontests_test.parquet
```

关闭 per-turn budget：

```bash
CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET=0 \
CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET=0 \
  bash scripts/evaluate_baseline_with_verl.sh codecontests_test.parquet
```

关闭 short-thinking prompt：

```bash
CODE_AGENT_PROMPT_STYLE= \
  bash scripts/evaluate_baseline_with_verl.sh codecontests_test.parquet
```

做 32 条 focused probe：

```bash
bash scripts/evaluate_2xa800_32_debug.sh codecontests_test
```

## 当前验证结果

2026-04-30 的早期 32-sample focused eval 对比显示，short-thinking prompt 明显缓解“第一轮长思考不调用工具”的问题。但后续 499 条 debug run 暴露出当时的 hard budget / terminal stop monkey patch 没有进入 `AgentLoopWorker`，因此这些结果只能作为调试对照，不作为正式 baseline。

旧 67GB 档位 smoke：

| 指标 | 数值 |
| --- | ---: |
| accepted rate | 18.8% |
| `unclosed_think_rate` | 78.1% |
| `num_tool_calls = 0` | 78.1% |
| submit sample rate | 21.9% |
| 平均输出长度 | 28,192 字符 |

short-thinking + 3072 per-turn budget，67GB 档位：

| 指标 | 数值 |
| --- | ---: |
| accepted rate | 31.2% |
| `unclosed_think_rate` | 50.0% |
| `num_tool_calls = 0` | 31.2% |
| submit sample rate | 68.8% |
| 平均输出长度 | 18,220 字符 |
| GPU max memory | 65.8GB / 66.0GB |
| wall-clock | 426s，约 13.3s/题 |

499 条 debug run：

| 指标 | 数值 |
| --- | ---: |
| 样本数 | 499 |
| score mean | 0.1940 |
| accepted rate | 19.2% |
| `num_tool_calls = 0` | 26.3% |
| 平均 tool calls | 13.92 |
| 最大 tool calls | 131 |
| 最大 turns | 264 |
| no-tool 样本生成到 8192 tokens | 74 条 |

结论：soft prompt 生效，但旧 hard patch 未生效。修复后的正式验证必须先看 no-tool 样本最大 token 是否接近 first-turn budget 3072，以及 `submission_limit_exceeded` 后是否停止 tool loop。

77GB 高显存 probe 使用更高并发和 `GPU_MEMORY_UTILIZATION=0.94`，能够把显存推到约 75-76GB，但 32 条下 wall-clock 变慢到约 15.2s/题，accepted rate 也略低。因此当前默认保留 67GB 档位：

```bash
VAL_BATCH_SIZE=16
AGENT_WORKERS=16
MAX_NUM_SEQS=32
GPU_MEMORY_UTILIZATION=0.82
MAX_NUM_BATCHED_TOKENS=32768
```

## 被拒绝的方案

- **直接把 `MAX_RESPONSE_LENGTH` 从 8192 降到 3072**：会限制整条 trajectory，损害多轮 repair。
- **关闭 `enable_thinking`**：会改变 Qwen3 的主要行为模式，不利于评估 thinking + tool use 的 baseline。
- **只靠 prompt 要求短思考**：模型可能不遵守，无法保证失败样本不会继续吃满 8192。
- **只靠 hard cap，不加 prompt**：能节省 token，但不一定提高模型主动调用 OJ tool 的意愿。
- **解析并截断 `<think>`**：和模型模板强耦合，容易破坏 tool parser 语义。
- **在 tool 层强制补一个 submit 或 run_public_tests**：会伪造模型行为，污染 baseline eval。
- **把 public tests 加 reward 来诱导工具调用**：违反当前 OJ-like reward 设计，public tests 只应提供 observation。

## 后果

- baseline eval 的默认指标应标注为 `short_thinking + first/followup budget 3072/2048`，避免和未加控制或旧无效 hard-patch run 混淆。
- 后续比较不同模型或 checkpoint 时，应固定这两个 eval-time 控制，或明确记录关闭/改值。
- 该控制会降低无工具长思考样本的 wall-clock 成本，但不会解决模型代码能力和 tool-call 格式能力本身。
- 如果未来训练也要启用类似 per-turn budget，需要另写训练决策；当前本文只确认 baseline eval 默认。
- 如果未来换模型，first/followup budget 应重新 probe，3072/2048 不是跨模型常数。

## 相关文档和代码

- `docs/project_status.md`
- `docs/operations/gpu_eval_tuning.md`
- `scripts/evaluate_baseline_with_verl.sh`
- `scripts/evaluate_2xa800_32_debug.sh`
- `scripts/analyze_eval_generations.py`
- `src/verl_dataset_adapter.py`
- `src/verl_runtime_patch.py`
- `src/verl_agent_loop.py`
- `configs/verl/code_agent_loop.yaml`
- `docs/decisions/2026-04-29-verl-validation-baseline.md`
- `docs/decisions/2026-04-29-oj-like-two-tool-protocol.md`
