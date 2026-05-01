# Thinking Budget 弯路总结

日期：2026-05-01

## 背景

2026-04-29 的 smoke 暴露了一个问题：Qwen3-8B 开启 `enable_thinking=true` 后，
`unclosed_think_rate` 高达 75%，大量样本把 8192 token 的 response budget 全花在
`<\think>` 推导上，从未调用 `run_public_tests` 或 `submit_solution`。一旦模型闭合
thinking 并进入工具协议，成功率并不低（出现 tool tag 的样本 ~80% accepted）。

于是我们开始了约两天、十几轮 A800 调试的"控制 thinking 长度"之路。

## 尝试

### Phase 1：采样参数（失败）

把 greedy（temperature=0）改成 sampling（temperature=0.6, top_p=0.95, top_k=20），
期望随机采样能打破 Qwen3 的重复推导循环。

**结果**：`unclosed_think_rate` 从 73.9% 变成 65-78%，没有方向性改善。thinking 长度
与采样策略无关——Qwen3-8B 启用 thinking 后就会想满 token 预算。

### Phase 2：Soft prompt control（已废弃）

在 system prompt 末尾追加 `short_thinking` 指令，要求模型"短思考、尽早调用 OJ tool"。
实现在 `src/verl_dataset_adapter.py`，由 `CODE_AGENT_PROMPT_STYLE=short_thinking` 控制。

**结果**：2026-05-01 废弃。原因是：(1) 修改 prompt 不够规范，属于 hack；(2) 只靠
自然语言指令无法保证模型不超预算。

### Phase 3：Per-turn token budget（当前仍在用）

通过 `src/verl_agent_loop.py` 的 `CodeAgentToolAgentLoop` 继承 `ToolAgentLoop`，
在每个 assistant generation turn 注入 `max_new_tokens` 上限：首轮 3072，后续轮 2048。
环境变量 `CODE_AGENT_FIRST/FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET` 控制。

**结果**：`unclosed_think_rate` 从 75% 降到~40%，`accepted` 从 20.1% 提升到 34.4%。
但代价是 `verl_agent_loop.py` 膨胀到 500+ 行。

### Phase 4：Two-phase generation（当前仍在用）

在 Phase 3 的基础上进一步拆分：每个 assistant turn 拆成 thinking 阶段
（`max_new_tokens=1024`）和 action 阶段（剩余预算）。如果 thinking 阶段耗尽但
`</think>` 未出现，注入 `_EARLY_STOPPING_TEXT` 强制闭合，然后继续生成 tool call。
由 `CODE_AGENT_ENABLE_THINKING_EARLY_STOP=1` 和 `CODE_AGENT_THINKING_TOKEN_BUDGET=1024`
控制。核心实现在 `_handle_generating_state_with_thinking_stop`（~200 行）。

**结果**：`unclosed_rate` 归零，但实现复杂——需要直接操作 token 序列、检测
`<\im_end>` 和 `</think>` 的 token ID、手动注入控制文本。

### Phase 5：SGLang 原生 thinking_budget（尝试后放弃）

调研发现 SGLang 已内置 `Qwen3ThinkingBudgetLogitProcessor`，支持通过 `custom_params.
thinking_budget` 原生控制 `<think>` 段长度。做了 32 条 retry 验证。

**结果**：
- 首轮有效：首轮闭合 thinking 约 1025-1026 tokens，符合预期
- **多轮失效**：verl 的 multi-turn 路径中，上一轮 `response_ids`（含 `</think>`）会被
  原样拼进下一轮 prompt。SGLang 原生 processor 看到历史 prompt 中已有 `</think>`，
  认为 thinking 已结束，跳过后续轮的控制
- 早期崩溃排查：传 `stop=["</think>"]` 到 SGLang token-in/token-out 路径时，
  scheduler 没有 tokenizer，走 string-stop decode 分支触发
  `AttributeError: 'NoneType' object has no attribute 'decode'`

**结论**：SGLang 原生 thinking_budget 在当前 verl multi-turn agent 场景下不可用。
短期继续用 two-pass 方案，但不作为长期方向。

## 反思

### 错在哪里

**根本问题是模型能力，不是 thinking 长度。** Qwen3-8B 作为 base model，未经 RL 训练，
代码 debug 能力差，所以在 thinking 阶段反复推导边界条件、验证逻辑，迟迟不收敛。
thinking 长短是表象，不会 debug 才是根因。

我们在评测侧加的各种 budget 控制，本质上是在**修补一个应该由训练解决的问题**。
这导致了三个后果：

1. **eval 环境不对等**：base model 在 eval 时被掐断 thinking，但训练时（GRPO rollout）
   模型不受这些约束。base eval 和 trained eval 的测试条件不一致，评测结果不可比。

2. **代码膨胀**：为了控制一个本应由模型自己收敛的行为，写了 200 行两阶段生成 + 
   4 个环境变量 + trace 记录。这些代码跟评测正确性无关。

3. **分散注意力**：原本目标是"证明 eval 环境可信"，但两周的调试重心全在 thinking
   budget，而不是验证 eval 语义正确性（tool call 不漏、terminal 正确触发、
   reward 口径一致等）。

### 正确做法

1. **提 `MAX_RESPONSE_LENGTH`**：给模型足够的 token 预算，不要人为限制
2. **去掉所有 thinking budget**：删除 Phase 3-4 的实现，保留最简单的 per-turn 
   `max_new_tokens`（仅作为安全上限，不拆分 thinking/action）
3. **确保 eval 环境正确**：专注 P0 验收（tool call 不丢失、terminal 正确、
   reward 一致），跑通 500 条 baseline
4. **交给训练**：模型 over-think 不调工具的问题，通过 RL 训练让模型学会
   在合适的时机结束思考并采取行动

### 可以保留的

- `CodeAgentToolAgentLoop` 的 terminal 检测（accepted 即停，6 行）
- `_call_tool` 的 parse error 处理（防止 malformed tool call 静默失败）
- per-turn `max_new_tokens` 注入（作为安全上限，防止单轮吃光全部 budget，但不要拆
  thinking/action）

### 下一步

1. 清理 `verl_agent_loop.py`：删两阶段生成、`_EARLY_STOPPING_TEXT`、thinking budget
   相关函数和常量
2. 清理 `evaluate_baseline_with_verl.sh`：删除 `CODE_AGENT_*` 环境变量
3. 提高 `MAX_RESPONSE_LENGTH` 到 16384
4. 跑 500 条 baseline，确认环境正确性
