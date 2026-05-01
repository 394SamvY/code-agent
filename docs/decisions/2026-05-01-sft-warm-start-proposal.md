# 2026-05-01 SFT Warm-Start 提案

状态：提案，尚未采纳。

## 背景

训练路线尚未正式确定（2026-05-01 移除了 RL-only 决策）。本文记录一个可能的方向：如果决定在 RL 前引入小规模行为 warm-start。

近期 Qwen3 thinking-budget 调查发现：

- Qwen3 在调用工具前会生成非常长的推理过程。
- SGLang 内置的 `Qwen3ThinkingBudgetLogitProcessor` 首轮有效，但不适合多轮 tool-agent rollout，因为历史 prompt 中已包含 `</think>` token。
- 现有的两阶段生成 runtime budget 可以控制正确性，但仍然是运行时 workaround。

替代思路是把 base model 的行为分布拉向期望的 agent 行为：短推理、及时调用工具、收到工具反馈后精炼修复。

## 目标

训练一个小规模 SFT 或 LoRA warm-start adapter，教会模型：

- 每次行动前做简短思考；
- 需要时在最终提交前调用 `run_public_tests`；
- 根据工具反馈进行针对性修复；
- 公开测试通过后调用 `submit_solution`；
- 避免在代码注释或工具调用后堆砌冗长自由推理；
- 保持当前 OJ-like 两工具协议。

Warm-start 应减少对复杂 thinking-budget 干预的依赖，但不移除硬性安全上限（`response_length`、per-turn `max_new_tokens`、`max_tool_calls`）。

## 非目标

- 不重新引入 function-benchmark 或 `execute_code` 风格协议。
- 不在最终测试集或 LiveCodeBench 保留数据上训练。
- 除非后续决策明确采纳，不将 SFT 作为主优化目标。
- 不为优美的 chain-of-thought 优化；为稳定的 agent 动作和 judge 反馈使用优化。

## 数据计划

仅使用 CodeContests 训练集题目。

用更强的 teacher 模型在当前环境协议下生成 trajectory：

1. 输入为相同的 OJ 风格 prompt 和 public tests。
2. 可用工具仅为 `run_public_tests` 和 `submit_solution`。
3. 公开测试调用只返回 observation。
4. 完整提交是唯一的 terminal reward 来源。
5. `max_submissions=5` 保持为环境规则。

建议的首批数据规模：

- 试点：200-500 条 trajectory，用于格式和过滤验证。
- 首次 SFT 实验：1k-5k 高质量 trajectory。
- 仅在 CodeContests 验证集行为指标提升后再扩大规模。

## 目标 Trajectory 格式

每轮 assistant turn 应遵循以下形式：

```text
<think>
简短计划或诊断，通常 128-512 token。
</think>

<tool_call>
{"name": "run_public_tests", "arguments": {"code": "..."}}
</tool_call>
```

公开测试失败后：

```text
<think>
简要指出失败行为和具体修复方案。
</think>

<tool_call>
{"name": "run_public_tests", "arguments": {"code": "...修复后代码..."}}
</tool_call>
```

最终接受尝试：

```text
<think>
公开行为一致；提交最终版本。
</think>

<tool_call>
{"name": "submit_solution", "arguments": {"code": "..."}}
</tool_call>
```

## 过滤规则

仅保留满足以下条件的 trajectory：

- 所有 tool call 可解析为 JSON；
- 所有 tool 名称在 `{run_public_tests, submit_solution}` 内；
- 每个 tool call 都有字符串类型 `code` 参数；
- 至少包含一次 `submit_solution`；
- terminal 语义与环境一致；
- 每轮 assistant turn 不超过设定的 per-turn token budget；
- 无未闭合的 thinking span；
- 代码为完整的 stdin/stdout Python 程序；
- trajectory 要么到达 accepted，要么展示了从公开反馈中明确修复的模式。

优先保留 accepted 的 trajectory。仅当失败但仍然有指导意义的 trajectory 展示了有效的工具使用和修复行为时，保留少量受控切片。

## 训练计划

从当前 Qwen3-8B base 开始，使用 LoRA SFT。

建议的首个配置：

- 仅训练 assistant token；
- 上下文中包含 tool response，但在 loss 中 mask 掉；
- 目标中包含 `<think>`、`</think>`、`<tool_call>` 和 `</tool_call>`；
- 仅在验证不会错误合并独立 tool trajectory 后再使用短序列 packing；
- 先训练 1 个 epoch；
- 在评估通过之前，将 adapter 与 RL 路径分离。

如果项目后续采纳此方向，添加独立脚本而非修改主 GRPO 入口。

## 评估关卡

在用于任何 RL 之前，用相同的 verl validation 路径评估 adapter。

主要行为指标：

- tool-call 解析成功率；
- 每轮 assistant turn 平均 thinking token 数；
- thinking span 闭合的 turn 占比；
- 每道题平均 tool call 数；
- 调用 `submit_solution` 的 trajectory 占比；
- 公开失败到修复的比率；
- CodeContests 验证集 accepted rate。

必要的正确性关卡：

- P0 audit 仍然通过。
- `run_public_tests` accepted 永不终止 trajectory。
- 仅 `submit_solution` accepted 或 submission-limit 触发终止。
- 不回退到旧工具或 function-completion 协议。

## 风险

- 如果数据集过窄，SFT 可能过度缩减 RL 所需的探索空间。
- Teacher trajectory 可能泄露脆弱的解题风格或过度拟合公开测试。
- 缩短 thinking 可能改善工具规范但损害难题推理能力。
- 引入 SFT 改变了项目策略，在成为默认路径之前应作为独立决策记录。

## 建议推进步骤

1. 保留当前的 runtime budget 作为 baseline 正确性保障。
2. 针对 CodeContests 训练集构建小型 teacher-trajectory 生成脚本。
3. 生成并审计 200-500 条 trajectory 试点。
4. 训练 LoRA SFT adapter。
5. 通过现有 verl eval 路径跑 32 条再 500 条 CodeContests 验证。
6. 在行为指标和 accepted rate 上与无 SFT baseline 对比。
7. 如果明显更好，写新决策正式采纳 SFT warm-start + RL。
