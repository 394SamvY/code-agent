# Debug: SFT 模型不调用 tool_call 问题排查

日期: 2026-05-03

## 问题描述

使用 verl SFT trainer 对 Qwen3-8B 进行全量微调后，评测时模型只输出 `<think>` 推理内容，不调用 `run_public_tests` / `submit_solution` 工具。评测 32 条样本，**0 次工具调用**，score 全为 0。

```
val-aux/codecontests/num_tool_calls/mean@1: 0.0
val-aux/num_turns/min: 2, max: 2, mean: 2.0  # 仅 user + assistant 各一轮
val-aux/codecontests/score/mean@1: 0.0
```

## 排查过程

### 1. 检查 SFT 训练数据格式

**结论：数据格式正确。**

训练数据 `data/verl/sft/sft_accepted_train.parquet`：
- 435 条样本，包含 `messages`, `tools`, `enable_thinking` 等字段
- 每条轨迹 6 条消息：system → user → assistant(think+tool_call) → tool → assistant(think+tool_call) → tool
- 1098 条 assistant 消息中，**100%** 同时包含 `</think>` 和 `<tool_call>`
- 工具 schema 包含 `run_public_tests` 和 `submit_solution`

### 2. 检查 SFT 训练是否保留了 `<think>` 内容

**发现：`FixedMultiTurnSFTDataset` 修复是有效的。**

verl 的 `MultiTurnSFTDataset` 逐条对消息调用 `tokenizer.apply_chat_template([单条消息])`。Qwen3 的 chat_template 有一个逻辑：

```jinja
{%- if loop.index0 > ns.last_query_index %}
  {# 渲染 <think> 包装 #}
{%- else %}
  {# 丢弃 reasoning_content，只输出 content #}
{%- endif %}
```

逐条传入时 `last_query_index=0, loop.index0=0`，`0 > 0` 为 false → `<think>` 被丢弃。

`FixedMultiTurnSFTDataset` 将 `>` 改为 `>=`：

```python
# src/verl_sft_dataset_fix.py
self.tokenizer.chat_template = template.replace(
    "loop.index0 > ns.last_query_index",
    "loop.index0 >= ns.last_query_index",
)
```

**验证结果**：修复后 per-message tokenization 同时保留了 `<think>` 和 `<tool_call>`：

```
<|im_start|>assistant
<think>
The problem: The bear can choose a day d...
</think>

<tool_call>{"name": "run_public_tests", "arguments": {...}}</tool_call><|im_end|>
```

### 3. 检查评测 prompt 是否包含 tool schema

**结论：prompt 包含完整的 tool schema。**

评测脚本 `evaluate_baseline_with_verl.sh` 配置：
- `data.tool_config_path=configs/verl/tool_config.yaml`
- `data.apply_chat_template_kwargs.enable_thinking=true`
- `actor_rollout_ref.rollout.multi_turn.enable=true`

verl 的 `ToolAgentLoop._handle_pending_state()` 调用 `apply_chat_template(messages, tools=self.tool_schemas)`，生成的 prompt 包含：

```
<|im_start|>system
You are an expert Python programmer...

# Tools

You may call one or more functions...
<tools>
{"type": "function", "function": {"name": "run_public_tests", ...}}
{"type": "function", "function": {"name": "submit_solution", ...}}
</tools>

For each function call, return a json object...within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
[题目内容]<|im_end|>
<|im_start|>assistant
```

### 4. 检查 SFT checkpoint 是否被正确评测

**确认：评测使用的是 SFT checkpoint，不是基座模型。**

评测输出目录：
```
outputs/verl_baseline_eval/codecontests_test_huggingface_mp4096_mr28672_20260503_045143/
```

目录名中 `huggingface` 来自 `basename(".../global_step_81/huggingface")`，不是指基座模型。评测日志确认模型路径：

```
model: /root/autodl-tmp/code-agent/outputs/verl_sft/qwen3_8b_oj_sft_20260503_040653/global_step_81/huggingface
```

### 5. 用 HuggingFace 直接测试 SFT 模型（关键实验）

**实验设置**：加载 SFT checkpoint，使用与评测完全相同的 prompt 和采样参数（temperature=0.6, top_p=0.95, top_k=20, enable_thinking=true, tools=tool_schemas）。

**`</think>\n\n` 之后的下一个 token 概率分布**：

```
=== Top-20 after '</think>\n\n' (inference prompt) ===
  [ 1] id=151657  93.0% '<tool_call>'  <- rank 1, 压倒性最高
  [ 2] id=    40   3.6% 'I'
  [ 3] id=  1654   1.1% 'We'
  ...

  <tool_call>:    prob=92.969%  rank=1
  ``` :          prob=0.000%
  \n:            prob=0.000%
```

即使用实际 eval 输出中的长 think block 作为上下文（~2400 tokens），`<tool_call>` 概率仍有 **99.6%**：

```
=== LONG think (eval example 0) (prompt_tokens=2480) ===
  [ 1] id=151657  99.6% '<tool_call>'
```

**实际生成测试（temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=3000）**：

8 条样本中，**2/8 (25%) 成功输出 `<tool_call>`**：

```
Ex 0: 1014 tokens, </think> at char 2089, <tool_call>=True  ✓
Ex 1: 3000 tokens, </think>=-1, <tool_call>=False          ✗ (卡在 think 循环)
Ex 2: 3000 tokens, </think> at char 8056, <tool_call>=True ✓
Ex 3: 3000 tokens, </think>=-1, <tool_call>=False          ✗
Ex 4: 3000 tokens, </think>=-1, <tool_call>=False          ✗
Ex 5: 3000 tokens, </think>=-1, <tool_call>=False          ✗
Ex 6: 3000 tokens, </think>=-1, <tool_call>=False          ✗
Ex 7: 3000 tokens, </think>=-1, <tool_call>=False          ✗
```

**结论**：token 级别的概率分布完美（`<tool_call>` rank 1, 99.6%），但实际生成时模型经常在 `<think>` 循环中出不来（6/8 在 3000 token 内未闭合 `</think>`）。

### 6. SGLang vs HF 对比

| | HF (HuggingFace) | SGLang (verl eval) |
|---|---|---|
| `</think>` 闭合率 | ~25% (2/8) | ~3% (1/32 example 0) |
| `<tool_call>` 生成 | 25% | **0%** |
| 评测 score | - | **0.0** |

**关键矛盾**：即使 HF 下只有 25% 成功率，SGLang 下也应该是 ~25% 而不是 0%。SGLang 比 HF 更差。

### 7. 排除的因素

| 因素 | 状态 | 证据 |
|---|---|---|
| 训练数据缺少 tool_call | ✗ 已排除 | 100% assistant 消息含 `<tool_call>` |
| chat_template 丢弃 `<think>` | ✗ 已排除 | `FixedMultiTurnSFTDataset` 修复生效 |
| 评测 prompt 缺少 tool schema | ✗ 已排除 | agent loop 正确注入 `tools=self.tool_schemas` |
| 评测用了基座模型而非 SFT | ✗ 已排除 | 日志确认模型路径为 SFT checkpoint |
| SFT tokenizer 缺少 `>=` fix | ✗ 已排除 | `chat_template.jinja` 确认有 `>=` |
| `<tool_call>` 是 special token 被跳过 | ✗ 已排除 | `tokenizer_config.json` 中 `special: false` |
| 采样参数导致问题 | ✗ 部分排除 | HF 同参数下有 25% 成功率 |
| `skip_tokenizer_init` 影响 | ✗ 已排除 | 只影响 MoE routed experts 提取 |

### 8. 关于 SGLang 的额外发现

- SGLang 版本: **0.5.9**
- 直接 `import sglang` 报错: `AssertionError: duplicate template name`（与 PyTorch 版本不兼容）
- 但 verl 通过自己的 launcher 启动 SGLang server 能正常运行
- `custom_chat_template: None`, `tokenizer_path: None` — SGLang 从模型路径加载 tokenizer

## 根因分析

问题有**两层**：

### 层一：SFT 训练不足

- 435 条样本、3 epoch、LR=1e-5、batch_size=16
- 模型学会了 `<think>`→`</think>`→`<tool_call>` 的 token 级别概率模式（99.6% 准确）
- 但**在实际生成中经常无法闭合 `</think>`**，陷入过长的推理循环
- HF 下只有 25% 成功率（8 条样本中 2 条）

### 层二：SGLang 推理进一步降低了成功率

- HF: 25% → SGLang: 0%
- 可能原因：
  - SGLang 0.5.9 与当前 PyTorch 的兼容性问题
  - tensor parallelism / GPU 设置影响 logits 精度
  - SGLang 的 KV cache 或 attention 实现差异

## 建议下一步

### 短期（立即尝试）

1. **Greedy 评测**排除采样随机性：
   ```bash
   VAL_TEMPERATURE=0 VAL_DO_SAMPLE=false VAL_MAX_SAMPLES=32 \
     bash scripts/evaluate_baseline_with_verl.sh codecontests_test \
     /root/autodl-tmp/code-agent/outputs/verl_sft/qwen3_8b_oj_sft_20260503_040653/global_step_81/huggingface
   ```

2. **换 vLLM**（如果 verl 支持）：
   修改 eval 脚本中 `actor_rollout_ref.rollout.name=vllm`

3. **检查 SGLang-PyTorch 兼容性**：
   ```bash
   python3 -c "import sglang"  # 当前报错
   ```

### 中期（改进 SFT）

4. **增加训练强度**：提高 epoch 数（5-10）或 LR（5e-5）
5. **增大训练数据**：当前 435 条偏少
6. **降低训练时的 max_length**：确保不截断 tool_call 部分（当前 20480 已足够）

### 长期

7. **对比不同推理引擎**（HF / vLLM / SGLang）对同一 SFT 模型的生成质量
8. **加 RL (GRPO)** 进一步训练 tool calling 行为
