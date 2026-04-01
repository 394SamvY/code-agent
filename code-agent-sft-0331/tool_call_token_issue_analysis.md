# SFT 模型不生成 `<tool_call>` 标签问题分析

## 问题现象

SFT 训练完成（10 epoch, val_loss 0.433）后，multi_turn 评测 pass@1 = 0%。
模型完全没有触发工具调用：tool call 使用率 0%，平均轮数 1.0。

模型输出示例：
```
I'll implement a simple `add` function... then run the given assertion as a quick test.
توج{"name":"execute_code","arguments":"def add(a: int, b: int) -> int:..."}
```

期望输出：
```
I'll implement a simple `add` function...
<tool_call>
{"name": "execute_code", "arguments": {"code": "def add(a, b):..."}}
</tool_call>
```

模型学会了工具调用的**意图**（JSON 结构正确），但缺少 `<tool_call></tool_call>` 标签包裹，
导致评测脚本的 `_parse_tool_call()` 无法识别。

---

## 排查路径

### 1. 确认评测侧解析逻辑

`src/eval/evaluate.py` 的 `_parse_tool_call()` 使用正则匹配 `<tool_call>...</tool_call>`，
没有 fallback。模型不输出这个标签 → 解析为 None → 当作普通文本 → 对话在第 1 轮就结束。

### 2. 确认训练数据中是否包含 `<tool_call>` 标签

训练数据（parquet）使用 OpenAI 格式的 `tool_calls` 字段：
```python
{"role": "assistant", "content": "...", "tool_calls": [{"function": {"name": "execute_code", ...}}]}
```

verl 的 `MultiTurnSFTDataset` 使用 `tokenizer.apply_chat_template()` 渲染每条消息。
验证渲染结果：
```python
tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False)
```

输出确认包含 `<tool_call>` 标签：
```
<|im_start|>assistant
I'll test this.
<tool_call>
{"name": "execute_code", "arguments": "..."}
</tool_call><|im_end|>
```

**结论：训练数据渲染正确，模型在训练时确实看到了 `<tool_call>` token。**

### 3. 排除推理侧 token 过滤

检查 `<tool_call>` 的 token 属性：
```
token id = 151657
special = False（不是 special token，不会被 skip_special_tokens 过滤）
不在 tokenizer.all_special_tokens 列表中
```

**结论：SGLang / HuggingFace 不会过滤这个 token。**

### 4. 对比基座模型行为

用完全相同的 prompt 测试基座模型（Qwen2.5-Coder-7B-Instruct，未经 SFT）：
```
To write a function... Here's how I'll do it:
```json
{"name": "execute_code", "arguments": {"code": "..."}}
```
```

**基座模型也不输出 `<tool_call>` 标签！** 它用 markdown 代码块包裹 JSON。
说明基座模型的 `lm_head` 对 token 151657 的先验概率就很低。

### 5. 逐 token 分析 SFT 模型输出

对 SFT 模型的生成结果逐 token 解码：
```
位置 [27] id=   624  text='.\n'          ← 描述文本结束
位置 [28] id=137230  text='توج'          ← 应该是 <tool_call>(151657)，实际生成了乱码
位置 [29] id=  4913  text='{"'           ← JSON 内容开始（正确）
位置 [30] id=   606  text='name'
```

对比训练数据中的 token 序列：
```
位置 [199] id=151657 text='<tool_call>'  ← 训练时确实有这个 token
位置 [200] id=   198 text='\n'
位置 [201] id=  4913 text='{"'
```

**关键发现：SFT 模型在 `<tool_call>` 应该出现的位置（[28]）生成了 token 137230（"توج"），
而不是 151657（`<tool_call>`）。**

---

## 根因

```
LoRA target_modules = [q_proj, k_proj, v_proj, o_proj]
```

这四个都是 **注意力层** 的投影矩阵。LoRA 改变了模型内部的隐藏表示（hidden states），
但 **`lm_head`（输出投影层：hidden_state → vocab 概率分布）没有被 LoRA 适配**。

因果链：
1. 基座模型的 `lm_head` 对 token 151657（`<tool_call>`）的权重就很低
   （基座模型自己在相同 prompt 下也不生成这个 token）
2. SFT 训练只调整了注意力层，产生了"想调用工具"的新 hidden state
3. 这个新 hidden state 经过 **未改变** 的 `lm_head` 映射时，无法准确指向 token 151657
4. 最终映射到了相近但错误的 token 137230（"توج"）

类比：模型学会了"想说什么"（注意力层），但"嘴"（lm_head）没变，发不出正确的音。

---

## 修复

在 `configs/sft/sft_qwen_7b.yaml` 的 `target_modules` 中添加 `lm_head`：

```yaml
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - lm_head    # 输出投影层，使模型能学会生成 <tool_call> 等特定 token
```

这样 `lm_head` 也会被 LoRA 适配（增加约 2MB 参数），
模型就能将正确的概率分配给 `<tool_call>` token。

---

## 本轮训练指标（修复前，仅供参考）

| Epoch | Train Loss (范围) | Val Loss |
|-------|-------------------|----------|
| 2     | 0.51 ~ 0.73       | 0.608    |
| 3     | 0.41 ~ 0.62       | 0.531    |
| 4     | 0.39 ~ 0.64       | 0.484    |
| 5     | 0.35 ~ 0.52       | 0.459    |
| 6     | 0.33 ~ 0.58       | 0.445    |
| 7     | 0.33 ~ 0.55       | 0.438    |
| 8     | 0.32 ~ 0.52       | 0.435    |
| 9     | 0.32 ~ 0.50       | 0.433    |
| 10    | 0.32 ~ 0.50       | 0.433    |

Loss 已收敛，模型学到了工具调用的语义（JSON 结构），只是缺少 `<tool_call>` 标签。

---

## Loss 与困惑度的直觉理解

数学关系：`perplexity = e^loss`。困惑度的物理含义是**模型在每个位置"等效选择"了几个 token**。

### 两个极端

| 场景 | 概率 p | loss = -log(p) | perplexity = e^loss | 含义 |
|------|--------|----------------|---------------------|------|
| 完全不懂 | 1/150000 | 11.9 | 150000 | 15 万个 token 中盲猜 |
| 完全确定 | 1.0 | 0 | 1 | 只"看到"1 个选择，毫无犹豫 |

### 本轮 SFT 模型：loss = 0.4

- perplexity = e^0.4 ≈ **1.5**
- 含义：模型在每个位置平均像是在 **1.5 个等概率候选 token** 之间做选择

### 参考刻度

| Loss | 置信度 p | 困惑度 | 直觉类比 |
|------|---------|--------|---------|
| 0.1 | 90% | 1.1 | 几乎确定下一个 token |
| **0.4** | **67%** | **1.5** | **在 1~2 个候选间犹豫** |
| 0.7 | 50% | 2.0 | 抛硬币 |
| 1.0 | 37% | 2.7 | 明显不确定 |
| 2.0 | 14% | 7.4 | 基本靠猜 |

> "困惑度"（perplexity）的英文原意就是"困惑、迷茫"。
> 它衡量模型有多困惑：1 = 毫不犹豫，2 = 抛硬币，10 = 掷骰子。

---

## 修复后 loss 变化预测

**预测：val loss 从 0.433 降至 0.38~0.42 左右，会降但幅度不大。**

### 会降低的原因

加 `lm_head` 直接打通了"隐藏表示 → 词表概率"的优化路径。
之前 `<tool_call>`（151657）等 token 在冻结的 `lm_head` 下预测概率接近 0，
单个 token 的 loss 极高，但被其他几百个低 loss 的 token 平均掉了。
修复后这些"钉子户"位置的 loss 会大幅下降，拉低整体均值。

### 幅度不大的原因

定量估算：每条样本约 300 个 assistant token，其中只有 ~4 个是 tool_call 相关 token：

```
修复前：(296 × 0.4 + 4 × 5.0) / 300 ≈ 0.46
修复后：(296 × 0.4 + 4 × 0.1) / 300 ≈ 0.40
```

大部分 token（代码内容、变量名、注释）的预测难度不因 `lm_head` 的 LoRA 而改变——
这些 token 在基座模型的 `lm_head` 中本就有正常概率，瓶颈在代码生成的固有不确定性。

### 核心观点

**真正的变化不在 loss 数字上，而在质量上。**
从"生成不了 `<tool_call>`"到"能可靠生成 `<tool_call>`"，是 0 → 1 的质变。
一个 loss 为 0.43 但不生成 `<tool_call>` 的模型，远不如 loss 为 0.45 但能正确触发工具调用的模型。
这也说明**平均 loss 可以掩盖关键 token 的失败**——少数 token 的极高 loss 被大量低 loss token 稀释。

---

## 为什么必须生成 `<tool_call>` 标签

不只是评测需要，更因为下游 **GRPO 训练的硬性要求**。

verl 的多轮 agent rollout（`verl/experimental/agent_loop/tool_agent_loop.py`）
使用 `ToolParser` 注册机制解析模型输出，默认是 `HermesToolParser`（`format: hermes`），
**只认 `<tool_call>...</tool_call>` 格式**。

GRPO rollout 的关键判断逻辑：
```python
_, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(response_ids, tools)
if agent_data.tool_calls:
    return AgentState.PROCESSING_TOOLS   # → 执行工具 → 多轮继续
else:
    return AgentState.TERMINATED         # → 直接结束，无工具交互
```

如果 SFT 模型不输出 `<tool_call>` 标签：
- parser 返回空 → agent loop 第一轮就终止
- GRPO 无法获得有效的多轮工具交互轨迹
- 没有轨迹就无法计算 advantage，整个 RL 训练失效

**SFT 的核心目标 = 让模型以 verl 能识别的格式（hermes）可靠地触发工具调用，
为 GRPO 提供有效的初始策略。**
