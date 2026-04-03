# LoRA 适配 lm_head 仍无法生成 `<tool_call>` — 问题分析

## 背景

上一轮分析（`code-agent-sft-0331/tool_call_token_issue_analysis.md`）确认了根因：
LoRA 只适配了注意力层（q/k/v/o_proj），冻结的 `lm_head` 无法将新学到的隐藏表示
映射到 `<tool_call>` 特殊 token（id=151657）。

本轮修复方案：将 `lm_head` 加入 LoRA `target_modules`，重新训练 10 epoch。

**结果：失败。模型仍然不生成 `<tool_call>` 标签。**

---

## 训练指标对比

| 指标 | 上一轮（无 lm_head） | 本轮（lm_head LoRA） |
|------|---------------------|---------------------|
| LoRA target_modules | q/k/v/o_proj | q/k/v/o_proj + **lm_head** |
| 最终 val_loss | 0.433 | **0.409** |
| Adapter 大小 | ~39 MB | **~1.1 GB** |

Val loss 确实从 0.433 降到了 0.409（在上轮预测的 0.38~0.42 范围内），
说明 `lm_head` LoRA 有效降低了部分 token 的预测误差。但生成测试仍然失败。

---

## 调试路径

### 1. 导出 & 合并模型

导出 LoRA adapter 时遇到第一个问题：`convert_checkpoint.py` 中 `target_modules` 是硬编码的
`[q_proj, k_proj, v_proj, o_proj]`，不包含 `lm_head`，导致 state_dict key 不匹配。

**修复**：改为从 checkpoint 的 key 名自动检测 target_modules：
```python
target_modules = sorted({
    k.split(".")[k.split(".").index("lora_A") - 1]
    for k in state_dict if "lora_A" in k
})
# 检测结果: ['k_proj', 'lm_head', 'o_proj', 'q_proj', 'v_proj']
```

随后合并为完整 HF 模型，启动 SGLang 服务。

### 2. 快速生成测试

用 `/generate` 端点测试原始输出（不经过 chat completions 的 tool call 解析）：

| 温度 | 输出示例 | 有 `<tool_call>`？ |
|------|---------|-------------------|
| 0.0 | `{"name": "execute_command", "arguments": "{...}"}` | 否，直接输出 JSON |
| 0.3 | `<{"name": "execute_command", "arguments": "{...}"}>` | 否，用 `<>` 包裹 |
| 0.7 | `<tool name="execute_command" arguments='...' />` | 否，XML 风格 |

所有温度下，模型都没有生成正确的 `<tool_call>` token。它学到了"应该调用工具"的意图，
甚至学到了"应该用某种标签包裹"，但就是产出不了那个特定的 special token。

### 3. 权重差异验证

先确认 LoRA 合并确实生效了：

```
lm_head shape: [152064, 3584]
Max diff (base vs SFT): 0.002075
Mean diff: 0.000280
Non-zero diffs: 504,393,010 / 544,997,376 (92.6%)

Token 151657 (<tool_call>) row:
  max_diff = 0.001511, mean_diff = 0.000401
Token 151658 (</tool_call>) row:
  max_diff = 0.001564, mean_diff = 0.000391
```

权重确实被修改了，但 max_diff 只有 0.002，改动幅度非常小。

### 4. 概率分布分析（核心发现）

对相同 prompt，比较 SFT 模型和基座模型在第一个生成位置的 token 概率分布。

Prompt 由 `tokenizer.apply_chat_template` 渲染，包含 system、tools schema、user 三部分：
```
<|im_start|>system
You are a helpful assistant with access to tools.
...（工具 schema：execute_command）...
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Read the file /tmp/test.py<|im_end|>
<|im_start|>assistant
                          ← 模型从这里开始生成，以下分析的是此位置的 token 概率分布
```

**SFT 模型（lm_head LoRA）**：
```
Top-1:  {"        50.20%   ← 直接输出 JSON
Top-2:  <         36.72%   ← 普通角括号
Top-3:  ```        1.42%
...
Rank 605: <tool_call>  0.0012%   ← 目标 token
```

**基座模型（未经训练）**：
```
Top-1:  <         36.91%
Top-2:  ```       36.91%
Top-3:  To        11.98%
...
Rank 66436: <tool_call>  ~0.0000%   ← 几乎为零
```

**对比分析**：

| 指标 | 基座模型 | SFT（lm_head LoRA） | 变化 |
|------|---------|---------------------|------|
| `<tool_call>` 概率 | ≈0% | 0.0012% | 从零到有 |
| `<tool_call>` 排名 | 66,436 | **605** | 提升 100 倍 |
| `{"` 概率 | 3.89% | **50.20%** | 成为压倒性 Top-1 |
| `<` 概率 | 36.91% | 36.72% | 基本不变 |

LoRA 确实将 `<tool_call>` 从 rank 66K 提升到了 rank 605——**相对提升巨大**。
但 0.0012% 的绝对概率在 top-p / temperature 采样下永远不会被选中。

更糟的是，SFT 反而让 `{"` 从 3.89% 飙升到 50.20%，成为压倒性的首选。
这说明 **注意力层的 LoRA 成功学会了"在这里应该输出工具调用"的语义**，
但 `lm_head` 的 LoRA 适配能力不足，只能把概率分配给已有高概率的 `{"`，
而不是将概率转移到原本近零的特殊 token。

---

## 根因

### LoRA rank 16 对 lm_head 适配不足

`lm_head` 的权重矩阵维度为 `[152064, 3584]`（vocab_size × hidden_size），
共 5.45 亿参数。LoRA rank 16 引入的可训练参数为：

```
LoRA_A: [16, 3584]    =   57,344
LoRA_B: [152064, 16]  = 2,433,024
合计: 2,490,368（~250 万）
```

250 万参数要修改 5.45 亿参数矩阵中特定行（token 151657/151658）的输出概率，
这本质上是一个**低秩约束下的精确点对点映射问题**。

LoRA 的修改方式是 `W' = W + BA`，其中 `B ∈ R^{152064×16}`, `A ∈ R^{16×3584}`。
这意味着所有 152064 个 token 的输出概率变化共享同一个 16 维子空间。
要让 token 151657 的概率从 ~0 变为 >50%，同时不破坏其他 15 万个 token 的概率，
在 rank 16 的约束下几乎不可能做到。

### 对比：注意力层的 LoRA 为什么有效

| | 注意力层 (q/k/v/o_proj) | lm_head |
|---|---|---|
| 权重维度 | [3584, 3584] | [152064, 3584] |
| 参数量 | 1284 万 | 5.45 亿 |
| LoRA 比例 | rank 16 / 3584 ≈ 0.45% | rank 16 / 152064 ≈ 0.01% |
| 任务性质 | 改变注意力模式（连续、渐进） | 翻转个别 token 概率（离散、尖锐） |
| 效果 | 成功学会工具调用语义 | 仅微幅提升概率，不足以改变采样结果 |

注意力层的 LoRA 修改的是连续空间中的注意力模式，low-rank 近似效果好。
`lm_head` 需要的是对词表中个别 token 做大幅度的概率翻转——这本质上是高秩的。

---

## 解决方案

### 方案：modules_to_save（对 lm_head 全量微调）

PEFT 提供 `modules_to_save` 参数：将指定模块排除出 LoRA，改为全量微调。
其余模块继续使用 LoRA。

```yaml
target_modules:      # LoRA 适配
  - q_proj
  - k_proj
  - v_proj
  - o_proj
modules_to_save:     # 全量微调（创建完整可训练副本）
  - lm_head
```

这样 `lm_head` 的全部 5.45 亿参数都可训练，不受低秩约束，
可以精确调整任意 token 的输出概率。

**问题：verl 当前不支持 `modules_to_save` 参数。** 
需要修改两处 verl 源码：

1. `verl/workers/config/model.py` — 添加 `modules_to_save` 配置字段
2. `verl/workers/engine/fsdp/transformer_impl.py` — 传递给 `LoraConfig`

改动量很小（各约 2 行），但涉及修改框架源码。

---

## 补充：为什么 Qwen2.5-Coder 的 `<tool_call>` 先验概率接近零

Qwen2.5-Coder-7B-**Instruct** 是代码专用模型。虽然词表中存在 `<tool_call>` (151657)
和 `</tool_call>` (151658) 两个特殊 token，但该模型大概率未经过工具调用微调。

证据：基座模型在工具调用 prompt 下，`<tool_call>` 排名 66,436（词表总大小 152,064 的后半部分），
概率接近均匀分布中的随机 token。这意味着预训练/Instruct 微调阶段几乎没有见过这个 token 被生成的样本。

这与 Qwen2.5-7B-Instruct（通用版，非 Coder）不同——通用版可能经过了工具调用微调，
`<tool_call>` 的先验概率会高得多，LoRA 可能就够用了。
