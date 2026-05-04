# SFT 轨迹数据分析

## 数据概览

**来源：** DeepSeek V4 + Claude Sonnet v1 + Claude Sonnet v2 三份 teacher 轨迹合并  
**文件：** `data/verl/sft/sft_trajectories_accepted.jsonl`（1328 条）  
**Train/Val 划分：** 95:5（train 1262, val 66）  
**Tokenizer：** Qwen3-8B  
**生成时间：** 2026-05-04 ~ 2026-05-05  

## 合并规则

| 来源 | 原始条数 | 说明 |
|------|:---:|------|
| DeepSeek V4 | 712 | `sft_trajectories_deepseek.jsonl` |
| Sonnet v1 | 488 | `sft_trajectories_sonnet.jsonl`, budget=2048 |
| Sonnet v2 | 313 | `sft_trajectories_sonnet_v2.jsonl`, budget=8192, 每轮思考提示 |
| **排除** | -185 | `no_tool_call` 类型（模型未调用工具） |
| **合计** | **1328** | |

## Verdict 分布

| Verdict | 数量 | 占比 | SFT 价值 |
|---------|:---:|:---:|------|
| accepted | 1054 | 79.4% | 完整正确轨迹 |
| wrong_answer | 197 | 14.8% | 调试策略、错误分析 |
| max_turns | 59 | 4.4% | 多次重试过程（部分有效） |
| time_limit_exceeded | 15 | 1.1% | 超时后的优化策略 |
| runtime_error | 3 | 0.2% | 报错修复过程 |

> 保留非 accepted 类型是为了让模型见到"失败→分析→修复"循环，增加数据鲁棒性。

## Token 长度分布

以下数据均使用 Qwen3-8B tokenizer 真实编码计算。

### 首轮 thinking

| 指标 | tokens |
|------|:---:|
| 最小 | 64 |
| 中位 | 981 |
| 均值 | 2,026 |
| P90 | 5,431 |
| P95 | 8,513 |
| 最大 | 17,813 |

### 后续轮 thinking（仅非零）

| 指标 | tokens |
|------|:---:|
| 最小 | 2 |
| 中位 | 571 |
| 均值 | 1,444 |
| P90 | 4,053 |
| P95 | 6,898 |
| 最大 | 18,748 |

> 后续轮 thinking 呈双峰分布：多数很短（微调修复），少数较长（重新分析）。

### 每条样本总量

| 指标 | 总 tokens | thinking tokens |
|------|:---:|:---:|
| 最小 | 617 | 72 |
| 中位 | 3,426 | 1,496 |
| 均值 | 8,625 | 4,839 |
| P90 | 21,283 | 12,799 |
| P95 | 30,633 | 21,745 |
| 最大 | 95,385 | 85,173 |

### 字符/token 比率验证

首轮 think 均值 6,632 字符 / 2,026 tokens ≈ **3.3 字符/token**。之前用 3.0 估算略微偏低，代码密集区域比率更高。

## 对话轮数分布

| Turns | 数量 | 占比 |
|------:|:---:|:---:|
| 2 | 843 | 63.5% |
| 3 | 67 | 5.0% |
| 4 | 58 | 4.4% |
| 5 | — | — |
| 6-9 | ~90 | ~7% |
| 10-19 | ~210 | ~16% |
| 20 | 64 | 4.8% |

> 63.5% 的题目一轮过（public test + submit），中位 2 轮，均值 5 轮。

## 训练注意事项

1. **长序列截断：** P95 总长 30,633 tokens，超过 `MAX_LENGTH=20480`（训练脚本默认值）的样本会被截断。`truncation=right` 会丢弃尾部，可能丢失最后的 tool response。建议提升 `MAX_LENGTH` 到 32768 或设 `truncation=left` 保留尾部。
2. **数据多样性：** 531 道题有 2 份不同 teacher 的轨迹（同题异构），有助于 SFT 模型学习多样化解法。
3. **思考覆盖率：** Sonnet v2（每轮思考提示）95.9% 的 assistant 消息含 `<think>`，v1 仅 63.3%。train set 中 v2 占比较高，思考模式更完整。
4. **错误恢复：** 14.8% 的 `wrong_answer` + 其他失败类型提供了丰富的错误处理样本，模型应能学会"遇到错误→分析原因→修复代码"的模式。
5. **序列长度与 padding：** 当前脚本使用 `pad_mode=no_padding` + `max_token_len_per_gpu` 做 dynamic batch，避免大量 padding 浪费。中位 3426 tokens 对 FSDP 训练内存友好，但 P90 以上长尾样本需注意 OOM。
