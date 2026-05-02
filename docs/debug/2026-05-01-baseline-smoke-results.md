# baseline smoke 三组对比

日期：2026-05-01

## 背景

清理全部 thinking budget 代码后，跑三组 smoke 验证：
1. eval 环境是否正确（样本正常启停、无 OOM、0.jsonl 完整）
2. `MAX_RESPONSE_LENGTH` token 限制是否精确生效
3. base model 实际行为，为 SFT warm-start 设计提供依据

## 运行配置

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| `MAX_RESPONSE_LENGTH` | 8192 | 16384 | 16384 |
| `GPU_MEMORY_UTILIZATION` | 0.82 | 0.82 | 0.94 |
| `MAX_NUM_SEQS` | 32 | 32 | 48 |
| `AGENT_WORKERS` | 16 | 16 | 24 |
| `VAL_BATCH_SIZE` | 16 | 16 | 24 |
| `MAX_NUM_BATCHED_TOKENS` | 32768 | 32768 | 49152 |
| `VAL_MAX_SAMPLES` | 32 | 32 | 48 |
| 输出目录 | `...mr8192_20260501_210009` | `...mr16384_20260501_210717` | `...mr16384_20260501_212124` |

## 全局指标

```
                         Run 1            Run 2              Run 3
─────────────────────────────────────────────────────────────────────
  accepted                0/32    (0%)     0/32     (0%)       0/48    (0%)
  submitted               0/32    (0%)     0/32     (0%)       0/48    (0%)
  any_tool_call           0/32    (0%)     0/32     (0%)       0/48    (0%)
  no_tool                 32/32  (100%)    32/32   (100%)      48/48  (100%)
  avg_tool_calls          0.0              0.0                 0.0
  avg_output_chars        25,789           37,399              43,253
  num_turns               min=2 max=2     min=2 max=2         min=2 max=2
─────────────────────────────────────────────────────────────────────
```

- `num_turns=2`：所有样本 user turn → assistant turn → 无工具调用 → 终止，没有任何交互
- eval 环境正确：无 OOM、无 shape mismatch、0.jsonl 完整写出
- 耗时因日志无时间戳未能精确提取，后续已在脚本中加入 `start/end` 时间记录

## Token 限制验证

取三次运行都有的同一道题 "1044_B. Intersecting S..."，用 Qwen3 tokenizer 精确计数：

```
                     chars      tokens    </think>    <tool_call>
───────────────────────────────────────────────────────────────
Run 1 (mr=8192)     30,175     8,192     无          无
Run 2 (mr=16384)    62,086     16,384    有          无
Run 3 (mr=16384)    57,986     15,802    有          无
```

Token 限制**完全精确**：mr=8192 时在 8192 tokens 处截断，mr=16384 时在 16384 tokens 处截断。Run 3 的 15,802 是因为模型在达到上限前自然结束了。

**同一题的行为变化**：

- Run 1（8192 tokens）：纯推导，未闭合 `</think>`，预算耗尽被掐断
- Run 2（16384 tokens）：97.6% 的 tokens 用于 thinking，最后 1,496 字符闭合 `</think>` 并开始写代码，但是裸 Python 代码 + 文字说明，不是 `<tool_call>` 格式，在 `"queries ="` 处被截断
- Run 3（16384 tokens）：94.0% 的 tokens 用于 thinking，闭合后写了方案文字说明，自然结束，没有 `<tool_call>`

## Sample 0 逐个分析

用 Qwen3 tokenizer 精确分析每个 run 的 sample 0：

### Run 1 — Filling Game（mr=8192）

```
chars=32,556  tokens=8,192  </think>=False  <tool_call>=False
```

纯英文推导洪水填充问题。开头读题分析，结尾停在 "the minimal number of steps is the number of distinct colors in the initial grid, minus one. But this"。8192 tokens 耗尽截断，从未闭合 `</think>`。

### Run 2 — Diameter of Graph（mr=16384）

```
chars=27,472  tokens=8,007  </think>=True  <tool_call>=False
```

**只用了 8,007 tokens 就自然结束了**，远低于 16,384 上限。89.6% 的 tokens 用于 thinking，闭合 `</think>` 后写了完整的解题方案：

- 文字解释 Approach（Single Node Case、Minimum Edges、Diameter Constraints）
- ` ```python ... ``` ` markdown 代码块里的完整 Python 解答
- 总结说明

模型**会解题、会写代码、会闭合 `</think>`**，但代码写在 markdown 格式里，完全不知道要用 `<tool_call>{"name":"run_public_tests","arguments":{"code":"..."}}</tool_call>`。

### Run 3 — Frog Traveler（mr=16384）

```
chars=57,297  tokens=16,384  </think>=False  <tool_call>=False
```

纯推导青蛙跳井问题。16,384 tokens 耗尽截断，结尾停在 "But how to find this. This is getting too complex. Perhaps"。从未闭合 `</think>`。

## 结论

1. **Token 限制精确生效**：mr=8192 → 8192 tokens，mr=16384 → 16384 tokens
2. **eval 环境正确**：所有样本正常启停、无 OOM、0.jsonl 完整写出
3. **base model 的核心缺陷不是思考能力，是输出格式**：
   - 给够 token 预算（16384）后，部分样本能闭合 `</think>` 并写出正确代码
   - 但代码写在 markdown 代码块里，不知道 OJ 工具调用协议
   - SFT warm-start 要教的核心行为：`</think>` → `<tool_call>{"name":"run_public_tests","arguments":{"code":"..."}}</tool_call>`
4. **`MAX_RESPONSE_LENGTH` 和 GPU 参数对 base model 无意义**：当前默认值即可，等模型会调工具后再调
