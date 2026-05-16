# GRPO Reward 设计（OJ-like）

当前 reward 只有一个实现入口：`src/reward.py::compute_score`。

reward 不再解析 `solution_str`，而是只使用 agent loop 记录的真实工具执行事件：

```text
extra_info["code_agent_tool_events"]
extra_info["code_agent_trace"]
```

如果缺少 `code_agent_tool_events`，`compute_score` 会直接报错。这是刻意设计，用来避免 reward 从模型文本里复原或信任伪造的 tool response。

## 总公式

```text
R = R_outcome + R_bad_pattern
```

然后做最终裁剪：

```text
if outcome 非 accepted:
    R = min(R, 0.0)

R = clamp(R, -0.5, 1.0)
```

两项职责边界：

| 项 | 范围 | 职责 |
| --- | ---: | --- |
| `R_outcome` | `[-0.20, 1]` | 最终提交正确性，主奖励 |
| `R_bad_pattern` | `[-0.60, 0]` | 已知坏模式惩罚 |

`acc` 仍是稳定评测指标：最后一次 `submit_solution` 为 `accepted` 时为 `1.0`，否则为 `0.0`。

为区分“不会解题”和“会解题但不会停止”，`reward_breakdown` 和 reward 顶层返回都会强制记录：

```json
{
  "acc_final": 0.0,
  "acc_any": 1.0,
  "best_submit_pass_rate": 1.0,
  "last_submit_pass_rate": 0.4,
  "outcome_submit_policy": "last_submit"
}
```

## 结构化事件

`CodeAgentToolAgentLoop` 在每次真实工具执行后记录一条 event，核心字段：

```json
{
  "tool": "run_public_tests",
  "verdict": "wrong_answer",
  "passed": 2,
  "total": 5,
  "pass_rate": 0.4,
  "code": "...",
  "code_hash": "...",
  "semantic_hash": "...",
  "observation": "...",
  "first_failed": {"stderr": "..."},
  "error_kind": "wrong_answer"
}
```

这些事件来自 tool adapter 返回的结构化 judge result，而不是从 `solution_str` 文本里解析。

## Outcome Reward

默认使用最后一次 submit：

```text
submit_for_outcome = last_submit
```

公式：

```text
accepted -> 1.0
non-AC   -> -0.20 * (1 - pass_rate)
no submit -> 0.0
```

也就是说，非 AC submit 只拿非正的客观 partial credit：private pass rate 越高越接近
0，但只有 accepted 可以得到正 reward。

代码里保留 `outcome_submit_policy` 开关，可选：

```text
last_submit
best_submit
any_ac_else_best
```

默认仍是 `last_submit`，保持 `score` 和 `acc` 口径容易解释。

## Bad Pattern Penalty

Bad pattern 只惩罚，不给正奖励。

当前覆盖和默认权重：

| 坏模式 | 权重 | 说明 |
| --- | ---: | --- |
| `no_submit` | `-0.25` | 没有任何正式提交 |
| `public_acc_no_submit` | `-0.10` | public accepted 后没有 submit |
| `public_fail_same_code_submit` | `-0.06/count, cap -0.12` | public 失败后同代码 submit |
| `submit_fail_same_code_submit` | `-0.08/count, cap -0.16` | submit 失败后同代码再次 submit |
| `duplicate_public_code` | `-0.02/count, cap -0.08` | 重复跑相同代码 public tests |
| `duplicate_submit_code` | `-0.03/count, cap -0.10` | 重复提交相同代码 |
| `too_many_public_tests` | `-0.01/extra, cap -0.05` | 超过 5 次 public test 后递增 |
| `too_many_submits` | `-0.03/extra, cap -0.09` | 超过 2 次 submit 后递增 |
| `accepted_then_tool_call` | `-0.15` | accepted 后继续调用工具 |
| `accepted_then_later_failed_submit` | `-0.25` | accepted 后又 submit 且失败 |
| `response_truncated` | `-0.15` | trajectory 撞到 `max_response_length` / `response_length_exceeded` |
| `truncated_no_submit` | `-0.05` | 撞到 response 上限且没有任何正式提交 |
| `public_test_limit_exceeded` | `-0.08` | 撞 public test 上限 |
| `submission_limit_exceeded` | `-0.12` | 撞 submit 上限 |
| `malformed_tool_call` | `-0.15/count, cap -0.30` | agent loop 记录到 parse failure / tool-call 格式错误 |
| `unknown_tool` | `-0.20/count, cap -0.30` | 模型调用了 schema 外工具 |
| `tool_execution_error` | `-0.20/count, cap -0.30` | 已知工具名下参数或 adapter 执行失败 |
| `accepted_then_long_text` | `-0.02/-0.05` | accepted 后继续输出超过 500/1500 chars |

`R_bad_pattern` 总体裁剪到 `[-0.60, 0.0]`。非 accepted 轨迹最终 reward 最高为 `0.0`；
`no_submit` 至少压到 `-0.25`，public accepted 但未 submit 至少压到 `-0.35`，截断且未提交至少压到 `-0.35`；
未 accepted 且出现 malformed / unknown tool / tool execution error 至少压到 `-0.30`。如果已经 accepted 但仍出现协议错误，最终 reward 最高为 `0.85`。

## 调试输出

`reward_breakdown` 会记录两项来源：

```json
{
  "reward_formula": "outcome + bad_pattern",
  "outcome_reward": 1.0,
  "acc_final": 1.0,
  "acc_any": 1.0,
  "best_submit_pass_rate": 1.0,
  "last_submit_pass_rate": 1.0,
  "bad_pattern": -0.15,
  "final": 0.85,
  "acc": 1.0,
  "bad_patterns": {
    "accepted_then_tool_call": -0.15
  }
}
```

后续调 reward 时优先看 `reward_breakdown`，不要只看 scalar `score`。
