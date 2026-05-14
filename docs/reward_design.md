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
R = R_outcome + R_debug_prm + R_bad_pattern
```

然后做最终裁剪：

```text
if outcome 非 accepted:
    R = min(R, 0.6)

R = clamp(R, -0.5, 1.0)
```

三项职责边界：

| 项 | 范围 | 职责 |
| --- | ---: | --- |
| `R_outcome` | `[0, 1]` | 最终提交正确性，主奖励 |
| `R_debug_prm` | `[-0.10, 0.15]` | 反馈条件下的有效 debug |
| `R_bad_pattern` | `[-0.40, 0]` | 已知坏模式惩罚 |

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
non-AC   -> min(0.5, 0.4 * pass_rate)
no submit -> 0.0
```

代码里保留 `outcome_submit_policy` 开关，可选：

```text
last_submit
best_submit
any_ac_else_best
```

默认仍是 `last_submit`，保持 `score` 和 `acc` 口径容易解释。

## Debug PRM

Debug PRM 按相邻工具执行 transition 打分，但不会把所有相邻事件都按同一公式比较：

```text
(prev_code, prev_feedback) -> next_code -> next_result
```

不同 tool pair 使用不同规则：

| transition | Debug PRM 规则 |
| --- | --- |
| `run_public_tests -> run_public_tests` | 完整 debug scoring，权重 `1.0` |
| `submit_solution -> submit_solution` | 弱 debug scoring，权重 `0.3`，且 submit-based debug 总额单独 cap 到 `[-0.03, +0.03]` |
| `run_public_tests -> submit_solution` | 不给 debug bonus；同代码坏模式交给 `R_bad_pattern` |
| `submit_solution -> run_public_tests` | 不比较 pass rate；只允许非常保守的反馈对齐信号 |

主要信号：

| 信号 | 说明 |
| --- | --- |
| `same_code_after_failed_feedback` | 失败反馈后同代码重试，扣分 |
| `state_rank_improved` | 修改后状态等级提升，加分 |
| `pass_rate_improved` | 修改后通过率提升，弱加分 |
| `feedback_aligned_not_worse` | 修改和错误类型高置信对齐，且结果不变差，加分 |
| `feedback_aligned_but_worse` | 看似对齐但结果变差，扣分 |
| `large_rewrite_worse` | 大幅重写且结果变差，扣分 |

public feedback 权重为 `1.0`，submit feedback 权重为 `0.3`，避免模型把 private submit 当 debug oracle 反复探测。

进步奖励使用分层优先级，`state_rank_improved` 和 `pass_rate_improved` 不同时叠加，而是取较大的 progress bonus，再额外叠加 alignment bonus。

正向 debug 信号只在对应 tool 类型的历史 best-so-far 状态被刷新时给分：

```text
next_state_rank > best_state_rank_so_far
or
next_pass_rate > best_pass_rate_so_far
```

这样避免模型在 `syntax_error -> wrong_answer -> syntax_error -> wrong_answer` 或相同通过率之间来回震荡时重复拿过程分。

当前高置信对齐规则只覆盖 `IndexError`、`KeyError`、`RecursionError`。`time_limit_exceeded` 只有在复杂度相关修改后结果同时变好时才给 alignment bonus。`wrong_answer` 不算高置信 alignment，只保留很弱的 `wrong_answer_logic_changed_not_worse` 信号，单次贡献为 `0.005 * weight`，且同样受 best-so-far 约束。

没有任何 `submit_solution` 的 trajectory 不允许拿正向 debug 分：

```text
if no_submit and R_debug_prm > 0:
    R_debug_prm = 0
```

负向 debug 信号仍保留，例如失败反馈后同代码重试仍会扣分。

## Bad Pattern Penalty

Bad pattern 只惩罚，不给正奖励。

当前覆盖：

| 坏模式 | 说明 |
| --- | --- |
| `no_submit` | 没有任何正式提交 |
| `public_acc_no_submit` | public accepted 后没有 submit |
| `public_fail_same_code_submit` | public 失败后同代码 submit |
| `submit_fail_same_code_submit` | submit 失败后同代码再次 submit |
| `duplicate_public_code` | 重复跑相同代码 public tests |
| `duplicate_submit_code` | 重复提交相同代码 |
| `too_many_public_tests` | public test 次数过多 |
| `too_many_submits` | submit 次数过多 |
| `accepted_then_tool_call` | accepted 后继续调用工具 |
| `accepted_then_later_failed_submit` | accepted 后又 submit 且失败 |
| `response_truncated` | trajectory 撞到 `max_response_length` / `response_length_exceeded` |
| `truncated_no_submit` | 撞到 response 上限且没有任何正式提交 |
| `public_test_limit_exceeded` | 撞 public test 上限 |
| `submission_limit_exceeded` | 撞 submit 上限 |
| `malformed_tool_call` | agent loop 记录到 parse failure |
| `accepted_then_long_text` | accepted 后继续长篇输出 |

`R_bad_pattern` 总体裁剪到 `[-0.40, 0.0]`，避免工程约束完全盖过 outcome。

## 调试输出

`reward_breakdown` 会记录三项来源：

```json
{
  "reward_formula": "outcome + debug_prm + bad_pattern",
  "outcome_reward": 1.0,
  "acc_final": 1.0,
  "acc_any": 1.0,
  "best_submit_pass_rate": 1.0,
  "last_submit_pass_rate": 1.0,
  "debug_prm": 0.0,
  "bad_pattern": -0.2,
  "final": 0.8,
  "acc": 1.0,
  "debug_signals": {},
  "bad_patterns": {
    "accepted_then_tool_call": -0.2
  }
}
```

后续调 reward 时优先看 `reward_breakdown`，不要只看 scalar `score`。
