# 评测/训练采样轨迹 JSONL Schema 草案

更新日期：2026-05-13

## 目标

后续 validation 的 `generations/0.jsonl`、`generations/partial_0.jsonl`，以及 RL 训练时落盘的采样轨迹，每行都应统一写成样本级结构化记录，避免把“episode 为什么停”和“最后一次正式提交结果”混在同一个字段里。

核心口径：

- `episode.stop_reason` 只描述 episode 为什么停止。
- `judge.final_submit_verdict` 只描述最后一次 `submit_solution` 的 verdict。
- `metrics.acc` / `metrics.acc_final` 按最后一次 `submit_solution` 计算。
- `metrics.acc_any` 记录 trajectory 中是否曾经有任意 submit accepted，用来区分“不会解题”和“会解题但不会停止”。
- reward 只消费 agent loop 记录的真实 `trajectory.tool_events` / `extra_info["code_agent_tool_events"]`，不解析 `trajectory.output`。
- `behavior.has_tool_call_after_submit_accepted` 记录模型在正式提交 AC 后是否又继续调用工具，可用于后续 reward 惩罚。

## JSONL 每行示例

```json
{
  "sample": {
    "task_id": "livecodebench/1873_A",
    "data_source": "livecodebench"
  },
  "trajectory": {
    "input": "system\nYou are an expert Python programmer...\nuser\nSolve the following OJ-style programming problem...",
    "output": "<think>\nWe need solve Short Sort...\n</think>\n<tool_call>{\"name\":\"run_public_tests\",\"arguments\":{\"code\":\"...\"}}</tool_call>\n<tool_response>\nrun_public_tests: accepted. 1/1 tests passed.\nAll tests passed.\n</tool_response>\nassistant\n<tool_call>{\"name\":\"submit_solution\",\"arguments\":{\"code\":\"...\"}}</tool_call>\n<tool_response>\nsubmit_solution: accepted. 4/4 tests passed.\nAll tests passed.\n</tool_response>\nassistant\nThe solution passed all tests.",
    "messages": [
      {
        "role": "system",
        "content": "You are an expert Python programmer. Write complete Python programs that read from stdin and write to stdout."
      },
      {
        "role": "user",
        "content": "Solve the following OJ-style programming problem: Title: A. Short Sort ..."
      },
      {
        "role": "assistant",
        "content": "<think>\nWe need solve Short Sort...\n</think>",
        "tool_calls": [
          {
            "id": "call_0",
            "type": "function",
            "function": {
              "name": "run_public_tests",
              "arguments": "{\"code\":\"t=int(input())\\nfor _ in range(t):\\n    s=input().strip()\\n    print('YES' if s in {'abc','acb','bac','cba'} else 'NO')\\n\"}"
            }
          }
        ]
      },
      {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": "run_public_tests: accepted. 1/1 tests passed.\nAll tests passed."
      },
      {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_1",
            "type": "function",
            "function": {
              "name": "submit_solution",
              "arguments": "{\"code\":\"t=int(input())\\nfor _ in range(t):\\n    s=input().strip()\\n    print('YES' if s in {'abc','acb','bac','cba'} else 'NO')\\n\"}"
            }
          }
        ]
      },
      {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "submit_solution: accepted. 4/4 tests passed.\nAll tests passed."
      },
      {
        "role": "assistant",
        "content": "The solution passed all tests."
      }
    ],
    "tool_events": [
      {
        "index": 0,
        "tool": "run_public_tests",
        "verdict": "accepted",
        "passed": 1,
        "total": 1,
        "pass_rate": 1.0,
        "code": "t=int(input())\nfor _ in range(t):\n    s=input().strip()\n    print('YES' if s in {'abc','acb','bac','cba'} else 'NO')\n",
        "code_hash": "...",
        "semantic_hash": "...",
        "observation": "run_public_tests: accepted. 1/1 tests passed.\nAll tests passed.",
        "first_failed": null,
        "tool_reward": 0.0,
        "error_kind": "accepted"
      },
      {
        "index": 1,
        "tool": "submit_solution",
        "verdict": "accepted",
        "passed": 4,
        "total": 4,
        "pass_rate": 1.0,
        "code": "t=int(input())\nfor _ in range(t):\n    s=input().strip()\n    print('YES' if s in {'abc','acb','bac','cba'} else 'NO')\n",
        "code_hash": "...",
        "semantic_hash": "...",
        "observation": "submit_solution: accepted. 4/4 tests passed.\nAll tests passed.",
        "first_failed": null,
        "tool_reward": 1.0,
        "error_kind": "accepted"
      }
    ]
  },
  "episode": {
    "stop_reason": "no_tool_call",
    "assistant_token_count": 620,
    "output_token_count": 690,
    "response_length": 8192,
    "parse_failures": 0
  },
  "judge": {
    "final_submit_verdict": "accepted",
    "final_submit_reward": 1.0,
    "final_submit_passed": 4,
    "final_submit_total": 4
  },
  "behavior": {
    "num_tool_calls": 2,
    "public_test_call_count": 1,
    "submission_count": 1,
    "has_tool_call_after_submit_accepted": false
  },
  "metrics": {
    "acc": 1.0,
    "acc_final": 1.0,
    "acc_any": 1.0,
    "best_submit_pass_rate": 1.0,
    "last_submit_pass_rate": 1.0,
    "reward": 1.0,
    "outcome_reward": 1.0,
    "debug_prm": 0.0,
    "bad_pattern": 0.0,
    "reward_breakdown": "{\"reward_formula\":\"outcome + debug_prm + bad_pattern\",...}"
  },
  "verl": {
    "step": 0,
    "rollout_index": 0
  }
}
```

## 字段说明

### `sample`

| 字段 | 说明 |
| --- | --- |
| `task_id` | 题目 ID，例如 `codecontests/cefcfc056dd0af1e` 或 `livecodebench/1873_A`。 |
| `data_source` | 数据源名，例如 `codecontests`、`livecodebench`。通常可由 `task_id` 前缀得到。 |

### `trajectory`

| 字段 | 说明 |
| --- | --- |
| `input` | decode 后的原始 prompt 字符串。 |
| `output` | decode 后的完整 verl response 字符串，包含 assistant 生成内容、tool call 和 tool observation。 |
| `messages` | 结构化 chat messages，用于复盘轨迹。 |
| `tool_events` | agent loop 记录的真实工具执行事件。当前 reward 的权威输入来自同一份事件流，而不是 `output` 文本解析。 |

### `trajectory.tool_events`

| 字段 | 说明 |
| --- | --- |
| `index` | 工具事件序号。 |
| `tool` | `run_public_tests` 或 `submit_solution`。 |
| `verdict` | judge verdict。 |
| `passed` / `total` / `pass_rate` | 本次工具执行的测试通过情况。 |
| `code` | 本次工具调用执行的完整 Python 程序。 |
| `code_hash` | 规范化文本 hash，用于重复代码检测。 |
| `semantic_hash` | AST 级 hash，用于语义重复检测。 |
| `observation` | 返回给模型的 observation 文本，仅用于复盘；reward 不从这里重新解析 verdict。 |
| `first_failed` | 首个失败 case 的结构化信息；无失败时为 `null`。 |
| `tool_reward` | verl tool 层即时 reward，主要用于 trace；最终训练 reward 由 `src/reward.py` 计算。 |
| `error_kind` | reward/debug PRM 使用的粗粒度错误类型，如 `index_error`、`wrong_answer`。 |

### `episode`

| 字段 | 说明 |
| --- | --- |
| `stop_reason` | episode 停止原因。只描述停止机制，不表示 judge 结果。 |
| `assistant_token_count` | assistant 生成 token 数；包含完整 tool call 文本，不包含 tool response observation。 |
| `output_token_count` | verl `response_ids` 对应的完整 token 数；包含 assistant 生成内容和 tool response observation。 |
| `response_length` | 本次 rollout 配置的 response budget，例如 `8192`。 |
| `parse_failures` | 模型输出疑似包含 `<tool_call>` 但解析失败的次数。 |

建议的 `stop_reason` 枚举：

```text
no_tool_call
tool_call_limit_exhausted
response_length_exceeded
malformed_tool_call
public_test_limit_exhausted
submission_limit_exhausted
```

说明：模型 `submit_solution` 得到 `accepted` 后，理想行为是再生成简短总结并不再调用工具，因此这类正常 AC 轨迹的 `episode.stop_reason` 应为 `no_tool_call`，而不是 `accepted`。

### `judge`

| 字段 | 说明 |
| --- | --- |
| `final_submit_verdict` | 最后一次 `submit_solution` 的 verdict；没有正式提交时为 `no_submission`。 |
| `final_submit_reward` | 最后一次正式提交对应的工具 reward；没有提交时为 `0.0`。 |
| `final_submit_passed` | 最后一次正式提交通过的测试数；没有提交时为 `null`。 |
| `final_submit_total` | 最后一次正式提交的测试总数；没有提交时为 `null`。 |

建议的 `final_submit_verdict` 枚举：

```text
accepted
wrong_answer
runtime_error
time_limit_exceeded
syntax_error
submission_limit_exceeded
no_tests
no_submission
```

### `behavior`

| 字段 | 说明 |
| --- | --- |
| `num_tool_calls` | 工具调用总次数。 |
| `public_test_call_count` | `run_public_tests` 调用次数。 |
| `submission_count` | 真实消耗的正式提交次数。 |
| `has_tool_call_after_submit_accepted` | 是否在某次 `submit_solution` 返回 `accepted` 后又继续调用工具。第一次 AC submit 本身不算“之后”。 |

### `metrics`

| 字段 | 说明 |
| --- | --- |
| `acc` | 稳定评测口径，按最后一次正式提交计算：`1.0 if judge.final_submit_verdict == "accepted" else 0.0`。 |
| `acc_final` | 同 `acc`，显式表示最后一次 submit 是否 accepted。 |
| `acc_any` | trajectory 中任意一次 submit accepted 即为 `1.0`。 |
| `best_submit_pass_rate` | 所有 submit 中最佳 pass rate。 |
| `last_submit_pass_rate` | 最后一次 submit 的 pass rate。 |
| `reward` | 训练优化目标，当前为 `R_outcome + R_debug_prm + R_bad_pattern`。 |
| `outcome_reward` | 最终正确性主奖励。 |
| `debug_prm` | 反馈条件下 debug 过程奖励。 |
| `bad_pattern` | 已知坏模式惩罚。 |
| `reward_breakdown` | JSON 字符串，记录 reward 三项来源、submit 诊断和具体信号。 |

### `verl`

| 字段 | 说明 |
| --- | --- |
| `step` | verl `global_steps`。 |
| `rollout_index` | 同一个 prompt 采样多条 rollout 时的编号；`n=1` 时为 `0`。 |

## 训练采样落盘

RL 训练时落盘的 rollout/sample JSONL 也使用同一套 schema。这样 validation、训练采样、失败样本分析和 reward debug 可以复用同一套统计脚本。

训练采样和 validation 的主要差异只应体现在文件组织上：

- validation：通常写到 `generations/{step}.jsonl` 和 `generations/partial_{step}.jsonl`。
- training rollout：可按 step 写到 `rollout_data/{step}.jsonl` 或类似目录。
- 同一 prompt 如果采样多条 rollout，每条 rollout 仍然是一行独立 JSONL。
- `sample.task_id`、`trajectory`、`episode`、`judge`、`behavior`、`metrics` 的字段和口径保持一致。
- reward 统计以 `trajectory.tool_events` / `metrics.reward_breakdown` 为准，不从 `trajectory.output` 反推 judge 结果。

`verl.step` 用于定位训练步数；`verl.rollout_index` 用于区分同一 prompt 的多条采样。

## 统计示例

正常 AC：

```text
episode.stop_reason = no_tool_call
judge.final_submit_verdict = accepted
metrics.acc = 1.0
metrics.acc_final = 1.0
metrics.acc_any = 1.0
behavior.has_tool_call_after_submit_accepted = false
```

曾经 AC 但后续又提交错：

```text
judge.final_submit_verdict != accepted
behavior.has_tool_call_after_submit_accepted = true
metrics.acc = 0.0
metrics.acc_final = 0.0
metrics.acc_any = 1.0
```

只通过 public tests 但没有正式提交：

```text
judge.final_submit_verdict = no_submission
metrics.acc = 0.0
```

response budget 耗尽：

```text
episode.stop_reason = response_length_exceeded
```

注意：verl 的 `response_length` budget 包含 assistant 生成 token 和 tool response observation token；但 `response_mask` 会把 tool response 标为 `0`，训练 loss 不计 tool response。
