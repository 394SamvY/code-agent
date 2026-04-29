# OJ-like 环境协议 v1

本文档冻结当前 OJ-like code agent 的 v1 环境协议。
实现层的 source of truth 是 `src/env/tools.py`；本文档解释协议语义和设计边界。

## Problem Contract

环境消费来自 `src/data/dataset.py` 的 `CodeProblem`。

v1 固定假设：

- 解答必须是完整的 Python stdin/stdout 程序。
- 测试用例使用结构化的 `OJTestCase(input, output)`。
- v1 只支持 stdin/stdout。
- File I/O、interactive problems、special judge、memory limits 不在 v1 范围内。
- `problem_statement` 保留原始题面；prompt 在数据层之上构造。

## Actions

环境只暴露两个 tools。

### `run_public_tests(code)`

用途：

- 用 public tests 调试。
- 在正式提交前提供详细反馈。

语义：

- 不消耗 submission attempt。
- 可以重复调用。
- 不终止 episode。
- 不给 reward。
- 返回详细 public case 反馈，包括 failed input、expected output、stdout、stderr 和 verdict。

### `submit_solution(code)`

用途：

- 运行 private/full judge。
- 提供主要 correctness signal。

语义：

- 消耗一次 submission attempt。
- 默认限制是 `max_submissions=5`。
- accepted 时终止。
- submission attempts 用尽时终止。
- v1 中失败时返回第一条 failed private case 的详细信息。

当前 training/eval feedback policy 有意暴露第一条 failed private case，用来支持 repair learning。
之后如果需要更严格的 benchmark mode，可以隐藏 private input/output，只返回 verdict 和计数。

## Verdicts

当前 verdict 集合：

- `accepted`
- `wrong_answer`
- `runtime_error`
- `time_limit_exceeded`
- `syntax_error`
- `no_tests`
- `submission_limit_exceeded`

未来可能加入、但 v1 不实现的 verdict：

- `presentation_error`
- `output_limit_exceeded`
- `memory_limit_exceeded`

## Result Schema

单个 case result 使用：

- `index`
- `passed`
- `verdict`
- `input`
- `expected`
- `stdout`
- `stderr`
- `returncode`
- `timed_out`
- `runtime_seconds`

单个 judge result 使用：

- `action`
- `verdict`
- `passed`
- `total`
- `first_failed`
- `tests`

结构化结果是 logging、metrics 和 reward 的主接口。Observation text 只是给模型看的展示层。

## Output Comparison

v1 输出比较规则：

- 统一换行符为 `\n`。
- 比较 `stdout.rstrip()` 和 `expected.rstrip()`。
- 不做 special judge。
- 不做 numeric tolerance。
- 除 trailing whitespace removal 外，不做 whitespace-insensitive comparison。

## Reward

Reward 以 submit 为主：

- `run_public_tests`: `0.0`
- `submit_solution` accepted: `1.0`
- failed `submit_solution`: 根据 private pass rate 给弱 shaped reward，当前由实现策略设置上限

这个设计是为了避免模型学会为了 reward 反复调用 public tests。Public tests 是诊断工具，不是优化目标。

## Termination

环境语义上的终止条件：

- `submit_solution` 返回 `accepted`
- `submit_solution` 返回 `submission_limit_exceeded`

工程层 rollout 保护：

- `max_tool_calls` 用于防止 infinite loops。
- 它不是 OJ rule，不应被当作任务语义。

如果 rollout 结束时没有任何 valid submission，则该 episode 视为失败。

## Sandbox

当前 v1 执行方式：

- 通过 `sys.executable` 启动 Python subprocess。
- stdin 来自 `OJTestCase.input`。
- 捕获并截断 stdout/stderr。
- 支持 per-case timeout。
- 执行后删除临时 `.py` 文件。

已知限制：

- 当前 process sandbox 很轻量，但不是强隔离。
- 没有 memory limit enforcement。
- 除 subprocess 边界外，没有额外 network 或 filesystem hardening。

未来 production training 应考虑 Docker 或 remote sandboxing，并补上 CPU、memory、filesystem、network、output limits。

## Dataset Policy

CodeContests:

- 用于 train/valid/test，其中 train/valid 进入训练配置，test 作为同分布 held-out eval。
- 过滤 File I/O problems。
- 过滤没有 Python reference solutions 的问题。
- `generated_tests` 合并进 `private_tests`。

LiveCodeBench:

- `test` 用于最终 eval/test。
- `public_test_cases` 映射到 `public_tests`。
- `private_test_cases` 映射到 `private_tests`。

## Compatibility Notes

不要重新引入 legacy contract：

- 不恢复 `execute_code` tool
- 不把 `test_list[str]` 当主协议
- 不恢复 `entry_point`
- 不恢复 function-completion benchmark assumptions

v1 agent 写完整 stdin/stdout 程序。
