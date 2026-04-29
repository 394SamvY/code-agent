# OJ-like 环境设计参考

本文档记录对当前 OJ-like v1 环境设计有参考价值的外部系统。
`docs/specs/env_protocol.md` 仍然是协议 source of truth；本文档只提供设计背景和取舍依据。

## 当前结论

当前环境形态适合作为第一阶段 RL 环境：

- action space 保持小而明确：`run_public_tests` 和 `submit_solution`。
- 模型输出完整的 stdin/stdout Python 程序。
- public tests 是诊断信息，不是 reward 来源。
- formal submission 是主要 correctness signal。
- `max_submissions` 是 OJ 规则，`max_tool_calls` 是 rollout 保护。
- judge results 先保存为结构化数据，再由结构化数据格式化 observation text。

当前主要缺口不是两工具协议，而是 sandbox 强度和 strict benchmark feedback mode。
如果进入大规模训练，轻量 subprocess runner 最终应该替换或封装为 Docker、cgroups、Judge0-like infrastructure、SandboxFusion-like infrastructure，或者其他带 CPU、memory、filesystem、network、output limits 的隔离 runner。

## Reference Map

| Project | Relevant design | Implication for this repo |
| --- | --- | --- |
| LiveCodeBench | Contest-style code generation，包含 hidden tests、release dates，以及基于 test feedback 的 self-repair 场景。 | 适合作为最终 eval 目标。应明确保留 public/private tests，并在可用时保留 time-window metadata。 |
| CodeContests / AlphaCode | 竞赛编程题，包含 paired inputs/outputs、public/private/generated tests 和 language-specific reference solutions。 | 适合作为 train/valid/test 来源。v1 将 generated tests 合并进 private judge，并过滤 unsupported I/O。 |
| SWE-agent | 强调 Agent-Computer Interface 设计：窄而明确的 commands 和 structured observations 会显著影响 agent 行为。 | 应优先使用两个 OJ actions，而不是 generic `execute_code` tool。环境接口本身就是学习问题的一部分。 |
| SWE-bench | 使用 Docker-based、reproducible evaluation harness：应用 prediction、运行 tests、输出 structured results/logs。 | production 阶段应超出本地 subprocess execution，把 judge logs/artifacts 作为一等产物。 |
| Terminal-Bench / Harbor | 将 task dataset、execution harness、sandbox、verifier、leaderboard scoring 分开，服务 long-horizon terminal agents。 | data、tool execution、reward、eval reporting 应保持分层。`max_tool_calls` 属于 harness engineering，不属于 task semantics。 |
| InterCode | 用 action-observation loop 评测带 execution feedback 的 interactive coding。 | multi-turn repair 是合理目标，但 action space 应和任务领域保持一致。 |
| CodeRL / RLTF-style work | 使用 compiler/unit-test feedback 和 RL signals 做 program synthesis。 | execution feedback 有价值，但 reward 不应过度激励 public tests，也不应泄漏 benchmark-only signals。 |
| Judge0 | 成熟 online judge API 暴露 stdin、expected output、stdout/stderr、status、time、memory 和 configurable limits。 | 当前 result schema 和真实 OJ 字段对齐；memory/output/process limits 是后续 production 要求。 |
| SandboxFusion / E2B-style sandboxes | 提供隔离 code execution APIs，支持 language/runtime selection、timeout、stdout/stderr 和 error reporting。 | 可作为未来 runner abstraction 参考；当前 v1 协议应保持独立于具体 runner backend。 |

## 当前采纳

小 action surface：

- `run_public_tests(code)` 用于 diagnosis
- `submit_solution(code)` 用于 full judge

Structured observations：

- per-case verdict、input、expected output、stdout、stderr、return code、timeout flag、runtime
- judge-level verdict、passed count、total count、first failed case，以及必要时的 full case metadata

Submit-dominant reward：

- public tests 返回 `0.0`
- accepted submit 返回主 reward
- failed submit 可使用弱 passed/total shaping；当前 judge 首错即停，因此这是首个失败前的前缀比例

Dataset separation：

- `CodeProblem` 是内部 problem protocol
- `problem_statement` 保持 raw
- prompt construction 位于 data loading 之上
- `verl_dataset.py` 序列化 `create_kwargs` 给 tool construction 使用，而不是输出 legacy benchmark fields

## 当前避免

把 generic code execution 当主工具：

- generic `execute_code` tool 容易鼓励 arbitrary probing，而不是 OJ behavior。
- 它也会重新带回旧的 `test_list` / function-completion assumptions。

奖励 public-test repetition：

- public tests 是有用反馈，但 public pass rate 不应该成为 objective。
- 重复成功运行 public tests 不应该产生额外 reward。

混淆 task semantics 和 rollout engineering：

- `max_submissions` 是 OJ rule。
- `max_tool_calls` 只是防 dead loops 的 hard cap。

过度绑定当前 runner implementation：

- 协议应能承受 runner 替换：从 local subprocess execution 换成 Docker、Judge0、SandboxFusion 或 remote judge 时，主协议不应变化。

## Open Production Decisions

进入 large-scale 或 benchmark-facing runs 前，需要决策：

1. final eval 是否隐藏 private failed input/output，只返回 verdict 和 counts。
2. untrusted generated code 使用哪种更强的 sandbox backend。
3. v1.1 是否加入 memory limits、output limits、process limits、network/file isolation。
4. 继续 Python-only training，还是加入 language metadata 和 multi-language runners。
5. generated tests 继续合并进 private tests，还是在 metadata 中单独记录用于分析。
6. 如何在 local laptop、training server、remote judge hardware 之间归一化 time limits。

## Sources

- LiveCodeBench: https://livecodebench.github.io/
- LiveCodeBench dataset card: https://huggingface.co/datasets/livecodebench/code_generation
- CodeContests dataset card: https://huggingface.co/datasets/Imandra/code_contests
- CodeContests sandboxed Harbor repackaging: https://huggingface.co/datasets/open-thoughts/CodeContests
- SWE-agent ACI docs: https://swe-agent.com/latest/background/aci/
- SWE-bench harness docs: https://www.swebench.com/SWE-bench/reference/harness/
- Terminal-Bench repository: https://github.com/harbor-framework/terminal-bench
- InterCode: https://intercode-benchmark.github.io/
- CodeRL paper page: https://huggingface.co/papers/2207.01780
- Judge0 docs: https://ce.judge0.com/docs
- SandboxFusion repository: https://github.com/bytedance/SandboxFusion
- E2B sandbox docs: https://e2b.dev/docs/sdk-reference/code-interpreter-python-sdk/v2.0.0/sandbox
