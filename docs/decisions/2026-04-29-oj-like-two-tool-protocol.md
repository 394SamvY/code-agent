# 决策：采用 OJ-like 两工具环境协议

日期：2026-04-29

## 决策

环境固定为两个工具：

- `run_public_tests`
- `submit_solution`

模型输出完整 Python stdin/stdout 程序。`run_public_tests` 只提供公开测试反馈，不给 reward；`submit_solution` 运行 private/full judge，是主要 reward 来源。

## 背景

项目目标是 OJ-like code agent，而不是函数 benchmark agent。真实竞赛编程的关键循环是：

1. 阅读题面。
2. 编写完整程序。
3. 用公开样例和公开测试调试。
4. 正式提交到 full judge。
5. 根据提交反馈修复。

因此环境接口本身应该表达 OJ 行为，而不是暴露一个泛化的代码执行工具。`max_submissions=5` 是环境语义；`max_tool_calls` 只是 rollout 工程保护，不应被当成任务规则。

## 被拒绝的方案

- **恢复 `execute_code`**：这个工具太泛化，会鼓励任意 probe，把环境拉回旧函数 benchmark 和本地执行器习惯。
- **恢复 `test_list` / `entry_point` / 函数补全协议**：这些字段服务旧 benchmark，不符合当前完整 stdin/stdout 程序目标。
- **让 public tests 产生 reward**：会鼓励模型重复调用 public test 或优化 public pass rate，而不是正式提交正确性。
- **把 `max_tool_calls` 写成 OJ 规则**：它只是防止 rollout 死循环的 hard cap，不是题目语义。

## 后果

- data、prompt、tool adapter、reward 和 eval 都应围绕 `run_public_tests` / `submit_solution` 维护。
- 新增数据集时必须映射到 `CodeProblem`、`OJTestCase`、public/private tests 和完整 stdin/stdout 程序。
- reward 以 submit 为主：accepted 为 1.0，failed submit 最多给弱 shaped reward；当前 judge 首错即停，因此按首个失败前的 passed/total 前缀比例计算。
- 修改协议时，先更新 `docs/specs/env_protocol.md` 和实现，再同步相关 tests。

## 相关文档

- `docs/specs/env_protocol.md`
- `docs/references/env_design_references.md`
- `src/env/tools.py`
- `src/env/code_env.py`
- `src/verl_tools/oj_tools.py`
