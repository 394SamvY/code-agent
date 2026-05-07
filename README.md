# code-agent

这是一个 **OJ-like code agent** 项目，目标是在竞赛编程式环境里训练和评测一个会写完整 stdin/stdout Python 程序的 agent。

环境固定为两类动作：

- `run_public_tests`：运行公开测试，用于调试。
- `submit_solution`：运行 full judge / private tests。

当前进度、blocker、最近 smoke 数据和下一步统一维护在 [docs/project_status.md](docs/project_status.md)。
