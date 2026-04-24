# Agent Guide

这个仓库正在构建一个 **OJ-like code agent**，并且只做强化学习，不做 SFT。

## 开工前先读什么

在修改代码之前，按下面顺序阅读：

1. `README.md`
2. `docs/env_protocol.md`
3. `docs/env_design_references.md`
4. `src/data/dataset.py`
5. `src/env/tools.py`
6. `src/env/sandbox.py`
7. `src/env/code_env.py`
8. `src/prompts.py`
9. `src/data/verl_dataset.py`
10. `src/verl_tools/oj_tools.py`
11. `src/reward.py`
12. `scripts/evaluate_with_verl.sh`
13. `src/eval/evaluate.py`

如果需要历史背景，可以查看 `obsidian/11-code-agent/`，但那个目录只是研究记录，不是当前仓库规范的来源。

## 当前硬约束

- `RL-only`：不要把 SFT 重新引回主训练路径
- 训练集是 `CodeContests`
- 最终测试集是 `LiveCodeBench`
- 项目目标是 OJ-like 环境中的 code agent，而不是函数 benchmark agent
- 环境固定为两类动作：
  - 公开测试调试动作
  - full judge 正式提交动作
- 工具命名固定为 `run_public_tests` 和 `submit_solution`
- 不要回退到旧的 `execute_code` / `test_list` / 函数补全协议
- 环境语义只限制 `max_submissions=5`
- `max_tool_calls` 只是工程 hard cap，用来防止 rollout 死循环，不是 OJ 规则
- `run_public_tests` 不给 reward，只返回 observation
- `submit_solution` 是主 reward 来源：accepted 为 1.0，failed submit 最多按 private pass rate 给弱 shaped reward
- 顶层规范文档只维护 `README.md` 和 `AGENTS.md`
- 环境协议说明维护在 `docs/env_protocol.md`
- 相近项目参考和生产化 checklist 维护在 `docs/env_design_references.md`，但不替代协议文档
- `src/data/dataset.py` 中的 v1 schema 是当前数据协议的 source of truth
- `data/verl` 应在实际训练服务器上由 `python3 -m src.data.verl_dataset` 重新生成，不要信任旧 parquet 缓存
- `data/verl` 当前标准文件是 `codecontests_train.parquet`、`codecontests_valid.parquet`、`codecontests_test.parquet`、`livecodebench_test.parquet`

## 当前主链路

当前主链路已经接到 OJ-like v1：

- `src/data/dataset.py` 定义 `CodeProblem` / `OJTestCase`，并加载 `CodeContests`、`LiveCodeBench`
- `src/env/tools.py` 定义 `run_public_tests` / `submit_solution`、verdict、observation、reward policy
- `src/env/sandbox.py` 负责 stdin/stdout 子进程执行
- `src/env/code_env.py` 把一道 `CodeProblem` 包成可交互环境
- `src/prompts.py` 从 `CodeProblem` 构造 one-shot / agentic prompt
- `src/data/verl_dataset.py` 导出四个显式 verl parquet，写入两工具 `create_kwargs`
- `src/verl_tools/oj_tools.py` 是 verl BaseTool 适配层
- `src/reward.py` 给 verl training / validation 暴露 `score` 和 `acc`
- `scripts/evaluate_with_verl.sh` 是当前主评测入口，复用 verl `main_ppo` validation 路径
- `src/eval/evaluate.py` 只保留为轻量本地 debug harness，不作为主评测路径

如果出现取舍，优先保证环境协议一致性，而不是兼容旧 benchmark 习惯。

当前阶段已经接通 OJ-like v1 的数据导出和评测主链路；旧评测链路、旧 verl 数据导出链路不再兼容。

## 不要继续旧方向

不要把项目重新带回旧路线：

- 不要恢复以前那套函数 benchmark 为中心的路线
- 不要重新引入重依赖 benchmark 选择
- 不要在没有新决策的前提下扩展当前主线训练数据集
- 不要把旧的单一执行代码接口当作目标环境
- 不要再以旧 run 指标和旧 benchmark 目标作为当前项目目标

`archive/legacy_outputs/2026-04-24/` 是旧 MBPP/HumanEval 和旧 GRPO 输出归档，只作历史参考。`outputs/` 是当前新运行的写入位置，不是规范来源。

`/Users/yang/code/verl/verl/trainer/main_eval.py` 只是对已有 responses 做离线 reward 打分，不是当前 OJ-like 在线工具评测入口。需要复用 verl 训练时 agent loop 时，优先走 `scripts/evaluate_with_verl.sh`。

## 当前默认假设

- 两个数据集已经统一到一套内部题目格式
- train / val / test 共用一套环境协议和 judge 结果格式
- 代码执行尽量保持轻量，并尽量跟随主进程 Python
- 时间限制已接入题目级配置
- 内存限制明确后置
