# Agent Guide

这个仓库正在构建一个 **OJ-like code agent**，并且只做强化学习，不做 SFT。

## 开工前先读什么

在修改代码之前，按下面顺序阅读：

1. `README.md`
2. `src/data/dataset.py`
3. `src/env/tools.py`
4. `src/env/sandbox.py`
5. `src/eval/evaluate.py`
6. `src/verl_tools/`

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
- 顶层规范文档只维护 `README.md` 和 `AGENTS.md`
- `src/data/dataset.py` 中的 v1 schema 是当前数据协议的 source of truth

## 实现优先级

实现顺序按下面走：

1. 统一 OJ 风格题目 schema
2. 统一训练和评测阶段的 judge 行为与返回格式
3. 用两动作 OJ 协议替换当前单工具假设
4. 支持题目级时间限制
5. 围绕 `CodeContests` 和 `LiveCodeBench` 打通完整训练/评测链路

如果出现取舍，优先保证环境协议一致性，而不是兼容旧 benchmark 习惯。

当前阶段已经接通 OJ-like v1 的数据导出和评测主链路；旧评测链路、旧 verl 数据导出链路不再兼容。

## 不要继续旧方向

不要把项目重新带回旧路线：

- 不要恢复以前那套函数 benchmark 为中心的路线
- 不要重新引入重依赖 benchmark 选择
- 不要在没有新决策的前提下扩展当前主线训练数据集
- 不要把旧的单一执行代码接口当作目标环境
- 不要再以旧 run 指标和旧 benchmark 目标作为当前项目目标

`outputs/` 和 `archive/` 下的内容都只是历史产物，不是当前规范。

## 当前默认假设

- 两个数据集最终会统一到一套内部题目格式
- train / val / test 共用一套环境协议
- 代码执行尽量保持轻量，并尽量跟随主进程 Python
- 时间限制属于近期实现目标
- 内存限制明确后置
