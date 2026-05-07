# Agent Guide

## Python 环境与依赖

- 本地环境使用当前项目下的 `.venv`，例如 `.venv/bin/python3 ...`；远程环境直接使用 `python3`。远程环境的 PyTorch、verl、SGLang 等重依赖已经安装好；本地环境没有这些重依赖，但可以参考 `/Users/yang/code/verl` 下拉取好的 verl 源码。

## 文档写作风格

- 写文档要简明扼要，交代清楚前因后果；不要堆砌背景、流水账或重复已有文档内容。

## 仓库随附 Codex skills

仓库内保留两个可版本化的 Codex user skill，供远程环境没有本地 skill 时使用：

- `skills/debug-cleanup/SKILL.md`
- `skills/project-context-sync/SKILL.md`

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
- `scripts/evaluate_baseline_with_verl.sh` 是当前 baseline 评测入口，复用 verl `main_ppo` validation 路径

当前 baseline 调试进展、验证记录、blocker 和下一步统一维护在 `docs/project_status.md`。
