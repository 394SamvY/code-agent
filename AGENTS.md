# Agent Guide

这个仓库正在构建一个 **OJ-like code agent**，并且只做强化学习，不做 SFT。

## 开工前先读什么

在修改代码之前，按下面顺序阅读：

1. `README.md`
2. `docs/project_status.md`
3. `docs/specs/env_protocol.md`
4. `docs/references/env_design_references.md`
5. `docs/specs/verl_parquet_dataset_analysis.md`
6. `src/data/dataset.py`
7. `src/env/tools.py`
8. `src/env/sandbox.py`
9. `src/env/code_env.py`
10. `src/prompts.py`
11. `src/data/verl_dataset.py`
12. `src/verl_tools/oj_tools.py`
13. `src/reward.py`
14. `scripts/evaluate_baseline_with_verl.sh`

如果需要历史背景，可以查看 `obsidian/11-code-agent/`，但那个目录只是研究记录，不是当前仓库规范的来源。

## 仓库随附 Codex skills

仓库内保留两个可版本化的 Codex user skill，供远程环境没有本地 skill 时使用：

- `skills/debug-cleanup/SKILL.md`
- `skills/project-context-sync/SKILL.md`

如果当前 Codex 运行环境不会自动发现仓库内 `skills/`，可以把它们复制到该机器的 `~/.codex/skills/` 后再使用：

```bash
mkdir -p ~/.codex/skills
cp -R skills/debug-cleanup ~/.codex/skills/
cp -R skills/project-context-sync ~/.codex/skills/
```

## 当前硬约束

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
- `submit_solution` 是主 reward 来源：accepted 为 1.0，failed submit 最多按首错前 passed/total 前缀比例给弱 shaped reward
- `README.md` 是人类入口，`AGENTS.md` 是 agent 操作规范，`docs/project_status.md` 是当前进度和交接入口
- `CLAUDE.md` 等工具专属文档只保留工具差异和公共规范链接，不复制完整项目规范
- 环境协议说明维护在 `docs/specs/env_protocol.md`
- 相近项目参考和生产化 checklist 维护在 `docs/references/env_design_references.md`，但不替代协议文档
- 长期决策记录维护在 `docs/decisions/`，改方向前先读相关 decision
- `src/data/dataset.py` 中的 v1 schema 是当前数据协议的 source of truth
- `data/verl` 应在实际训练服务器上由 `python3 -m src.data.verl_dataset` 重新生成，不要信任旧 parquet 缓存
- `data/verl` 当前标准文件是 `codecontests_train.parquet`、`codecontests_valid.parquet`、`codecontests_test.parquet`、`livecodebench_test.parquet`
- 当前四个标准 Parquet 已经完成一次本地整理，可作为第一版 baseline 输入；详细快照见 `docs/specs/verl_parquet_dataset_analysis.md`

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

如果出现取舍，优先保证环境协议一致性，而不是兼容旧 benchmark 习惯。

当前阶段已经接通 OJ-like v1 的数据导出和评测主链路；旧评测链路、旧 verl 数据导出链路不再兼容。

## 当前数据和 baseline 状态

截至 2026-04-25，当前 `data/verl/` 下四个标准文件状态如下：

| 文件 | 用途 | 行数 | 备注 |
| --- | --- | ---: | --- |
| `codecontests_train.parquet` | GRPO 训练 | 9,698 | 从原 train 中抽出部分样本补充 valid/test 后的训练集 |
| `codecontests_valid.parquet` | 训练过程 validation | 500 | 原 valid 95 条 + train 抽样 405 条 |
| `codecontests_test.parquet` | CodeContests held-out test | 500 | 原 test 122 条 + train 抽样 378 条 |
| `livecodebench_test.parquet` | 最终泛化评测 | 611 | 文件很大，主要因为 hidden/private tests 很大 |

当前四个文件共用 verl schema：

- `data_source`
- `prompt`
- `ability`
- `reward_model`
- `extra_info`

`prompt`、`reward_model`、`extra_info` 在 Parquet 中都是 JSON string。`extra_info` 保存每道题的 tool 创建参数，特别是 public/private tests、time limit 和 `max_submissions`。

当前数据处理目标已经基本完成，下一步优先级是跑出可复现 baseline，而不是继续改数据。建议第一版 baseline 使用 1 epoch，不要沿用旧小数据实验里的 `total_epochs=7`。当前 `train_batch_size=128`、`rollout.n=4` 时，每个 step 是 512 条 rollout；`9698` 条训练数据约等于每 epoch 75 step，1 epoch 的 rollout 数量已经约为旧 374 条训练集跑 7 epoch 的 5.4 倍。

当前 baseline 调试进展、验证记录、blocker 和下一步统一维护在 `docs/project_status.md`。不要把阶段性状态分散写进新的临时入口文档。

prompt 长度方面：

- `max_prompt_length=512` 明显太小。
- `max_prompt_length=1024` 适合快速 baseline。
- `max_prompt_length=2048` 数据覆盖更好，但 2xA800 80GB 全参 GRPO 可能需要降低 micro batch。
- `response_length=1024` 是 multi-turn 整条 trajectory 的总 response budget，不是单轮 assistant 的 1024 tokens。

## 不要继续旧方向

不要把项目重新带回旧路线：

- 不要恢复以前那套函数 benchmark 为中心的路线
- 不要重新引入重依赖 benchmark 选择
- 不要在没有新决策的前提下扩展当前主线训练数据集
- 不要把旧的单一执行代码接口当作目标环境
- 不要再以旧 run 指标和旧 benchmark 目标作为当前项目目标

`docs/legacy/2026-04-24-legacy-outputs/` 是旧 MBPP/HumanEval 和旧 GRPO 输出归档，只作历史参考。`outputs/` 是当前新运行的写入位置，不是规范来源。

`/Users/yang/code/verl/verl/trainer/main_eval.py` 只是对已有 responses 做离线 reward 打分，不是当前 OJ-like 在线工具评测入口。需要复用 verl 训练时 agent loop 时，优先走 `scripts/evaluate_baseline_with_verl.sh`。

## 当前默认假设

- 两个数据集已经统一到一套内部题目格式
- train / val / test 共用一套环境协议和 judge 结果格式
- 代码执行尽量保持轻量，并尽量跟随主进程 Python
- 时间限制已接入题目级配置
- 内存限制明确后置
