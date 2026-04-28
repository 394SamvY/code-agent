# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

**OJ-like code agent**，只做强化学习（RL-only），不做 SFT。使用 Qwen3-8B 全参微调，GRPO 算法，veRL 框架 + SGLang rollout。目标是训练代码智能体在"公开测试调试 + 正式提交评测"两阶段交互中完成竞赛编程题。

## 硬约束

- **RL-only**：不要把 SFT 引入主训练路径
- 训练集 `CodeContests`，最终测试集 `LiveCodeBench`
- 两个固定工具：`run_public_tests`（调试，无reward）和 `submit_solution`（正式提交，主reward来源）
- `max_submissions=5` 是环境语义限制，`max_tool_calls` 只是工程 hard cap 防死循环
- reward 策略：accepted=1.0，failed=0.2 * private_pass_rate，public tests=0.0
- 不要回退到旧的 `execute_code`/`test_list`/函数补全协议
- 旧 MBPP/HumanEval 输出已归档到 `archive/legacy_outputs/2026-04-24/`

## 阅读顺序

在修改代码前，按顺序读：
1. `README.md`
2. `docs/env_protocol.md`
3. `docs/env_design_references.md`
4. `docs/verl_parquet_dataset_analysis.md`
5. `src/data/dataset.py`
6. `src/env/tools.py`
7. `src/env/sandbox.py`
8. `src/env/code_env.py`
9. `src/prompts.py`
10. `src/data/verl_dataset.py`
11. `src/verl_tools/oj_tools.py`
12. `src/reward.py`
13. `scripts/evaluate_baseline_with_verl.sh`
14. `src/eval/evaluate.py`

## 主链路架构

```
src/data/dataset.py       → CodeProblem/OJTestCase schema，加载 CodeContests/LiveCodeBench
src/data/verl_dataset.py  → 导出 verl Parquet（含 tools_kwargs）
src/env/sandbox.py        → 子进程 stdin/stdout 代码执行
src/env/tools.py          → 两工具语义、judge、observation、reward policy
src/env/code_env.py       → 单道题的 episode 封装
src/prompts.py            → agentic prompt 构造
src/verl_tools/oj_tools.py→ verl BaseTool 适配层（RunPublicTestsTool/SubmitSolutionTool）
src/reward.py             → verl 自定义 reward 函数（compute_score，返回 score 和 acc）
src/verl_dataset_adapter.py → RLHFDataset 子类，解码 Parquet 中 JSON 字符串字段
scripts/evaluate_baseline_with_verl.sh → 主评测入口（复用 verl main_ppo validation 路径）
src/eval/evaluate.py      → 轻量本地 debug harness（非主评测路径）
```

## 常用命令

### 测试

```bash
python3 tests/test_verl_tools.py
python3 tests/test_dataset_protocol.py
python3 tests/test_e2e_protocol.py
# 编译检查
python3 -X pycache_prefix=/tmp/code-agent-pycache -m compileall src tests
```

### 生成 verl 数据与配置

```bash
# 生成 Parquet 数据（在实际训练服务器上运行）
python3 -m src.data.verl_dataset --data_dir /root/autodl-tmp/datasets
# 从 TOOLS_SCHEMA 生成 tool_config.yaml
python3 -m src.env.tools
```

### 评测

```bash
# 主评测入口（复用 verl validation 路径）
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
bash scripts/evaluate_baseline_with_verl.sh livecodebench_test

# 常用覆盖参数
VAL_MAX_SAMPLES=8 bash scripts/evaluate_baseline_with_verl.sh codecontests_test
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=2048 \
  bash scripts/evaluate_baseline_with_verl.sh livecodebench_test

# 轻量本地 debug（非主评测路径，需先启动 SGLang 服务）
python3 -m src.eval.evaluate --model /root/autodl-tmp/models/Qwen3-8B \
  --datasets livecodebench --mode multi_turn --max_samples 1
```

### 训练

```bash
# 准备数据 + 生成 tool config + 启动 GRPO 训练
bash scripts/prepare_verl_training.sh          # 默认 2 GPU
bash scripts/prepare_verl_training.sh 4         # 指定 GPU 数量

# 一键启动（GPU 监控 + 训练）
bash scripts/start_training.sh
```

## 配置要点

- 配置文件：`configs/verl/grpo_qwen3_8b.yaml`
- `max_prompt_length=1024` 适合快速 baseline，`2048` 覆盖更好但吃显存
- `response_length=1024` 是 multi-turn 整条 trajectory 的总 response budget
- 当前训练集 9,698 条，`train_batch_size=128` + `rollout.n=4`，每 epoch 约 75 step
- 第一版 baseline 建议 `total_epochs=1`（1 epoch 的 rollout 量已远超旧实验 7 epoch）
- 目标硬件：2x A800-80G

## 数据文件

| 文件 | 用途 | 行数 |
| --- | --- | ---: |
| `data/verl/codecontests_train.parquet` | GRPO 训练 | 9,698 |
| `data/verl/codecontests_valid.parquet` | 训练过程 validation | 500 |
| `data/verl/codecontests_test.parquet` | CodeContests held-out test | 500 |
| `data/verl/livecodebench_test.parquet` | 最终泛化评测 | 611 |

Parquet 中 `prompt`、`reward_model`、`extra_info` 字段是 JSON 字符串，由 `OJLikeRLHFDataset`（`src/verl_dataset_adapter.py`）在加载时解码。
