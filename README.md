# code-agent

这是一个 **OJ-like code agent** 项目，只做强化学习，不做 SFT。

项目目标是训练一个能够在统一在线评测环境中解竞赛编程题的代码 Agent。它不是依赖 one-shot 代码生成，也不是围绕旧的函数补全 benchmark 打转，而是通过“公开测试调试 + 正式提交全量评测”的交互闭环完成写码、调试、提交与修复。

## 项目目标

这个项目当前形态如下：

- 模型读取一道编程题，生成完整可执行代码
- 它面对的是 OJ-like 环境，而不是一个泛化的执行代码工具
- 它先通过公开测试动作验证基本正确性
- 再通过正式提交动作跑 full judge，并根据返回结果继续修复

这是一个以 RL 为核心的项目：

- `RL-only`
- 不引入 SFT 训练链路
- 不把 benchmark 专属 prompt trick 当作主策略

## 数据集策略

当前只保留两个核心数据集：

- `CodeContests`：用于训练
- `LiveCodeBench`：用于最终测试

这样选择的原因是：

- `CodeContests` 体量足够，且协议天然接近竞赛编程环境
- `LiveCodeBench` 是更现代、更强的最终 benchmark，也和 OJ-like 交互逻辑对齐

仓库会把这两个数据集统一映射到内部同一套 schema 中，使训练和评测共享同一个环境协议。

## 环境设计

环境已经固定为两类工具：

- `run_public_tests`：运行公开 stdin/stdout 测试，只提供调试反馈
- `submit_solution`：运行 private/full judge，消耗一次正式提交

交互规则：

- train / val / test 共用同一套环境协议
- 不同 split 的错误格式和交互语义保持一致
- 环境语义只限制 `max_submissions=5`
- `max_tool_calls` 只是 rollout 工程保护，用来防止死循环，不是 OJ 规则
- `run_public_tests` 不给 reward，只返回 observation
- reward 以正式提交为主：accepted 为主奖励，failed submit 可按 private pass rate 给弱 shaped reward
- 支持题目级时间限制
- 内存限制后置，不作为第一阶段重点

执行环境保持轻量：

- 尽量直接使用主进程 Python 运行子进程代码
- 尽量避免引入重第三方依赖的 benchmark 设计

## 当前实现状态

当前 e2e 主链路已经接到 OJ-like v1 协议：

- `CodeProblem` 统一题目 schema
- `run_public_tests` / `submit_solution` 两动作环境协议
- `CodeContests` 导出 `codecontests_train/valid/test` parquet
- `LiveCodeBench` 用作最终 eval/test 数据
- 主评测入口复用 verl validation，尽量和训练 rollout / tool / reward 保持一致
- `src.eval.evaluate` 只保留为轻量本地 debug harness
- 旧的 `execute_code` / `test_list` / 函数补全协议已经退出主链路
- 旧 MBPP/HumanEval 输出已归档到 `archive/legacy_outputs/2026-04-24/`

当前四个 verl Parquet 文件已经准备好，可用于第一版 baseline：

| 文件 | 用途 | 行数 |
| --- | --- | ---: |
| `data/verl/codecontests_train.parquet` | GRPO 训练集 | 9,698 |
| `data/verl/codecontests_valid.parquet` | 训练过程 validation | 500 |
| `data/verl/codecontests_test.parquet` | CodeContests held-out test | 500 |
| `data/verl/livecodebench_test.parquet` | LiveCodeBench final/generalization eval | 611 |

其中 CodeContests 的 valid/test 已从 train 中固定抽样补到 500 条，三个 CodeContests split 的 `task_id` 两两无交集。LiveCodeBench 文件较大，主要因为 hidden/private tests 的 input/output 很大，不是因为 prompt 很长。

详细数据快照、schema、token 长度、文件大小和 baseline 建议见 `docs/verl_parquet_dataset_analysis.md`。

当前已经冻结的 v1 数据 schema 方向是：

- 保存原始题面 `problem_statement`
- 单独保留 `starter_code`
- 使用结构化 `public_tests` / `private_tests`
- 顶层保留 `time_limit_seconds`，内存限制下沉到 `metadata["memory_limit_bytes"]`
- 参考解统一放在 `reference_solutions`

当前 schema 的 source of truth 在 `src/data/dataset.py`。

环境协议的冻结说明在 `docs/env_protocol.md`，实现入口是 `src/env/tools.py`。
相近项目的环境设计参考和后续生产化 checklist 在 `docs/env_design_references.md`。
当前四个 Parquet 文件的分析记录在 `docs/verl_parquet_dataset_analysis.md`。

## Baseline 状态

当前阶段先测一个可复现 baseline，数据先不继续扩展。

历史小数据实验使用 `374` 条训练样本跑 `7` epoch；现在训练集是 `9,698` 条。在当前配置下：

```text
train_batch_size = 128
rollout.n = 4
每 step = 128 × 4 = 512 条 rollout
1 epoch ≈ floor(9698 / 128) = 75 steps
```

所以现在 1 epoch 约等于：

```text
75 × 512 = 38,400 条 rollout
```

已经明显大于旧实验完整 7 epoch 的 rollout 数量。第一版 baseline 建议先跑 `total_epochs=1`，不要直接沿用旧配置里的 `total_epochs=7`。

prompt length 方面：

- `max_prompt_length=512` 明显偏小，会覆盖不了大量 CodeContests prompt
- `max_prompt_length=1024` 更适合第一版快速 baseline
- `max_prompt_length=2048` 覆盖更好，但全参 GRPO 在 2xA800 80GB 上可能需要把 micro batch 降到 1
- `response_length=1024` 是 multi-turn 整条 trajectory 的总 response budget，不是每一轮 assistant 的单轮上限

## 常用命令

准备 verl 数据。推荐在实际训练服务器上生成，而不是依赖本地旧 parquet：

```bash
python3 -m src.data.verl_dataset \
  --data_dir /root/autodl-tmp/datasets
```

标准输出文件是：

- `data/verl/codecontests_train.parquet`
- `data/verl/codecontests_valid.parquet`
- `data/verl/codecontests_test.parquet`
- `data/verl/livecodebench_test.parquet`

同步生成 verl tool config：

```bash
python3 -m src.env.tools
```

用 verl validation 跑和训练一致的 tool / rollout / reward 链路：

```bash
bash scripts/evaluate_with_verl.sh livecodebench_test /root/autodl-tmp/models/Qwen3-8B 1
```

单卡 baseline 评测入口。一次只测一个 parquet，不会自动重新生成数据：

```bash
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
bash scripts/evaluate_baseline_with_verl.sh livecodebench_test
```

常用调试参数：

```bash
VAL_MAX_SAMPLES=8 bash scripts/evaluate_baseline_with_verl.sh codecontests_test
VAL_BATCH_SIZE=2 MAX_PROMPT_LENGTH=2048 bash scripts/evaluate_baseline_with_verl.sh livecodebench_test
```

可选：用轻量本地 harness 做快速调试。这个入口不作为主评测路径：

```bash
python3 -m src.eval.evaluate \
  --model /root/autodl-tmp/models/Qwen3-8B \
  --datasets livecodebench \
  --mode multi_turn \
  --max_samples 1
```

说明：

- `scripts/evaluate_with_verl.sh` 复用 `verl` 的 `main_ppo` validation 路径，行为更接近训练时 rollout
- `scripts/evaluate_baseline_with_verl.sh` 是当前单卡 baseline 入口，默认模型路径为 `/root/autodl-tmp/models/Qwen3-8B`
- `src.eval.evaluate` 只保留为轻量、本地可调试的评测 harness
- `verl/trainer/main_eval.py` 不是在线工具评测入口；它只对已经生成好的 responses 做离线 reward 打分

本地协议测试：

```bash
python3 tests/test_verl_tools.py
python3 tests/test_dataset_protocol.py
python3 tests/test_e2e_protocol.py
python3 -X pycache_prefix=/tmp/code-agent-pycache -m compileall src tests
```

## 仓库结构

当前主要代码区域如下：

```text
src/
  data/          数据集加载、OJ schema、verl parquet 导出
  env/           OJ judge、两工具协议、sandbox
  eval/          LiveCodeBench / CodeContests 评测入口
  verl_tools/    verl BaseTool 适配层
scripts/         训练、评测、数据准备入口
configs/verl/    verl 训练与工具配置
```

当前重点相关文件包括：

- `docs/env_protocol.md`
- `docs/env_design_references.md`
- `src/data/dataset.py`
- `src/data/verl_dataset.py`
- `src/env/tools.py`
- `src/env/sandbox.py`
- `src/env/code_env.py`
- `src/verl_tools/oj_tools.py`
- `scripts/evaluate_with_verl.sh`
- `src/eval/evaluate.py`

历史输出和旧实验记录在：

- `archive/legacy_outputs/2026-04-24/`
- `archive/`

## 快速开始

建议按下面顺序理解项目：

- 先读 `AGENTS.md`，了解当前协作规范和项目硬约束
- 再看 `src/data/`、`src/env/`、`src/eval/` 里的主链路代码
- 训练与评测入口集中在 `scripts/` 下

顶层文档分工：

- `README.md`：面向外部读者，解释项目目标、方法与当前方向
- `AGENTS.md`：面向后续协作者，规定实现优先级与当前约束
