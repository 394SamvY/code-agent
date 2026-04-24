# code-agent

这是一个 **OJ-like code agent** 项目，只做强化学习，不做 SFT。

项目目标是训练一个能够在统一在线评测环境中解竞赛编程题的代码 Agent。它不是依赖 one-shot 代码生成，也不是围绕旧的函数补全 benchmark 打转，而是通过“公开测试调试 + 正式提交全量评测”的交互闭环完成写码、调试、提交与修复。

## 项目目标

这个项目正在收敛成如下形态：

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

环境已经明确收敛到两类工具形态：

- 公开测试调试动作
- 正式提交 full judge 动作

虽然工具最终命名还没有冻结，但交互模型已经固定：

- train / val / test 共用同一套环境协议
- 不同 split 的错误格式和交互语义保持一致
- 需要支持题目级时间限制
- 内存限制后置，不作为第一阶段重点

执行环境保持轻量：

- 尽量直接使用主进程 Python 运行子进程代码
- 尽量避免引入重第三方依赖的 benchmark 设计

## 当前实现方向

当前 e2e 主链路已经接到 OJ-like v1 协议：

- `CodeProblem` 统一题目 schema
- `run_public_tests` / `submit_solution` 两动作环境协议
- `CodeContests` 导出 verl train/val parquet
- `LiveCodeBench` 用作最终 eval/test 数据
- one-shot 和 multi-turn 评测都通过同一套 private judge 判定

当前已经冻结的 v1 数据 schema 方向是：

- 保存原始题面 `problem_statement`
- 单独保留 `starter_code`
- 使用结构化 `public_tests` / `private_tests`
- 顶层保留 `time_limit_seconds`，内存限制下沉到 `metadata["memory_limit_bytes"]`
- 参考解统一放在 `reference_solutions`

当前 schema 的 source of truth 在 `src/data/dataset.py`。

## 常用命令

准备 verl 数据：

```bash
python3 -m src.data.verl_dataset --data_dir /root/autodl-tmp/datasets
```

同步生成 verl tool config：

```bash
python3 -m src.env.tools
```

评测 LiveCodeBench：

```bash
python3 -m src.eval.evaluate \
  --model /root/autodl-tmp/models/Qwen3-8B \
  --datasets livecodebench \
  --mode multi_turn \
  --max_samples 1
```

本地协议测试：

```bash
python3 tests/test_verl_tools.py
python3 tests/test_dataset_protocol.py
python3 tests/test_e2e_protocol.py
```

## 仓库结构

当前主要代码区域如下：

```text
src/
  data/          数据集加载与内部题目 schema
  env/           执行环境、工具接口与 sandbox
  eval/          评测入口
  verl_tools/    verl 工具适配层与 reward 侧衔接
scripts/         训练 / 评测 / 数据准备入口
configs/verl/    verl 配置
```

当前重点相关文件包括：

- `src/data/dataset.py`
- `src/data/verl_dataset.py`
- `src/env/tools.py`
- `src/env/sandbox.py`
- `src/eval/evaluate.py`
- `src/verl_tools/`

## 快速开始

建议按下面顺序理解项目：

- 先读 `AGENTS.md`，了解当前协作规范和项目硬约束
- 再看 `src/data/`、`src/env/`、`src/eval/` 里的主链路代码
- 训练与评测入口集中在 `scripts/` 下

顶层文档分工：

- `README.md`：面向外部读者，解释项目目标、方法与当前方向
- `AGENTS.md`：面向后续协作者，规定实现优先级与当前约束
