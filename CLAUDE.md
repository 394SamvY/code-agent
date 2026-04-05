# code-agent

用 GRPO 强化学习全量微调 Qwen3-8B 成为多轮代码 Agent，能够自主调用工具完成"写码 → 测试 → 调试"的迭代闭环。

## 项目结构

```
src/
  prompts.py              # 共用 prompt 模板（one-shot + agentic）
  env/                    # 核心执行环境
    code_env.py           #   CodeEnvironment：单道题的交互环境，统一管理状态
    tools.py              #   单工具 execute_code + TOOLS_SCHEMA
    sandbox.py            #   subprocess 沙箱，安全执行 Python 代码
  eval/                   # 评测
    evaluate.py           #   支持 one_shot / multi_turn / baseline 三种模式
  data/                   # 数据
    dataset.py            #   CodeProblem 数据结构 + MBPP/HumanEval/APPS 加载
    verl_dataset.py       #   CodeProblem → verl parquet 格式转换
  verl_tools/             # verl 训练工具层
    execute_code_tool.py  #   verl BaseTool 适配：执行代码+跑测试+calc_reward()
configs/verl/
  grpo_qwen3_8b.yaml     # verl GRPO 训练主配置（Qwen3-8B 全量微调，2×A800）
  grpo_qwen_7b.yaml      # 旧配置（Qwen2.5-Coder-7B LoRA，仅供参考）
  tool_config.yaml        # 工具类注册（映射到 verl_tools/）
scripts/
  train_verl.sh           # 训练入口
  evaluate.sh             # 评测入口
  convert_checkpoint.py   # FSDP checkpoint 转换（支持全量微调和 LoRA）
  prepare_resources.sh    # 模型/数据集下载（超算环境）
tests/
  test_verl_tools.py      # 工具逻辑 + reward 计算单元测试
```

## 核心架构

单工具设计，训练和评测共用同一个执行环境栈：

```
ExecuteCodeTool  或  evaluate.py
        ↓
  CodeEnvironment.execute_tool("execute_code", code=...)
        ↓
  tools.py (tool_execute_code: 语法检查 + 逐条跑测试)
        ↓
  sandbox.py (execute_with_tests)   ← subprocess 隔离执行
```

- **CodeEnvironment** 管理单道题的状态（当前代码、测试历史）
- **ExecuteCodeTool** 是 verl BaseTool 适配层，实现 create/execute/calc_reward/release
- 每次 tool call 独立的 create → execute → release 生命周期，无状态，天然适配 verl

## 模型和工具

- 基座模型：`Qwen3-8B`（本地路径 `/root/autodl-tmp/models/Qwen3-8B`）
- 训练方式：GRPO 全量微调（2×A800-80G，FSDP）
- Qwen3 思考模式：训练时禁用（`enable_thinking: false`）
- 单工具：`execute_code`（接收完整代码 → 语法检查 → 跑全部测试 → 返回 pass/fail + traceback）
- 工具调用格式：Qwen 原生 `<tool_call>...</tool_call>`

## Reward 设计

最终 reward 由 `ExecuteCodeTool.calc_reward()` 计算，2 个组分：

| 组分 | 范围 | 说明 |
|------|------|------|
| exec_reward | 0.0 ~ 1.0 | 最终一次尝试的 passed / total |
| fix_reward | 0 或 0.2 | 首次测试失败但最终修复通过 |

每次 execute 还有 step_reward（通过率 × 0.1），鼓励模型写出更好的代码。

## 常用命令

```bash
# 本地测试（不需要 GPU 和 verl）
python3 tests/test_verl_tools.py

# 准备 verl 训练数据（本地数据集，不下载 APPS）
python -m src.data.verl_dataset --data_dir /root/autodl-tmp/datasets --no_apps

# 训练（需要 GPU）
bash scripts/train_verl.sh              # 默认 2 GPU + Qwen3-8B 全量微调
bash scripts/train_verl.sh 2 grpo_qwen3_8b  # 显式指定

# 导出 checkpoint 为 HF 模型（全量微调）
python scripts/convert_checkpoint.py --mode full-merge \
    --ckpt_path outputs/verl_grpo/checkpoints/global_step_50 \
    --output_dir /root/autodl-tmp/models/Qwen3-8B-GRPO

# 评测
python -m src.eval.evaluate \
    --model /root/autodl-tmp/models/Qwen3-8B \
    --mode multi_turn \
    --datasets mbpp_test humaneval
```

## 依赖

核心依赖：`torch`, `transformers`, `datasets`, `verl`, `peft`, `accelerate`, `pandas`, `pyarrow`

当前环境版本：verl 0.7.1, sglang 0.5.9, torch 2.9.1, transformers 4.57.1

## Baseline（Qwen3-8B，no-think，训练前）

| 数据集 | 模式 | pass@1 | passed/total | 额外指标 |
|--------|------|--------|-------------|----------|
| MBPP test | one-shot | **61.4%** | 307/500 | — |
| MBPP test | multi-turn | **72.4%** | 362/500 | avg_turns=3.24, fix_rate=36.2% |
| HumanEval | one-shot | **76.8%** | 126/164 | — |
| HumanEval | multi-turn | **81.1%** | 133/164 | avg_turns=2.62, fix_rate=40.5% |

Multi-turn 比 one-shot 提升：MBPP +11.0pp，HumanEval +4.3pp。
模型已具备基础的多轮调试能力（fix_rate ~36-40%），GRPO 训练目标是进一步强化这个能力。

## 训练目标

MBPP test multi-turn pass@1 ≥ 80%（当前 baseline 72.4%），同时提高 fix_rate 和降低 avg_turns。
