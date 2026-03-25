# code-agent

用 GRPO 强化学习训练 Qwen2.5-Coder-7B-Instruct 成为多轮代码 Agent，能够自主调用工具完成"写码 → 测试 → 调试"的迭代闭环。

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
  grpo_qwen_7b.yaml      # verl GRPO 训练主配置
  tool_config.yaml        # 工具类注册（映射到 verl_tools/）
scripts/
  train_verl.sh           # 训练入口
  evaluate.sh             # 评测入口
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

- 基座模型：`Qwen/Qwen2.5-Coder-7B-Instruct`
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

# 准备 verl 训练数据
python -m src.data.verl_dataset

# 训练（需要 verl + GPU）
bash scripts/train_verl.sh 4          # 4 卡

# 评测
python -m src.eval.evaluate \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --mode multi_turn \
    --datasets mbpp_test humaneval
```

## 依赖

核心依赖：`torch`, `transformers`, `datasets`, `verl`, `peft`, `pandas`, `pyarrow`

verl 仅在训练服务器需要安装；本地可以正常运行评测和测试。

## 训练目标

MBPP test pass@1 ≥ 75%（当前 one-shot baseline 69.6%），通过 agent 多轮调试能力超越单次生成。
