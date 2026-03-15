# code-agent

用 GRPO 强化学习训练 Qwen2.5-Coder-7B-Instruct 成为多轮代码 Agent，能够自主调用工具完成"写码 → 测试 → 调试 → 提交"的完整闭环。

## 项目结构

```
src/
  prompts.py              # 共用 prompt 模板（one-shot + agentic）
  env/                    # 核心执行环境
    code_env.py           #   CodeEnvironment：单道题的交互环境，统一管理状态
    tools.py              #   3 个工具函数（write_code / run_tests / submit）+ TOOLS_SCHEMA
    sandbox.py            #   subprocess 沙箱，安全执行 Python 代码
  eval/                   # 评测
    evaluate.py           #   支持 one_shot / multi_turn / baseline 三种模式
    parser.py             #   tool call 解析器（Qwen 原生 + Markdown JSON fallback）
  data/                   # 数据
    dataset.py            #   CodeProblem 数据结构 + MBPP/HumanEval/APPS 加载
    verl_dataset.py       #   CodeProblem → verl parquet 格式转换
  verl_tools/             # verl 训练工具层
    state_manager.py      #   线程安全的 CodeEnvironment 实例管理器（按 instance_id 索引）
    write_code_tool.py    #   verl BaseTool 适配：写代码
    run_tests_tool.py     #   verl BaseTool 适配：跑测试
    submit_tool.py        #   verl BaseTool 适配：提交 + calc_reward()
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

训练和评测共用同一个执行环境栈：

```
verl_tools/*  或  evaluate.py
        ↓
  CodeEnvironment.execute_tool()    ← 自动追踪 tool_history
        ↓
  tools.py (tool_write_code / tool_run_tests / tool_submit)
        ↓
  sandbox.py (execute_with_tests)   ← subprocess 隔离执行
```

- **CodeEnvironment** 是唯一的状态持有者，verl 工具和评测脚本都通过它交互
- **verl_tools/** 是薄适配层，只做 verl BaseTool 接口适配和 step_reward / calc_reward 计算
- **CodeEnvStateManager** 按 instance_id 管理多个 CodeEnvironment 实例，供 verl 并行 rollout 使用

## 模型和工具

- 基座模型：`Qwen/Qwen2.5-Coder-7B-Instruct`
- 3 个工具：`write_code`（写代码+语法检查）、`run_tests`（跑测试+traceback）、`submit`（提交最终答案）
- 工具调用格式：Qwen 原生 `<tool_call>...</tool_call>`，parser 兼容 Markdown JSON fallback

## Reward 设计

最终 reward 由 `SubmitTool.calc_reward()` 计算，4 个组分：

| 组分 | 范围 | 说明 |
|------|------|------|
| exec_reward | 0.0 ~ 1.0 | passed_tests / total_tests |
| order_reward | 0 或 0.1 | write_code 出现在首个 run_tests 之前 |
| fix_reward | 0 或 0.2 | 首次测试失败但最终修复通过 |
| submit_reward | 0 或 0.1 | 全部通过后调用 submit |

每个工具调用还有 step_reward（0.05），鼓励模型做有意义的交互。

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
