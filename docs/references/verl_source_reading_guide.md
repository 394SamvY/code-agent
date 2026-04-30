# verl 源码阅读路线

目标：理解 `generate_sequences` 从调用到模型生成的完整链路，判断 `src/verl_agent_loop.py`
的改动是否合理。

## 调用链路总览

```
scripts/evaluate_baseline_with_verl.sh
  → scripts/verl_main_wrapper.py::CodeAgentTaskRunner.run()
    → verl/trainer/main_ppo.py::run_ppo()
      → verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit()
        → RayPPOTrainer._validate()                          ← 本章入口
          → async_rollout_manager.generate_sequences(batch)  ← 具体调度入口
            → AgentLoopManager.generate_sequences()          ← 批量分发
              → AgentLoopWorker.generate_sequences()         ← 单 worker
                → AgentLoopWorker._run_agent_loop()          ← 逐样本循环
                  → ToolAgentLoop.run()                      ← 状态机（我们继承的基类）
                    → _handle_generating_state()             ← 我们覆写的核心方法
```

## 阅读顺序

### 第 1 步：validation 入口

**文件：** `~/code/verl/verl/trainer/ppo/ray_trainer.py`

看 `_validate()` 方法（约 line 1180-1270）。

- 目标：理解 validation batch 如何组装、如何传给 `generate_sequences`、返回后如何处理
- 关联我们自己代码：`src/verl_runtime_patch.py` 的 `validate_with_partial_dump` patch 就是这里

### 第 2 步：批量调度层

**文件：** `~/code/verl/verl/experimental/agent_loop/agent_loop.py`

看 `AgentLoopManager`（line 902-1080）：

- `__init__`: 如何创建多个 `AgentLoopWorker`（Ray actor）
- `generate_sequences()` (line 1030): 如何将 batch 分发给 workers，收集结果，`DataProto.concat`

关键问题：为什么 `DataProto.concat` 会报 shape mismatch？答案在 `AgentLoopOutput`
的 `extra_fields` 如何跨样本累积。

### 第 3 步：单 worker 的逐样本循环

**文件：** 同上 `agent_loop.py`

看 `AgentLoopWorker`（line 392-530）：

- `generate_sequences()` (line 454): 对 batch 中每条样本 `_run_agent_loop()`
- `_run_agent_loop()` (line 535): 创建 `AgentLoopBase` 实例，调用 `run()`

### 第 4 步：状态机基类

**文件：** `~/code/verl/verl/experimental/agent_loop/tool_agent_loop.py`

看 `ToolAgentLoop`（line 96-420）：

- `run()`: 主循环，generating → processing_tools → interacting 三态切换
- `_handle_generating_state()` (line 214): **这就是我们继承并覆写的方法**
- `_handle_processing_tools_state()`: 工具调用后如何组装 tool response message
- `_call_tool()` (line 421): 工具执行与返回处理

**对比阅读：** 打开 `src/verl_agent_loop.py`，看看我们的 `CodeAgentToolAgentLoop`
对每个方法做了什么改动，是否必要。

### 第 5 步（选读）：Ray 分布式执行层

**文件：** `~/code/verl/verl/single_controller/ray/base.py`

> 你当前在看这里。这层处理如何通过 Ray 创建 worker actor、在 worker 上调用函数、
> 收集结果。理解 `FusedWorkers` 和 `RayWorkerGroup` 就够了。

- `RayWorkerGroup`: 一组 Ray actor 的抽象
- `FusedWorkers`: 将多个 `RayWorkerGroup` 合并

大多数情况下不需要深入这层——agent loop 的调度已经封装在 `AgentLoopManager` 里了。

## 对照阅读：我们的改动

读完第 4 步后，回到 `src/verl_agent_loop.py` 逐方法对比：

| verl 原始 | 我们 | 改动原因 |
|---|---|---|
| `_handle_generating_state` (65行) | 35行入口 + 165行两阶段实现 | verl 是单次 `generate()`，无法中间介入 |
| `_handle_processing_tools_state` (35行) | 6行，仅加 terminal 检测 | super() + 判断 accepted 即终止 |
| `_call_tool` (50行) | 110行，加 trace + parse error + verdict terminal | verl 缺少 tool 级 hook |

核心问题是：verl 的 `_handle_generating_state` 调 `server_manager.generate()` 是单次调用，
不会在 token 生成中途介入。Qwen3 thinking 需要中间截断，所以必须把单次生成拆成
thinking phase + action phase。这是无法用 hook 解决的——verl 没有提供 streaming callback
或 per-token hook。

## 不需要看的

- `verl/experimental/agent_loop/prometheus_utils.py`: metrics，与逻辑无关
- `verl/experimental/agent_loop/single_turn_agent_loop.py`: 无 tool call 的简化版
- `verl/experimental/vla/`: 具身智能相关，与 OJ 评测无关
- `verl/experimental/fully_async_policy/`: 异步 RL 训练，与 validate 路径不同
