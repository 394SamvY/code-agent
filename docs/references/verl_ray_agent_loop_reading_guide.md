# verl Ray / agent loop 代码阅读路线

本文记录当前阅读 verl validation / agent loop 源码时的路线图。目标给本项目的 OJ-like baseline 评测和训练建立一张稳定的源码地图，避免每次重新梳理对象关系。

## 当前项目背景

本仓库的 baseline 评测目标是复用 verl `main_ppo` validation 路径，而不是自己写一套独立 evaluate loop。

当前 OJ-like 主链路：

- 数据来自 `data/verl/*.parquet`
- parquet schema 由 `src/data/verl_dataset.py` 导出
- JSON string schema 由 `src/verl_dataset_adapter.py` 适配给 verl `RLHFDataset`
- tool schema 由 `src/env/tools.py` 生成到 `configs/verl/tool_config.yaml`
- verl tool adapter 在 `src/verl_tools/oj_tools.py`
- reward 聚合在 `src/reward.py`
- baseline 入口是 `scripts/evaluate_baseline_with_verl.sh`

因此读 verl 源码时，重点不是重新设计环境，而是理解：

- validation batch 如何进入 rollout
- Ray actor 和 worker group 如何建立
- SGLang server 如何被 agent loop 调用
- tool call reward 如何被带回 validation reward
- 为什么 multi-turn variable-length batch 会在后处理阶段触发 shape mismatch

## 总体 mental model

```text
RayPPOTrainer
  ├─ ActorRolloutRefWorker Ray actors
  │    ├─ actor model / ref model
  │    └─ hybrid rollout 权重来源
  │
  ├─ AgentLoopManager
  │    ├─ SGLang rollout server replicas
  │    │    └─ GPU 上真正做 token generation
  │    └─ AgentLoopWorker Ray actors
  │         └─ CPU 侧 async 调度：
  │            prompt -> SGLang -> tool -> observation -> SGLang
  │
  └─ RewardLoopManager（可选）
       └─ async reward model / reward loop worker
```

对本项目当前 OJ-like baseline：

- `ActorRolloutRefWorker`：负责模型加载、FSDP actor/ref、hybrid rollout 权重管理。
- `SGLang server`：真正占 GPU 做 token generation。
- `AgentLoopWorker`：Ray actor，主要做 CPU 侧 async orchestration，不直接跑模型 forward。
- `ToolAgentLoop`：单条 trajectory 的状态机，负责生成、解析 tool call、执行 tool、拼 observation。
- `src.verl_tools.oj_tools`：真正执行 `run_public_tests` / `submit_solution`。
- `RewardLoopManager`：当前 rule reward 主链路基本不用；tool step rewards 会进入 `extra_info/tool_rewards`，再由 `src/reward.py` 聚合。

## 推荐阅读顺序

### 1. 先读 trainer 控制面

入口：

- `/Users/yang/code/verl/verl/trainer/ppo/ray_trainer.py`

建议先看：

- `RayPPOTrainer.__init__`
- `RayPPOTrainer.init_workers`
- `RayPPOTrainer.fit`
- `RayPPOTrainer._validate`
- `RayPPOTrainer._get_gen_batch`

目标：

- 知道 validation dataloader 什么时候被迭代。
- 知道 `DataProto.from_single_dict(test_data)` 是在哪里构造的。
- 知道 `_get_gen_batch()` 从 full batch 中拆出哪些字段给 rollout。
- 知道 validation 如何调用 `self.async_rollout_manager.generate_sequences(...)`。
- 知道 rollout output 如何和原 batch `union`，再进入 `extract_reward(...)`。

当前 `_validate` 核心链路：

```text
for test_data in self.val_dataloader:
  test_batch = DataProto.from_single_dict(test_data)
  test_batch = test_batch.repeat(...)
  test_gen_batch = self._get_gen_batch(test_batch)
  test_gen_batch.meta_info["validate"] = True
  test_gen_batch_padded = pad_dataproto_to_divisor(...)
  test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(...)
  test_output_gen_batch = unpad_dataproto(...)
  test_batch = test_batch.union(test_output_gen_batch)
  reward_tensor, reward_extra_info = extract_reward(test_batch)
```

### 2. 再读 Ray worker group 抽象

入口：

- `/Users/yang/code/verl/verl/single_controller/ray/base.py`
- `/Users/yang/code/verl/verl/single_controller/base/decorator.py`
- `/Users/yang/code/verl/verl/protocol.py`

目标：

- 理解 `RayWorkerGroup` 不是单个 worker，而是一组 Ray actor 的管理器。
- 理解 worker group 如何把一个方法调用分发到多个 Ray actor。
- 理解 `DataProto` 如何在 batch 维度切分、concat、padding。
- 理解哪些调用是本地 Python object method，哪些调用会变成 Ray remote call。

读这部分时只需要抓住：

- `DataProto.chunk(...)`
- `DataProto.concat(...)`
- `pad_dataproto_to_divisor(...)`
- worker group method dispatch / decorator 的 split 和 collect 逻辑

当前 shape mismatch 相关点：

- `DataProto.concat(...)` 内部是 `torch.cat(batch_lst, dim=0)`。
- 如果不同 worker 返回的 tensor 宽度不同，跨 worker concat 也会失败。

### 3. 读 FSDP worker 和 hybrid rollout 资源面

入口：

- `/Users/yang/code/verl/verl/workers/fsdp_workers.py`
- `/Users/yang/code/verl/verl/workers/rollout/`
- `/Users/yang/code/verl/verl/workers/rollout/sglang_rollout/`（如果本地目录存在）

目标：

- 理解 actor/ref/rollout 在 hybrid 模式下如何 colocate。
- 理解为什么单卡 5090 会同时承载 actor FSDP 和 SGLang rollout server，从而容易 OOM。
- 理解 `ActorRolloutRefWorker` 和 SGLang server 之间不是简单的“同一个对象生成”，而是权重管理和 server 调用分离。

对当前 baseline，先不必深入 FSDP 训练细节，优先看：

- worker 初始化
- rollout server 启动
- sleep / wake up / update weights 相关方法
- `generate_sequences` 是否直接在 FSDP worker 上执行，还是交给 async rollout manager

### 4. 读 AgentLoopManager 建立过程

入口：

- `/Users/yang/code/verl/verl/experimental/agent_loop/agent_loop.py`

建议先看：

- `AgentLoopManager.__init__`
- `AgentLoopManager.create`
- `AgentLoopManager._init_servers`
- `AgentLoopManager._init_agent_loop_workers`
- `AsyncLLMServerManager`

目标：

- 理解 `AgentLoopManager` 是普通 Python manager，不是每条样本的 agent。
- 它负责建立两类东西：
  - SGLang rollout server replicas
  - `AgentLoopWorker` Ray actors
- `AgentLoopWorker` 通过 `AsyncLLMServerManager` 调 SGLang server。

需要特别区分：

- `AgentLoopManager`：batch-level orchestration。
- `AgentLoopWorker`：worker-level orchestration。
- `ToolAgentLoop`：single-sample state machine。
- SGLang server：真正 token generation backend。

### 5. 读 AgentLoopManager.generate_sequences

入口：

- `/Users/yang/code/verl/verl/experimental/agent_loop/agent_loop.py`
- `AgentLoopManager.generate_sequences`

当前逻辑：

```text
chunkes = prompts.chunk(len(self.agent_loop_workers))
outputs = await asyncio.gather(
  worker.generate_sequences.remote(chunk)
  for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
)
output = DataProto.concat(outputs)
```

目标：

- 看懂 validation batch 如何按 `agent.num_workers` 切 chunk。
- 看懂每个 chunk 被发给一个 `AgentLoopWorker` Ray actor。
- 看懂 worker outputs 回来后如何 `DataProto.concat`。

当前全量 eval 失败的一半风险在这里：

- 如果每个 worker 只返回单样本，但不同 worker 的 output tensor 宽度不同，`DataProto.concat(outputs)` 仍可能失败。

### 6. 读 AgentLoopWorker.generate_sequences

入口：

- `/Users/yang/code/verl/verl/experimental/agent_loop/agent_loop.py`
- `AgentLoopWorker.generate_sequences`
- `AgentLoopWorker._run_agent_loop`
- `AgentLoopWorker._agent_loop_postprocess`
- `AgentLoopWorker._postprocess`

当前逻辑：

```text
if "agent_name" not in batch.non_tensor_batch:
  agent_name = config.agent.default_agent_loop

for i in range(len(batch)):
  kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
  tasks.append(asyncio.create_task(
    self._run_agent_loop(..., **kwargs)
  ))

outputs = await asyncio.gather(*tasks)
output = self._postprocess(outputs, input_non_tensor_batch=batch.non_tensor_batch)
```

目标：

- 理解一个 `AgentLoopWorker` 内部还会并发跑 chunk 内多条样本。
- 理解 `agent_name` 决定走 `single_turn_agent`、verl 原生 `tool_agent`，还是本项目注册的 `code_agent_tool_agent`。
- 理解 `_agent_loop_postprocess` 会把单样本输出 pad 成 tensor。
- 理解 `_postprocess` 会把同一 worker 内多条样本 `torch.cat` 成 batch。

当前必须注意：

- verl 默认 `actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent`。
- 本项目 baseline 必须显式设为 `code_agent_tool_agent`，否则不会启用 AgentLoopWorker 内的 OJ terminal stop 和 assistant turn budget。
- `scripts/evaluate_baseline_with_verl.sh` 已显式覆盖：

```text
actor_rollout_ref.rollout.agent.default_agent_loop=code_agent_tool_agent
actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/verl/code_agent_loop.yaml
```

### 7. 最后读 ToolAgentLoop.run

入口：

- `/Users/yang/code/verl/verl/experimental/agent_loop/tool_agent_loop.py`

建议看：

- `ToolAgentLoop.__init__`
- `ToolAgentLoop.run`
- `_handle_pending_state`
- `_handle_generating_state`
- `_handle_processing_tools_state`
- `_call_tool`

单样本状态机：

```text
PENDING
  -> apply_chat_template(messages, tools=tool_schemas)
  -> GENERATING

GENERATING
  -> server_manager.generate(prompt_ids, sampling_params)
  -> parse tool calls
  -> no tool call: TERMINATED
  -> has tool call: PROCESSING_TOOLS

PROCESSING_TOOLS
  -> _call_tool(...)
  -> append tool response message
  -> tokenize tool observation
  -> GENERATING
```

对 OJ-like baseline：

- `tools_kwargs` 来自 parquet `extra_info.tools_kwargs`
- `run_public_tests` / `submit_solution` 的具体执行在 `src/verl_tools/oj_tools.py`
- 每次 tool call 返回：
  - `ToolResponse`
  - step reward
  - tool extra fields
- step reward 被追加到 `agent_data.tool_rewards`
- 最终进入 output `extra_fields["tool_rewards"]`

## 当前 baseline 的关键调用链

```text
scripts/evaluate_baseline_with_verl.sh
  -> scripts/verl_main_wrapper.py
    -> verl.trainer.main_ppo
      -> RayPPOTrainer._validate
        -> _get_gen_batch(test_batch)
        -> self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
          -> AgentLoopManager.generate_sequences
            -> prompts.chunk(num_agent_workers)
            -> AgentLoopWorker.generate_sequences.remote(chunk)
              -> for each sample:
                   _run_agent_loop(...)
                     -> ToolAgentLoop.run(...)
                       -> server_manager.generate(...)  # SGLang
                       -> tool_parser.extract_tool_calls(...)
                       -> _call_tool(...)               # OJ tool
                       -> append tool observation
              -> _postprocess(outputs)
            -> DataProto.concat(worker_outputs)
        -> test_batch.union(test_output_gen_batch)
        -> extract_reward(test_batch)
          -> src/reward.py::compute_score
```

## 对象关系速查

| 对象 | 类型 | 主要位置 | 主要职责 |
| --- | --- | --- | --- |
| `RayPPOTrainer` | 普通 trainer object | `trainer/ppo/ray_trainer.py` | 控制训练/验证主流程 |
| `RayWorkerGroup` | Ray actor group manager | `single_controller/ray/base.py` | 管理和分发一组 Ray workers |
| `ActorRolloutRefWorker` | Ray actor | `workers/fsdp_workers.py` | actor/ref/rollout hybrid worker |
| `AgentLoopManager` | 普通 manager object | `experimental/agent_loop/agent_loop.py` | 建立 rollout server 和 agent workers，batch 分发 |
| `AgentLoopWorker` | Ray actor | `experimental/agent_loop/agent_loop.py` | 对 chunk 内样本跑 async agent loop |
| `ToolAgentLoop` | 单样本 agent loop | `experimental/agent_loop/tool_agent_loop.py` | 多轮 tool calling 状态机 |
| `AsyncLLMServerManager` | server client manager | `experimental/agent_loop/agent_loop.py` | 选择并调用 SGLang server |
| SGLang server replica | 推理服务 | verl rollout backend | GPU token generation |
| `RewardLoopManager` | 可选 reward manager | reward loop 相关源码 | async reward model / reward loop |
| `DataProto` | batch 数据容器 | `protocol.py` | tensor / non-tensor / meta_info 传输 |

## 本项目当前配置注意点

baseline 入口：

```bash
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
bash scripts/evaluate_baseline_with_verl.sh livecodebench_test
```

关键 override：

```text
trainer.val_only=True
trainer.val_before_train=True
actor_rollout_ref.rollout.name=sglang
actor_rollout_ref.rollout.multi_turn.enable=true
actor_rollout_ref.rollout.multi_turn.tool_config_path=configs/verl/tool_config.yaml
actor_rollout_ref.rollout.agent.default_agent_loop=code_agent_tool_agent
actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/verl/code_agent_loop.yaml
data.custom_cls.path=src/verl_dataset_adapter.py
data.custom_cls.name=OJLikeRLHFDataset
reward.custom_reward_function.path=src/reward.py
```

短期稳定 eval 建议：

```bash
VAL_BATCH_SIZE=1 AGENT_WORKERS=1 \
  bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

原因：

- verl experimental agent loop 当前对 variable-length multi-turn batch 后处理不安全。
- 同一 worker 内多条样本 final prompt 宽度不同会在 `_postprocess()` 的 `torch.cat` 失败。
- 不同 worker 返回宽度不同也可能在 `DataProto.concat()` 失败。

## 已知失败点：variable-length multi-turn output

失败位置一：

```text
AgentLoopWorker._postprocess()
  prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
```

失败原因：

- `_agent_loop_postprocess()` 对 prompt 使用 `padding="max_length"` 和 `max_length=rollout_config.prompt_length`
- 如果 multi-turn 后 final prompt 超过 `prompt_length`，tokenizer pad 不会截断成 `prompt_length`
- 不同样本 final prompt 长度可能不同
- `torch.cat(..., dim=0)` 要求除 batch 维外其他维度完全一致

失败位置二：

```text
AgentLoopManager.generate_sequences()
  output = DataProto.concat(outputs)
```

失败原因：

- 即使每个 worker 只处理一条样本，不同 worker 输出宽度不同，也可能在跨 worker concat 失败。

短期规避：

- `VAL_BATCH_SIZE=1`
- `AGENT_WORKERS=1`

长期方案：

- shard eval parquet，多进程并行跑 `VAL_BATCH_SIZE=1`
- 或 patch verl agent loop / DataProto concat 的 padding 逻辑
- patch 前必须加最小复现测试，避免静默污染 reward / logprob / mask

## 阅读时要持续问的三个问题

### 这个对象是不是 Ray actor？

判断方法：

- 是否通过 `ray.remote(...)` 创建
- 是否通过 `.remote(...)` 调用
- 是否被 worker group 管理

### 这个对象跑在 CPU 还是 GPU？

经验判断：

- `AgentLoopWorker` 多数是 CPU orchestration
- SGLang server 是 GPU 推理
- FSDP actor/ref worker 持 GPU 权重
- tool execution 在本项目里主要是 Python subprocess，本身不依赖 GPU

### 当前代码处理的是 batch 还是 single sample？

层级判断：

- `_validate`：validation batch
- `AgentLoopManager.generate_sequences`：batch split / worker dispatch
- `AgentLoopWorker.generate_sequences`：worker chunk
- `_run_agent_loop` / `ToolAgentLoop.run`：single sample trajectory
- `src/verl_tools/oj_tools.py`：single tool call

## 下一步深入建议

如果继续读 `generate_sequences`，建议按这个顺序逐行标注：

1. `_validate` 中 `test_gen_batch` 的 tensor keys 和 non-tensor keys。
2. `AgentLoopManager.generate_sequences` 的 chunk 形状。
3. `AgentLoopWorker.generate_sequences` 中每个 `kwargs` 包含哪些字段。
4. `_run_agent_loop` 如何根据 `agent_name` instantiate `ToolAgentLoop`。
5. `ToolAgentLoop._handle_generating_state` 如何调用 SGLang。
6. `ToolAgentLoop._handle_processing_tools_state` 如何把 tool response 编码回 token。
7. `_agent_loop_postprocess` 如何构造 `prompts/responses/response_mask/input_ids/attention_mask/position_ids`。
8. `_postprocess` 和 `DataProto.concat` 为什么要求 tensor 宽度一致。
