# 项目状态与交接

更新日期：2026-04-30

本文是当前项目进度和下一步工作的统一入口。长期协议、schema 和实现规范仍分别维护在对应 source-of-truth 文档中；本文只同步当前阶段、验证状态、未完成事项和交接信息。

## 文档分工

| 文档 | 职责 |
| --- | --- |
| `README.md` | 面向人类读者的项目入口：项目目标、主链路、常用命令、目录概览。 |
| `AGENTS.md` | 面向代码 agent 的操作规范：开工阅读顺序、硬约束、当前主链路和禁止回退方向。 |
| `CLAUDE.md` | Claude Code 专属入口；只保留工具差异和指向 `AGENTS.md` / 本文的链接，不复制完整规范。 |
| `docs/project_status.md` | 当前进度、验证状态、blocker、下一步和交接摘要。 |
| `docs/specs/` | 稳定协议、schema、接口和数据契约。 |
| `docs/operations/` | 运行、训练、评测、部署和排障手册。 |
| `docs/references/` | 参考资料、源码阅读路线和设计背景。 |
| `docs/decisions/` | 长期决策记录，避免反复重新争论。 |
| `docs/debug/` | 调试记录、失败分析和历史 bug 过程；用于追溯，不作为当前状态入口。 |
| `docs/legacy/` | 已退出当前主线的大块历史输出、旧实验和旧训练资产。 |

## 当前阶段

项目已经接到 OJ-like v1 主链路，当前优先级是把 baseline eval 跑得更稳定、更省卡时，再跑出可复现指标。不要继续扩展数据或恢复旧评测路线。

当前主线：

- 训练数据：`CodeContests`
- 最终测试：`LiveCodeBench`
- 环境工具：`run_public_tests` 和 `submit_solution`
- verl 数据：四个标准 Parquet 文件已经整理完成
- baseline 入口：`scripts/evaluate_baseline_with_verl.sh`
- 评测方向：复用 verl `main_ppo` validation / agent loop / tool / reward 链路
- 当前优化主题：提高 2xA800 80GB baseline eval 的单位卡时有效样本吞吐

## 当前核心决策

| 决策 | 记录 |
| --- | --- |
| 主训练路径保持 RL-only，不恢复 SFT 主线 | `docs/decisions/2026-04-29-rl-only.md` |
| 环境采用 OJ-like 两工具协议：`run_public_tests` / `submit_solution` | `docs/decisions/2026-04-29-oj-like-two-tool-protocol.md` |
| baseline 评测复用 verl validation 路径 | `docs/decisions/2026-04-29-verl-validation-baseline.md` |
| baseline eval 增加 short-thinking prompt 与 per-assistant-turn token budget | `docs/decisions/2026-04-30-eval-time-thinking-budget.md` |

## 当前工作状态

截至本次同步，工作区存在本轮 trajectory 控制相关未提交改动：

| 文件 | 当前作用 |
| --- | --- |
| `src/env/tools.py` | judge 遇到首个失败 case 即停；public tests 增加 15 次调用上限；observation 只保留首个失败并裁剪。 |
| `src/env/code_env.py` | 本地环境同步 `max_public_test_calls` 状态，便捷接口复用 tool executor。 |
| `src/verl_tools/oj_tools.py` | verl BaseTool instance state 同步 public call cap 与首错 observation 策略；通过 `agent_data.code_agent_oj_tool_state` 持久化 trajectory-level public/submission 计数，并在 accepted / submit 次数耗尽时标记 terminal。 |
| `src/trajectory_parser.py` / `scripts/parse_verl_generations.py` | 将 verl decoded `output` 转为标准 HuggingFace/OpenAI `messages` 格式，支持在线落盘和离线补 `messages` jsonl。 |
| `src/verl_runtime_patch.py` | TaskRunner 侧 patch：partial / final validation generation dump 增加标准 `messages` 字段；不再在这里 patch `ToolAgentLoop`。 |
| `src/verl_agent_loop.py` / `configs/verl/code_agent_loop.yaml` | AgentLoopWorker 侧自定义 `CodeAgentToolAgentLoop`：实现 first/followup assistant turn token budget，并在 OJ tool 标记 terminal 后结束当前 trajectory。 |
| `src/verl_dataset_adapter.py` | 修正 JSON-string prompt 过滤路径：`filter_overlong_prompts=true` 时先 decode，再按真实 chat template token 长度过滤；支持 eval-time short-thinking prompt 约束。 |
| `src/data/verl_dataset.py` | 新导出的 parquet tool create kwargs 带 `max_public_test_calls`；旧 parquet 仍走默认值。 |
| `scripts/evaluate_baseline_with_verl.sh` | 默认面向 2xA800-80GB：未指定 `CUDA_VISIBLE_DEVICES` 时使用 `0,1`，`VAL_MAX_SAMPLES=500`、`VAL_BATCH_SIZE=16`、`AGENT_WORKERS=16`、`MAX_NUM_SEQS=32`、`GPU_MEMORY_UTILIZATION=0.82`、`MAX_NUM_BATCHED_TOKENS=32768`、`CODE_AGENT_PROMPT_STYLE=short_thinking`、first/followup budget `3072/2048`；使用 `code_agent_tool_agent`，仍默认 `MAX_PROMPT_LENGTH=4096`、`MAX_RESPONSE_LENGTH=8192`、`enable_thinking=true`。 |
| `scripts/evaluate_2xa800_32_debug.sh` / `scripts/analyze_eval_generations.py` | 新增 2xA800 32-sample focused eval 脚本，自动开 GPU monitor，默认启用 short-thinking prompt 和 per-assistant-turn token budget，并输出行为/GPU summary。 |
| `tests/test_verl_tools.py` / `tests/test_e2e_protocol.py` / `tests/test_oj_verl_tools_state.py` / `tests/test_verl_dataset_adapter.py` | 覆盖首错即停、public cap 不消耗 submit 次数、create kwargs 导出、verl create/release 后计数持久化、accepted terminal 标记、JSON prompt decode 过滤。 |
| `AGENTS.md` / `README.md` / `docs/*` | 同步当前协议、reward 口径和交接状态。 |

合入前仍应做一次 focused diff review；本地 `py_compile`、`tests/test_dataset_protocol.py`、`tests/test_verl_tools.py`、`tests/test_e2e_protocol.py`、`tests/test_trajectory_parser.py`、`tests/test_oj_verl_tools_state.py`、`tests/test_verl_dataset_adapter.py` 已通过。2026-04-29 已补跑 2xA800 focused smoke，见下方最新记录。

## 最近重要结论

- `docs/debug/verl_baseline_eval_debug_2026-04-28.md` 修正了 2026-04-25 对 batch shape mismatch 的早期判断：关键问题在于 Parquet `prompt` 是 JSON string，默认 `RLHFDataset.maybe_filter_out_long_prompts` 低估真实 chat template token 长度。
- 不应使用 `sitecustomize.py` 或全局启动钩子提前 import verl/torch；这会干扰 Ray GPU worker 的 CUDA 绑定，可能触发 NCCL `Duplicate GPU detected`。
- TaskRunner 侧 patch 安装点应在 `scripts/verl_main_wrapper.py` 的 `CodeAgentTaskRunner.run()` 内，即 Ray CPU TaskRunner actor 中；AgentLoopWorker 侧行为应通过 `configs/verl/code_agent_loop.yaml` 注册自定义 agent loop，不依赖 TaskRunner monkey patch。
- validation 需要增量落盘：`src/verl_runtime_patch.py` patch `RayPPOTrainer._validate`，每个 validation batch 完成后追加写 `generations/partial_0.jsonl`。
- 默认评测不再维护“两张卡各跑一个 verl 进程”的 sharded fallback；应通过单个 verl 进程统一调度多卡。

### 2026-04-29 baseline eval 效率分析

最近一次重点分析的输出目录：

```text
outputs/verl_baseline_eval/codecontests_test_Qwen3-8B_mp4096_mr8192_20260428_201713
```

该 run 使用 2xA800，主要配置为 `MAX_PROMPT_LENGTH=4096`、`MAX_RESPONSE_LENGTH=8192`、`MAX_MODEL_LEN=12288`、`VAL_BATCH_SIZE=8`、`AGENT_WORKERS=8`、`MAX_NUM_SEQS=16`、`GPU_MEMORY_UTILIZATION=0.70`、`ENFORCE_EAGER=true`。本地只有 `generations/partial_0.jsonl`，共 56 条。

关键数据画像：

| 指标 | 数值 |
|---|---|
| 总样本 | 56 |
| 平均 tool calls | 14 次 |
| p50 / p90 / 最大 | 16 / 26 / 31 次 |
| score = 0（从未有效提交） | 45/56 (80.4%) |
| score = 1.0 (Accepted) | 5/56 (8.9%) |
| failed submit (0 < score < 1) | 6/56 (10.7%) |
| 平均输出长度 | 20,760 字符 |
| Tool call JSON decode 失败 | 7 次 |

暴露的问题分三类：

**需要工程解决（当前优先级）**

1. (P2) **重复代码消耗**：模型反复用相同代码调 `run_public_tests`，每次重新跑 subprocess judge 和生成 observation。方案：相同 `tool_name + code_hash` 结果缓存，连续相同代码返回短 observation。当前先不做。

2. (P5) **输出格式不直观**：已改为在 `partial_0.jsonl` 和 `0.jsonl` 中附加标准 `messages` 字段。旧 generation jsonl 可用 `python3 scripts/parse_verl_generations.py <jsonl>` 离线补 `messages`。

3. (P4) **eval budget 暂不调整**：当前 baseline eval 默认保持输入限制 `MAX_PROMPT_LENGTH=4096`，整条 trajectory 输出限制 `MAX_RESPONSE_LENGTH=8192`。

**RL 训练解决（当前不处理）**

9. **Base model 改不对代码**：反馈信息足够（具体报错 + input/expected/stdout），Qwen3-8B 缺乏 code debugging 能力。属模型能力差距，应通过 RL 训练提升。

10. **Tool call JSON 格式错误**：7 次 decode 失败，base model 未经 tool calling 训练。同样留到 RL 训练解决。

**已解决**

1. **500 条没跑完**：可能是手动中断，不追查。
2. (P1) **公开测试无限循环保护**：`run_public_tests` 增加 `max_public_test_calls=15` 默认上限，达到后返回短提示引导调用 `submit_solution`，不消耗正式提交次数。
3. (P3) **Observation 累积过长**：public/private judge 均改为遇到首个失败 case 即停止，observation 只展示首个失败 case，并保留文本裁剪。
4. (P6) **评测开启 thinking**：`scripts/evaluate_baseline_with_verl.sh` 默认 `enable_thinking=true`。
5. (P5) **多轮输出标准 messages 保存**：新增标准 `messages` 在线落盘和离线解析脚本。

### 2026-04-29 thinking eval smoke 分析

最新一次用户关注的输出目录：

```text
outputs/verl_baseline_eval/smoke_structured_2gpu_
```

该 run 使用 2xA800，配置中 `enable_thinking=true`、`VAL_BATCH_SIZE=8`、`VAL_MAX_SAMPLES=-1`，因此实际开始跑完整 `codecontests_test`，不是 4 条小 smoke。目录里只有 `generations/partial_0.jsonl` 和 `verl_eval.log`，没有最终 `generations/0.jsonl`。`partial_0.jsonl` 共 264 条，对应 33 个完整 validation batch。

关键数据画像：

| 指标 | 数值 |
|---|---:|
| 已完成样本 | 264 |
| score = 1.0 (Accepted) | 53 |
| failed submit (0 < score < 1) | 6 |
| score = 0 | 205 |
| 平均 score | 0.201 |
| num_tool_calls = 0 | 198 |
| `<think>` 未闭合且无 tool call | 195 |
| 闭合 `</think>` 且出现 tool tag | 69 |
| tool 调用样本平均 tool calls | 约 2.03 |

本次暴露出的主要问题是 **thinking 过长**，而不是工具反馈不足：大量样本第一轮 assistant 一直停留在 `<think>` 内，30k 字符左右仍在推导边界条件或实现细节，完全没有进入 `run_public_tests` / `submit_solution`。一旦模型闭合 thinking 并进入工具协议，成功率并不低：出现 tool tag 的 69 条里有 53 条 accepted。

因此当前不能关闭 thinking，但需要约束 thinking 行为。优先采用 verl / Qwen3 原生超参方向，而不是立刻 patch agent loop：

- thinking mode 下不要使用 greedy validation。当前 `val_kwargs.temperature=0`、`do_sample=false` 容易导致 Qwen3 长思考和重复推导。
- 下一次 focused smoke 先调整 validation sampling：`temperature=0.6`、`top_p=0.95`、`top_k=20`。
- 保持 `enable_thinking=true`、`MAX_PROMPT_LENGTH=4096`、`MAX_RESPONSE_LENGTH=8192` 不变，先观察 `unclosed_think_rate`、`num_tool_calls=0` 比例、accepted rate 和平均输出长度。
- 如果超参调整后仍然大量卡在 `<think>`，再进入第二步：通过 verl 的 `agent_loop_config_path` / custom agent loop 扩展实现 Qwen3 thinking budget 两段生成，而不是直接修改 verl 核心源码。

本次 run 的报错：

```text
RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 6663 but got size 4096 for tensor number 1 in the list.
```

该错误发生在 `async_rollout_manager.generate_sequences()` 内部的 `DataProto.concat(outputs)`，当前判断与上次 batch shape mismatch 属同类问题：仍有样本在 agent loop 返回时 tensor 宽度没有统一到 batch 内固定形状，可能还是 prompt / multi-turn prompt 过滤或截断没有完全覆盖。由于 partial dump 在每个 batch 完成后才写出，崩溃中的第 34 个 batch 没有落盘；已完成的 264 条可用于行为分析，但不能算完整 baseline。

此外还有两个 trajectory 控制问题需要关注：

1. **accepted 后没有立即终止 trajectory**：本次 accepted submit 样本中，`submit_solution: accepted` 后仍会继续生成 assistant thinking/text，平均多消耗约 2400 字符。这会浪费 response budget 和卡时，并可能污染后续结构化分析。后续应确认 verl `ToolAgentLoop` 是否支持基于 tool result 的 terminal stop；如果不支持，需要在 custom agent loop 或 tool adapter 层补 accepted terminal 语义。
2. **verl tool state 可能没有跨 tool call 持久化**：当前 verl `ToolAgentLoop` 每次 tool 调用都会 `create -> execute -> release`，而 `src/verl_tools/oj_tools.py` 的 `public_test_call_count` / `submission_count` 存在 per-instance state 中。这样 `max_public_test_calls` / `max_submissions` 在 verl 在线 eval 中可能不会跨回合累计，和本地 `CodeEnvironment` 语义不完全一致。后续需要通过 focused smoke 或单测确认真实行为；如确认不持久，应改为基于 `request_id` / `agent_data` / trajectory-level session 的状态管理。

### 2026-04-29 follow-up 本地修复与 A800 smoke

本轮没有直接启动完整 baseline；先做了代码级修复、本地最小验证，然后补跑 2xA800 focused smoke。

- `scripts/evaluate_baseline_with_verl.sh` 已将 validation sampling 默认改为 Qwen3 thinking 推荐方向：`temperature=0.6`、`top_p=0.95`、`top_k=20`、`do_sample=true`，继续保持 `enable_thinking=true`、`MAX_PROMPT_LENGTH=4096`、`MAX_RESPONSE_LENGTH=8192`。
- shape mismatch 的一个明确代码原因已修复：`src/verl_dataset_adapter.py` 原先在 `filter_overlong_prompts=true` 时反而走父类过滤，仍可能按 JSON string 低估长度；现在会 decode 后按真实 chat template token 长度过滤。eval 脚本也补了 `data.tool_config_path=configs/verl/tool_config.yaml`，让过滤阶段包含 tool schema。
- `submit_solution` accepted / submit 次数耗尽会在 tool adapter 中写入 trajectory terminal 标记；`src/verl_runtime_patch.py` patch verl `ToolAgentLoop._call_tool()` 和 `_handle_processing_tools_state()`，看到 tool result `terminal=true` 后结束 trajectory 状态机。
- `public_test_call_count` / `submission_count` 已从 per-instance state 同步到 `agent_data.code_agent_oj_tool_state` 普通属性，即使 verl 每次 tool call 都 `create -> execute -> release`，同一 trajectory 内也会跨 tool call 累计。不要放进 `agent_data.extra_fields`，否则 `DataProto.concat` 会按 batch 维度校验非 tensor 字段并报长度不一致。
- generation dump 只保留标准 `messages` 字段，消息含 `role`、`tool_calls`、`tool_call_id`；不再落盘自定义 `structured_output`。

#### 2xA800 focused smoke：67GB 配置

输出目录：

```text
outputs/verl_baseline_eval/smoke_sampling_t06_topk20_vbs16_32_v3
```

命令关键参数：`VAL_MAX_SAMPLES=32`、`VAL_BATCH_SIZE=16`、`AGENT_WORKERS=16`、`MAX_NUM_SEQS=32`、`GPU_MEMORY_UTILIZATION=0.82`、`MAX_NUM_BATCHED_TOKENS=32768`、`temperature=0.6`、`top_p=0.95`、`top_k=20`。`partial_0.jsonl` 和 `0.jsonl` 均写出 32 条，没有 shape mismatch。

| 指标 | 数值 |
|---|---:|
| 样本数 | 32 |
| score/accepted rate | 18.75% |
| `unclosed_think_rate` | 75.0% |
| `num_tool_calls = 0` | 78.1% |
| submit sample rate | 21.9% |
| 平均输出长度 | 28,192 字符 |
| 平均 tool calls | 0.44 |
| max public / submit calls | 1 / 1 |
| public/submission limit observation | 0 / 0 |
| accepted 后额外输出平均 / max | 2,813 / 5,098 字符 |
| `structured_output` | 无 |
| GPU max memory | 67.5GB / 67.5GB |
| GPU avg active util | 75.6% / 67.9% |
| high-memory 每题耗时估算 | 12.3s |

结论：

- Qwen3 推荐 sampling 能完整跑通 32 条，当前没有复现 shape mismatch。
- thinking 过长仍是主要问题：`unclosed_think_rate` 仍约 75%，`num_tool_calls=0` 仍约 78%。
- accepted 后不再出现重复 submit，说明 tool state 跨 `create -> execute -> release` 累计生效；但 accepted tool call 后仍有 1 条 assistant 尾部文本。定位结果：这些 token 多半是同一次 assistant generation 在 `<tool_call>` 后继续生成的尾部说明，工具 accepted 后终止状态机不能回收已经生成的尾部 token。下一步需要在 agent loop 中截断 tool_call 后的同轮尾部，或在 parser/dump 层先裁剪标准 messages。

#### 2xA800 focused smoke：77GB probe

调参说明见 `docs/operations/gpu_eval_tuning.md`。

输出目录：

```text
outputs/verl_baseline_eval/smoke_sampling_t06_topk20_vbs24_32_mem094
```

命令关键参数：`VAL_MAX_SAMPLES=32`、`VAL_BATCH_SIZE=24`、`AGENT_WORKERS=24`、`MAX_NUM_SEQS=48`、`GPU_MEMORY_UTILIZATION=0.94`、`MAX_NUM_BATCHED_TOKENS=49152`。`partial_0.jsonl` 和 `0.jsonl` 均写出 32 条，没有 OOM，也没有 shape mismatch。

| 指标 | 数值 |
|---|---:|
| 样本数 | 32 |
| score mean | 0.1267 |
| accepted rate | 12.5% |
| `unclosed_think_rate` | 65.6% |
| `num_tool_calls = 0` | 65.6% |
| submit sample rate | 25.0% |
| 平均输出长度 | 27,114 字符 |
| 平均 tool calls | 0.53 |
| max public / submit calls | 1 / 1 |
| accepted 后额外输出平均 / max | 3,501 / 4,534 字符 |
| GPU max memory | 77.8GB / 77.8GB |
| GPU avg active util | 80.9% / 67.8% |
| high-memory 每题耗时估算 | 14.2s |

结论：

- `GPU_MEMORY_UTILIZATION=0.94` 能把 A800 显存推到约 77.8GB，32 条 smoke 稳定完成。
- 77GB probe 没有比 67GB 配置更快：high-memory 区间每题耗时从约 12.3s 变成约 14.2s。继续加显存和并发开始受少数长 thinking 轨迹、自回归长尾和调度成本影响，暂不建议作为默认。
- 当前默认更建议保持 67GB 级配置（`VAL_BATCH_SIZE=16`、`AGENT_WORKERS=16`、`MAX_NUM_SEQS=32`、`GPU_MEMORY_UTILIZATION=0.82`），下一步优先解决 thinking 过长和 tool_call 后尾部截断。

对 `outputs/verl_baseline_eval/smoke_structured_2gpu_/generations/partial_0.jsonl` 复算的行为基线：

| 指标 | 数值 |
|---|---:|
| 样本数 | 264 |
| `unclosed_think_rate` | 73.9% |
| `num_tool_calls = 0` | 74.2% |
| submit 率 | 23.9% |
| accepted 率 | 20.1% |
| 平均输出长度 | 27,808 字符 |
| 平均 tool calls | 0.54 |
| accepted 后额外输出平均 / p50 / max | 2,419 / 2,425 / 4,602 字符 |

`verl_eval.log` 没有记录按时间采样的 GPU 利用率，只能看到本次配置 `gpu_memory_utilization=0.55`；下一次 focused smoke 应显式并行跑 `scripts/monitor_gpu.sh` 或等价 `nvidia-smi` 采样。

### 2026-04-30 short-thinking eval debug

本轮新增两个 eval-time 控制，用于解决 Qwen3 thinking 过长耗尽 `MAX_RESPONSE_LENGTH=8192` 的问题：

- `CODE_AGENT_PROMPT_STYLE=short_thinking`：在 dataset adapter decode 现有 parquet prompt 后，对 system prompt 追加“短思考、尽早调用 OJ tool”的评测约束。不需要重新导出 parquet。
- `CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET=3072` / `CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET=2048`：在 `src.verl_agent_loop.CodeAgentToolAgentLoop` 中给 first/followup assistant generation turn 设置 `max_new_tokens` 上限，但保留整条 trajectory 的 `MAX_RESPONSE_LENGTH=8192`。该逻辑运行在 verl `AgentLoopWorker` 进程内。

新增 focused 脚本：

```bash
bash scripts/evaluate_2xa800_32_debug.sh codecontests_test
```

默认参数为 2xA800 67GB 档位：`VAL_MAX_SAMPLES=32`、`VAL_BATCH_SIZE=16`、`AGENT_WORKERS=16`、`MAX_NUM_SEQS=32`、`GPU_MEMORY_UTILIZATION=0.82`、`MAX_NUM_BATCHED_TOKENS=32768`、first/followup budget `3072/2048`。

#### 32-sample short-thinking smoke：67GB 默认档

输出目录：

```text
outputs/verl_baseline_eval/debug32_shortthink_tb3072_v1
```

`partial_0.jsonl` 和 `0.jsonl` 均写出 32 条，没有 OOM 或 shape mismatch。

| 指标 | 数值 |
|---|---:|
| 样本数 | 32 |
| score mean | 0.3126 |
| accepted rate | 31.2% |
| `unclosed_think_rate` | 50.0% |
| `num_tool_calls = 0` | 31.2% |
| submit sample rate | 68.8% |
| 平均输出长度 | 18,220 字符 |
| 平均 tool calls | 8.78 |
| accepted 后额外输出平均 / max | 818 / 3,105 字符 |
| GPU max memory | 65.8GB / 66.0GB |
| GPU avg active util | 82.2% / 76.2% |
| 总 wall-clock 估算 | 426s，约 13.3s/题 |

对比上一轮 `smoke_sampling_t06_topk20_vbs16_32_v3`：

| 指标 | 旧 67GB | short-thinking 67GB |
|---|---:|---:|
| accepted rate | 18.8% | 31.2% |
| `unclosed_think_rate` | 78.1% | 50.0% |
| `num_tool_calls = 0` | 78.1% | 31.2% |
| submit sample rate | 21.9% | 68.8% |
| 平均输出长度 | 28,192 字符 | 18,220 字符 |
| accepted 后额外输出平均 | 2,882 字符 | 818 字符 |

结论：short-thinking prompt + per-turn budget 明显缓解了“第一轮长思考不调用工具”的主问题。它没有关闭 thinking，也没有缩小整条 trajectory 的 8192 budget；它只是防止单个 assistant turn 把 8192 全吃完。

#### 32-sample short-thinking smoke：77GB 高显存档

输出目录：

```text
outputs/verl_baseline_eval/debug32_shortthink_tb3072_mem094_v1
```

命令关键参数：`GPU_MEMORY_UTILIZATION=0.94`、`VAL_BATCH_SIZE=24`、`AGENT_WORKERS=24`、`MAX_NUM_SEQS=48`、`MAX_NUM_BATCHED_TOKENS=49152`。本轮同样完整写出 32 条，没有 OOM 或 shape mismatch。

| 指标 | 数值 |
|---|---:|
| score mean | 0.2814 |
| accepted rate | 28.1% |
| `unclosed_think_rate` | 53.1% |
| `num_tool_calls = 0` | 31.2% |
| submit sample rate | 68.8% |
| 平均输出长度 | 20,302 字符 |
| 平均 tool calls | 9.47 |
| GPU max memory | 75.8GB / 75.5GB |
| GPU avg active util | 76.7% / 83.6% |
| 总 wall-clock 估算 | 486s，约 15.2s/题 |

结论：77GB 档确实能把显存推到约 75-76GB，但在 short-thinking 行为下仍未比 67GB 档更快，且指标略差。当前默认仍建议 67GB 档位；77GB 档只保留为 probe override，不作为 baseline eval 默认。

#### 500-sample debug run 暴露的问题

输出目录：

```text
outputs/verl_baseline_eval/codecontests_test_500_shortthink_tb3072
```

该 run 写出 499 条 `0.jsonl`，但不作为正式 baseline。最终统计为 `score_mean=0.1940`、`accepted_rate=19.2%`、`num_tool_calls_zero_rate=26.3%`、`avg_tool_calls=13.92`、`max_tool_calls=131`、`num_turns max=264`。离线 tokenizer 检查显示 no-tool 样本仍有 74 条生成到 8192 tokens，说明旧的 hard budget monkey patch 没有进入真正执行样本的 `AgentLoopWorker` 进程；`submission_limit_exceeded` 后反复 submit 也说明 terminal stop 同样没有进入 AgentLoopWorker。

当前修复方向已经改为自定义 agent loop：`src.verl_agent_loop.CodeAgentToolAgentLoop` 通过 `configs/verl/code_agent_loop.yaml` 注册，并由 eval 脚本设置 `actor_rollout_ref.rollout.agent.default_agent_loop=code_agent_tool_agent`。下一次 32 条 smoke 的核心验收是 no-tool 最大 token 接近 first budget 3072，而不是 8192，并且 `max_tool_calls` 不再出现 50/100+。

#### 32-sample custom agent loop 验证

输出目录：

```text
outputs/verl_baseline_eval/debug32_code_agent_loop_f3072_u2048_verify_v2
```

命令关键参数：`VAL_MAX_SAMPLES=32`、`VAL_BATCH_SIZE=16`、`AGENT_WORKERS=16`、`MAX_NUM_SEQS=32`、`GPU_MEMORY_UTILIZATION=0.82`、`MAX_NUM_BATCHED_TOKENS=32768`、`CODE_AGENT_PROMPT_STYLE=short_thinking`、first/followup budget `3072/2048`。本轮完整写出 `generations/0.jsonl`，没有 OOM 或 shape mismatch。

第一轮实跑 `debug32_code_agent_loop_f3072_u2048_verify` 失败在 `DataProto.concat(outputs)`：只给 terminal 样本写 `agent_data.extra_fields["code_agent_terminal_reason"]` 会造成不同样本的 `non_tensor_batch` key 不一致。修复后不再向 `extra_fields` 写动态 terminal 字段，只把 terminal 标记放在普通 `agent_data` 属性和 metrics 中。

| 指标 | 数值 |
|---|---:|
| 样本数 | 32 |
| score mean | 0.2237 |
| accepted rate | 21.9% |
| `unclosed_think_rate` | 78.1% |
| `num_tool_calls = 0` | 34.4% |
| submit sample rate | 65.6% |
| 平均输出长度 | 9,079 字符 |
| 平均 tool calls | 5.75 |
| max tool calls | 21 |
| num_turns max / mean | 43 / 12.875 |
| GPU max memory | 65.8GB / 66.2GB |
| GPU avg active util | 65.5% / 52.6% |
| 总 wall-clock 估算 | 303s，约 9.5s/题 |

tokenizer 级验收：

| 检查项 | 数值 |
|---|---:|
| no-tool 样本数 | 11 |
| no-tool 最大 output tokens | 3072 |
| 第一段 assistant 最大 tokens | 3072 |
| 第一段达到 8192 的样本数 | 0 |
| max tool calls | 21 |

结论：`CodeAgentToolAgentLoop` 已经在真实 verl `AgentLoopWorker` eval 路径中生效。旧 500-sample debug run 中 no-tool 样本顶满 8192、`max_tool_calls=131`、`num_turns max=264` 的现象，在这次 32 条实跑中没有复现。下一步可以用同一配置跑 500 条 CodeContests held-out eval。

## 验证状态

最近记录在 `docs/debug/verl_baseline_eval_debug_2026-04-28.md` 的验证：

- 1GPU partial dump smoke 通过，`partial_0.jsonl` 和 `0.jsonl` 均写出 2 条。
- 2GPU 统一调度 smoke 通过，两个 SGLang server 分别绑定 GPU0 和 GPU1，`partial_0.jsonl` 和 `0.jsonl` 均写出 4 条。
- 清理 sharded fallback 和 `sitecustomize.py` 后，2GPU smoke 再次通过。
- 2026-04-29 1GPU structured output smoke 通过：`smoke_structured_output_20260429_185821`，`VAL_MAX_SAMPLES=2`，`partial_0.jsonl` 和 `0.jsonl` 均写出 2 条。该历史 run 使用过旧的自定义结构字段；当前新输出只保留标准 `messages`。
- 2026-04-29 2xA800 67GB focused smoke 通过：`smoke_sampling_t06_topk20_vbs16_32_v3`，32 条完整写出，无 shape mismatch，无 `structured_output`。
- 2026-04-29 2xA800 77GB probe 通过：`smoke_sampling_t06_topk20_vbs24_32_mem094`，32 条完整写出，无 OOM、无 shape mismatch；但比 67GB 配置更慢，暂不建议作为默认。
- 2026-04-30 2xA800 custom agent loop focused eval 通过：`debug32_code_agent_loop_f3072_u2048_verify_v2`，32 条完整写出；tokenizer 检查显示 no-tool 最大 output tokens 为 3072，第一段 assistant 没有样本达到 8192。

2026-04-29 对 `codecontests_test_Qwen3-8B_mp4096_mr8192_20260428_201713` 做了离线输出分析。随后分析了 `smoke_structured_2gpu_`，该 run 已完成 264 条 partial generation，但最终因 `DataProto.concat` shape mismatch 中断，没有写出最终 `0.jsonl`。任何“当前代码已完全通过正式 baseline”或“效率问题已解决”的表述都应等下一次真实 A800 smoke / full eval 后再写入。

## 下一步

实现顺序按优先级排列：

### Phase 1：trajectory 可控

1. `max_public_test_calls=15`、首错即停、首错 observation、eval thinking 和 Qwen3 thinking validation sampling 默认值已实现；下一步用该配置重跑 focused smoke，并保持 `enable_thinking=true`。
2. (P2) 重复代码短路暂不实现，除非 smoke 仍显示重复 judge 是主要瓶颈。
3. accepted / submission-limit terminal 语义已迁移到 `CodeAgentToolAgentLoop`，32 条真实 eval 已验证没有再出现 50/100+ tool-call 循环；同一轮 assistant generation 在 `<tool_call>` 后的尾部文本仍会落入输出，accepted 后额外输出还不是 0。下一步可再评估 tool_call 后尾部 token 裁剪。
4. verl tool state 持久化已验证：`max_public_test_calls` 和 `max_submissions` 在同一道题的一条 trajectory 内通过 `agent_data.code_agent_oj_tool_state` 跨 tool call 累计；不要使用 `extra_fields` 保存该 state。
5. (P5) 多轮交互输出格式已补标准 `messages` 格式（`role: user/assistant/tool` + `tool_calls` + `tool_call_id`）；在线和离线 jsonl 都不再输出自定义 `structured_output` 字段。`src/trajectory_parser.py` 只保留 `to_messages()` 作为标准 messages 转换入口。

### Phase 2：budget 暂不动

6. baseline eval 默认保持 `MAX_PROMPT_LENGTH=4096`、`MAX_RESPONSE_LENGTH=8192`，暂不下调整条 trajectory response budget。当前通过 custom agent loop 控制 first/followup assistant turn budget，而不是缩短整条 trajectory。

### Phase 3：验证与评测

7. Review 当前未提交改动，确认无意外行为。
8. 本地单元级验证已覆盖：
   - public/private judge 遇到首个失败 case 即停止
   - `max_public_test_calls` 不破坏 `max_submissions=5` 语义
   - verl parquet `create_kwargs` 携带 public call cap
9. `scripts/evaluate_2xa800_32_debug.sh` 已验证 custom agent loop 生效；no-tool 最大 token 为 3072，第一段 assistant 没有达到 8192，`max_tool_calls=21`。
10. 当前推荐并发参数：`VAL_BATCH_SIZE=16`、`AGENT_WORKERS=16`、`MAX_NUM_SEQS=32`、`GPU_MEMORY_UTILIZATION=0.82`、`MAX_NUM_BATCHED_TOKENS=32768`、first/followup budget `3072/2048`、`CODE_AGENT_PROMPT_STYLE=short_thinking`。77GB probe 虽稳定但更慢，暂不作为默认。
11. CodeContests held-out baseline：下一步用当前默认配置跑 `VAL_MAX_SAMPLES=500`。
12. `livecodebench_test`。
13. 结果同步回本文。

## 交接提示

下一位 agent 开工时建议先读：

1. `README.md`
2. `AGENTS.md`
3. `docs/project_status.md`
4. `docs/specs/env_protocol.md`
5. `docs/specs/verl_parquet_dataset_analysis.md`
6. `docs/debug/verl_baseline_eval_debug_2026-04-28.md`

如果任务是继续 baseline eval，优先检查 `scripts/evaluate_baseline_with_verl.sh`、`scripts/verl_main_wrapper.py`、`src/verl_dataset_adapter.py`、`src/verl_runtime_patch.py` 和 `src/verl_tools/oj_tools.py`。
