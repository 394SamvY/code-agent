# 项目状态与交接

更新日期：2026-04-29

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

## 当前工作状态

截至本次同步，工作区存在本轮 trajectory 控制相关未提交改动：

| 文件 | 当前作用 |
| --- | --- |
| `src/env/tools.py` | judge 遇到首个失败 case 即停；public tests 增加 15 次调用上限；observation 只保留首个失败并裁剪。 |
| `src/env/code_env.py` | 本地环境同步 `max_public_test_calls` 状态，便捷接口复用 tool executor。 |
| `src/verl_tools/oj_tools.py` | verl BaseTool instance state 同步 public call cap 与首错 observation 策略。 |
| `src/trajectory_parser.py` / `scripts/parse_verl_generations.py` | 将 verl decoded `output` 解析为 `structured_output.events`（当前自定义格式），支持在线落盘和离线补结构化 jsonl。待改为标准 HuggingFace `messages` 格式。 |
| `src/verl_runtime_patch.py` | partial / final validation generation dump 增加 `structured_output` 字段。 |
| `src/data/verl_dataset.py` | 新导出的 parquet tool create kwargs 带 `max_public_test_calls`；旧 parquet 仍走默认值。 |
| `scripts/evaluate_baseline_with_verl.sh` | 默认 `MAX_PROMPT_LENGTH=4096`、`MAX_RESPONSE_LENGTH=8192`、`enable_thinking=true`。 |
| `tests/test_verl_tools.py` / `tests/test_e2e_protocol.py` | 覆盖首错即停、public cap 不消耗 submit 次数、create kwargs 导出。 |
| `AGENTS.md` / `README.md` / `docs/*` | 同步当前协议、reward 口径和交接状态。 |

合入前仍应做一次 focused diff review；本地 `py_compile`、`tests/test_verl_tools.py`、`tests/test_e2e_protocol.py` 已通过，A800 focused smoke 尚未重跑。

## 最近重要结论

- `docs/debug/verl_baseline_eval_debug_2026-04-28.md` 修正了 2026-04-25 对 batch shape mismatch 的早期判断：关键问题在于 Parquet `prompt` 是 JSON string，默认 `RLHFDataset.maybe_filter_out_long_prompts` 低估真实 chat template token 长度。
- 不应使用 `sitecustomize.py` 或全局启动钩子提前 import verl/torch；这会干扰 Ray GPU worker 的 CUDA 绑定，可能触发 NCCL `Duplicate GPU detected`。
- 当前 patch 安装点应在 `scripts/verl_main_wrapper.py` 的 `CodeAgentTaskRunner.run()` 内，即 Ray CPU TaskRunner actor 中。
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

2. (P5) **输出格式不直观**：已改为在 `partial_0.jsonl` 和 `0.jsonl` 中附加 `structured_output`，其 `events` 会按顺序保存 `assistant_text` / `tool_call` / `tool_response`。旧 generation jsonl 可用 `python3 scripts/parse_verl_generations.py <jsonl>` 离线补结构化视图。

3. (P4) **eval budget 暂不调整**：当前 baseline eval 默认保持输入限制 `MAX_PROMPT_LENGTH=4096`，整条 trajectory 输出限制 `MAX_RESPONSE_LENGTH=8192`。

**RL 训练解决（当前不处理）**

9. **Base model 改不对代码**：反馈信息足够（具体报错 + input/expected/stdout），Qwen3-8B 缺乏 code debugging 能力。属模型能力差距，应通过 RL 训练提升。

10. **Tool call JSON 格式错误**：7 次 decode 失败，base model 未经 tool calling 训练。同样留到 RL 训练解决。

**已解决**

1. **500 条没跑完**：可能是手动中断，不追查。
2. (P1) **公开测试无限循环保护**：`run_public_tests` 增加 `max_public_test_calls=15` 默认上限，达到后返回短提示引导调用 `submit_solution`，不消耗正式提交次数。
3. (P3) **Observation 累积过长**：public/private judge 均改为遇到首个失败 case 即停止，observation 只展示首个失败 case，并保留文本裁剪。
4. (P6) **评测开启 thinking**：`scripts/evaluate_baseline_with_verl.sh` 默认 `enable_thinking=true`。
5. (P5) **多轮输出结构化保存**：新增 `structured_output.events` 在线落盘和离线解析脚本。

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

## 验证状态

最近记录在 `docs/debug/verl_baseline_eval_debug_2026-04-28.md` 的验证：

- 1GPU partial dump smoke 通过，`partial_0.jsonl` 和 `0.jsonl` 均写出 2 条。
- 2GPU 统一调度 smoke 通过，两个 SGLang server 分别绑定 GPU0 和 GPU1，`partial_0.jsonl` 和 `0.jsonl` 均写出 4 条。
- 清理 sharded fallback 和 `sitecustomize.py` 后，2GPU smoke 再次通过。
- 2026-04-29 1GPU structured output smoke 通过：`smoke_structured_output_20260429_185821`，`VAL_MAX_SAMPLES=2`，`partial_0.jsonl` 和 `0.jsonl` 均写出 2 条，且均包含 `structured_output.events`。

2026-04-29 对 `codecontests_test_Qwen3-8B_mp4096_mr8192_20260428_201713` 做了离线输出分析。随后分析了 `smoke_structured_2gpu_`，该 run 已完成 264 条 partial generation，但最终因 `DataProto.concat` shape mismatch 中断，没有写出最终 `0.jsonl`。任何“当前代码已完全通过正式 baseline”或“效率问题已解决”的表述都应等下一次真实 A800 smoke / full eval 后再写入。

## 下一步

实现顺序按优先级排列：

### Phase 1：trajectory 可控

1. `max_public_test_calls=15`、首错即停、首错 observation 和 eval thinking 已实现；最新 2GPU run 暴露主要问题是 thinking 过长，下一步先用 Qwen3 thinking 推荐采样超参重跑 focused smoke：`temperature=0.6`、`top_p=0.95`、`top_k=20`，并保持 `enable_thinking=true`。
2. (P2) 重复代码短路暂不实现，除非 smoke 仍显示重复 judge 是主要瓶颈。
3. 确认并修复 accepted terminal 语义：`submit_solution` accepted 后应立即结束当前 trajectory，不再继续生成 assistant 文本。
4. 确认并修复 verl tool state 持久化：`max_public_test_calls` 和 `max_submissions` 必须在同一道题的一条 trajectory 内跨 tool call 累计，不能因每次 `create/release` 重置。
5. (P5) 多轮交互输出格式已实现为 `structured_output.events`，但当前自定义格式（`type: assistant_text/tool_call/tool_response`）不是标准 HuggingFace chat messages 格式。需要改为标准 `messages` 格式（`role: user/assistant/tool` + `tool_calls` + `tool_call_id`），方便直接喂给 `tokenizer.apply_chat_template()` 或被 HuggingFace `datasets`/TRL 等工具链消费。

   **当前格式 → 标准格式对照：**
   
   ```text
   # 当前 events 格式
   {"type": "assistant_text", "content": "模型思考..."}
   {"type": "tool_call", "name": "run_public_tests", "arguments": {"code": "..."}}
   {"type": "tool_response", "content": "测试结果"}
   
   # 目标 messages 格式 (HuggingFace/OpenAI 标准)
   {"role": "assistant", "content": "模型思考..."}
   {"role": "assistant", "content": null, "tool_calls": [
     {"id": "call_0", "type": "function", "function": {"name": "run_public_tests", "arguments": "{\"code\": \"...\"}"}}
   ]}
   {"role": "tool", "tool_call_id": "call_0", "content": "测试结果"}
   ```
   
   **改动点：** `src/trajectory_parser.py` → 增加 `to_messages()` 函数；`src/verl_runtime_patch.py` → 同时在 jsonl 中写入 `messages` 字段；`structured_output.events` 可暂时保留作为内部中间表示。

### Phase 2：budget 暂不动

6. baseline eval 默认保持 `MAX_PROMPT_LENGTH=4096`、`MAX_RESPONSE_LENGTH=8192`，暂不下调整条 trajectory response budget。当前先通过 validation sampling 超参减少过长 thinking；如果 focused smoke 仍显示 `<think>` 大量未闭合，再设计 custom agent loop 的 per-thinking budget / early-stopping prompt。

### Phase 3：验证与评测

7. Review 当前未提交改动，确认无意外行为。
8. 本地单元级验证已覆盖：
   - public/private judge 遇到首个失败 case 即停止
   - `max_public_test_calls` 不破坏 `max_submissions=5` 语义
   - verl parquet `create_kwargs` 携带 public call cap
9. 训练服务器 focused smoke（建议先 `VAL_MAX_SAMPLES=32`，再视情况缩小到触发 shape mismatch 的边界样本），重点观察 `unclosed_think_rate`、`num_tool_calls=0` 比例、submit 率、accepted rate、accepted 后额外输出长度、public/submission 计数是否跨 tool call 累计、平均输出长度、每题耗时和 GPU 利用率。
10. smoke 通过后调并发参数：`VAL_BATCH_SIZE=16`、`AGENT_WORKERS=16`、`MAX_NUM_SEQS=32` 等。
11. CodeContests held-out baseline：`VAL_MAX_SAMPLES=500`。
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
