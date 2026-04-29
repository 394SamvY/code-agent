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

截至本次同步，工作区存在 3 个未提交代码改动：

| 文件 | 当前作用 |
| --- | --- |
| `scripts/verl_main_wrapper.py` | 记录 driver 进程与 Ray CPU TaskRunner actor 的 patch 安装链路。 |
| `src/verl_dataset_adapter.py` | 记录 JSON-string Parquet schema 到 verl `RLHFDataset` 的 decode / prompt length filter 适配逻辑。 |
| `src/verl_runtime_patch.py` | 记录 numpy JSON 序列化 patch 与 validation partial dump patch 的安装点和语义。 |

这些改动主要是围绕 2026-04-28 的 baseline eval 调试结论补充代码内说明，并涉及 verl runtime/dataset 适配路径。合入前仍应做一次 focused diff review 和 smoke verification。

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

1. (P1) **公开测试无限循环**：45/56 条样本 score=0，大部分从未调用 `submit_solution`。典型样本 23 次 `run_public_tests`、0 次 `submit_solution`，公测一直失败但不提交。方案：`max_public_test_calls` 上限（默认 15），达到后返回短提示引导提交。加在 `RunPublicTestsTool` 的 instance state 里。

2. (P2) **重复代码消耗**：模型反复用相同代码调 `run_public_tests`，每次重新跑 subprocess judge 和生成 observation。方案：相同 `tool_name + code_hash` 结果缓存，连续相同代码返回短 observation。

3. (P3) **Observation 累积过长**：平均 20,760 字符，`run_public_tests` 展开所有失败 public case（含 full input/expected/stdout/stderr），随多轮不断累积进上下文。方案：压缩 observation（只展示首个失败 case 或设总字符预算）。

4. (P4) **`max_response_length=8192` 偏大**：给模型过多空间浪费，也占 KV cache。方案：trajectory 可控后降到 4096 或 2048 say smoke，相应降低 `MAX_MODEL_LEN` 释放 KV cache。

**即将处理（P5/P6，已讨论待排期）**

7. (P5) **输出格式不直观**：`partial_0.jsonl` 把整条 trajectory 的 prompt 和 response 各存为一个字符串，多轮交互全糊在一起。方案：离线解析 output 文本，用正则拆分 `<tool_call>` / `<tool_response>` 边界，还原成结构化 turns（不做在线改动，只加事后分析脚本或 partial dump 时做）。

8. (P6) **评测未开启 thinking**：`enable_thinking=false`（eval 脚本第 241 行 + YAML 训练配置）。base model 缺乏调试能力，开 thinking 可能帮助它在 tool-based 交互中做出更好的修复决策。方案：评测先开 `enable_thinking=true`，训练保持关闭，等基线结果出来后再实验。

**RL 训练解决（当前不处理）**

9. **Base model 改不对代码**：反馈信息足够（具体报错 + input/expected/stdout），Qwen3-8B 缺乏 code debugging 能力。属模型能力差距，应通过 RL 训练提升。

10. **Tool call JSON 格式错误**：7 次 decode 失败，base model 未经 tool calling 训练。同样留到 RL 训练解决。

**已解决**

7. **500 条没跑完**：可能是手动中断，不追查。

## 验证状态

最近记录在 `docs/debug/verl_baseline_eval_debug_2026-04-28.md` 的验证：

- 1GPU partial dump smoke 通过，`partial_0.jsonl` 和 `0.jsonl` 均写出 2 条。
- 2GPU 统一调度 smoke 通过，两个 SGLang server 分别绑定 GPU0 和 GPU1，`partial_0.jsonl` 和 `0.jsonl` 均写出 4 条。
- 清理 sharded fallback 和 `sitecustomize.py` 后，2GPU smoke 再次通过。

2026-04-29 对 `codecontests_test_Qwen3-8B_mp4096_mr8192_20260428_201713` 做了离线输出分析，但没有重新运行远端 GPU baseline。任何“当前代码已完全通过正式 baseline”或“效率问题已解决”的表述都应等下一次真实 A800 smoke / full eval 后再写入。

## 下一步

实现顺序按优先级排列：

### Phase 1：trajectory 可控（P1/P2/P3，先讨论后实现）

1. (P1) `max_public_test_calls` 上限：在 `RunPublicTestsTool` 的 instance state 里加计数器，默认 15。达到后返回短 observation 引导提交，不强制终止 trajectory。配置方式待讨论：写死 / tools_kwargs / 环境变量。

2. (P3) 压缩 `run_public_tests` observation：默认只展示首个失败 case，长 input / expected / stdout / stderr 摘要裁剪。减少上下文污染。

3. (P2) 重复代码短路：相同 `tool_name + code_hash` 结果缓存；连续相同代码返回短 observation，避免重复 subprocess judge。

4. (P6) 评测开启 thinking：`enable_thinking=false` → `true`（仅评测，训练保持关闭）。

5. (P5) 多轮交互输出格式：离线解析 output 文本，拆成结构化 turns，方便人工查看和调试。

### Phase 2：释放资源（P4）

4. trajectory 可控后，把 `MAX_RESPONSE_LENGTH` 从 8192 降到 4096 或 2048 做 smoke，相应降低 `MAX_MODEL_LEN` 释放 KV cache。

### Phase 3：验证与评测

5. Review 当前 3 个未提交代码改动，确认无意外行为。
6. 本地补单元级验证：
   - 相同代码重复调用 public tests 不重复跑 judge
   - 长 observation 被稳定裁剪
   - hard cap 不破坏 `max_submissions=5` 语义
7. 训练服务器 focused smoke（`VAL_MAX_SAMPLES=4`），观察 tool call、submit 率、每题耗时、GPU 利用率。
8. smoke 通过后调并发参数：`VAL_BATCH_SIZE=16`、`AGENT_WORKERS=16`、`MAX_NUM_SEQS=32` 等。
9. CodeContests held-out baseline：`VAL_MAX_SAMPLES=500`。
10. `livecodebench_test`。
11. 结果同步回本文。

## 交接提示

下一位 agent 开工时建议先读：

1. `README.md`
2. `AGENTS.md`
3. `docs/project_status.md`
4. `docs/specs/env_protocol.md`
5. `docs/specs/verl_parquet_dataset_analysis.md`
6. `docs/debug/verl_baseline_eval_debug_2026-04-28.md`

如果任务是继续 baseline eval，优先检查 `scripts/evaluate_baseline_with_verl.sh`、`scripts/verl_main_wrapper.py`、`src/verl_dataset_adapter.py`、`src/verl_runtime_patch.py` 和 `src/verl_tools/oj_tools.py`。
