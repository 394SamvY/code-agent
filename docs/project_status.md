# 项目状态与交接

更新日期：2026-05-01

本文只记录当前目标、验收状态和下一步。历史 run 分析、失败过程和调参细节不要继续堆在这里；需要追溯时读 `docs/debug/`、`docs/decisions/` 和 `docs/operations/`。

## 当前目标

当前目标不是先追 baseline 指标，而是先证明 baseline eval 环境可信。

严格验收入口：

- `docs/specs/eval_acceptance_criteria.md`

核心判断标准：

- P0 未通过前，不宣称 eval 环境已经调通。
- P0 未通过前，不用 accepted rate、score mean、no-tool rate 判断模型能力。
- P0 必须基于真实 verl `AgentLoopWorker` 路径，不接受只在 TaskRunner patch、离线 parser 或本地环境成立的证据。
- 2xA800 效率调优是 P1，必须排在 P0 环境正确性之后。

## 当前主线

- 项目目标：OJ-like code agent，只做 RL，不做 SFT。
- 训练数据：`CodeContests`。
- 最终测试：`LiveCodeBench`。
- 环境工具：`run_public_tests` 和 `submit_solution`。
- baseline 入口：`scripts/evaluate_baseline_with_verl.sh`。
- focused smoke 入口：`scripts/evaluate_2xa800_32_debug.sh`。
- eval 路径：复用 verl `main_ppo` validation / agent loop / tool / reward 链路。
- 当前默认：`enable_thinking=true`、`MAX_RESPONSE_LENGTH=8192`、first/followup assistant turn budget `3072/2048`。
- 已废弃：`CODE_AGENT_PROMPT_STYLE` / short-thinking prompt soft control；eval 不应动态改写 system prompt。

## 当前实现状态

已接入的关键控制：

- `src/verl_agent_loop.py`：`CodeAgentToolAgentLoop` 在真实 AgentLoopWorker 侧控制 first/followup assistant turn token budget，并处理 tool terminal。
- `configs/verl/code_agent_loop.yaml`：注册 `code_agent_tool_agent`。
- `src/verl_tools/oj_tools.py`：通过 `agent_data.code_agent_oj_tool_state` 持久化 public/submission 计数；accepted 或 submit 次数耗尽时标记 terminal。
- `src/env/tools.py`：public/private judge 首错即停；`run_public_tests` 默认 15 次上限；`submit_solution` 默认 5 次上限。
- `src/verl_dataset_adapter.py`：decode JSON-string parquet 字段，并按真实 chat template token 长度过滤 overlong prompt；不再修改 prompt 内容。
- `src/trajectory_parser.py` / `scripts/parse_verl_generations.py`：把 verl decoded output 转为标准 `messages`，便于审计 multi-turn 交互。

当前 P0 结果：

- 2026-05-01，32-sample strict smoke 已通过真实 verl `AgentLoopWorker` 路径。
- 有效 run：`outputs/verl_baseline_eval/p0_strict_32_tokenbudget_20260501_0126/generations/0.jsonl`。
- 审计命令：`python3 scripts/audit_eval_p0.py outputs/verl_baseline_eval/p0_strict_32_tokenbudget_20260501_0126/generations/0.jsonl --expected-rows 32`。
- 审计结果：`records=32`、`max_valid_tool_calls=3`、`sample_think_then_tool_rows=[0,1,2]`、`sample_failed_feedback_rows=[3,5,8]`，`P0 audit passed`。
- Qwen3 thinking budget 已按 two-pass 语义接入：第一段只用 token budget，不传 SGLang string stop；预算耗尽且未见 `</think>` 时插入 early-stopping prompt 和 `</think>`，再继续生成最终 answer / tool call。
- 失败排查结论：早期崩溃不是 OOM，也不是并发过高；根因是 SGLang token-in/token-out 路径下 scheduler 没有 tokenizer，传 `stop=["</think>"]` 会进入 string-stop decode 分支并触发 `AttributeError: 'NoneType' object has no attribute 'decode'`。

## 验证状态

本地已验证过的单元级能力：

- public/private judge 首错即停。
- `max_public_test_calls` 不消耗 `max_submissions`。
- verl tool state 能跨 `create -> execute -> release` 累计。
- accepted / submission-limit 会写入 trajectory terminal 标记。
- JSON-string prompt 会 decode 后再参与 token length 过滤。
- adapter 即使看到旧 `CODE_AGENT_PROMPT_STYLE=short_thinking` 环境变量，也不会动态修改 prompt。

最近可参考但不能作为最终 baseline 的 A800 记录：

- `outputs/verl_baseline_eval/debug32_code_agent_loop_f3072_u2048_verify_v2`：32 条完整写出，证明 custom agent loop 已进入真实 AgentLoopWorker；但仍需按新的 P0 验收标准重新审计。
- `outputs/verl_baseline_eval/codecontests_test_500_shortthink_tb3072`：499 条 debug run，不作为正式 baseline；该 run 暴露过旧 hard budget / terminal stop 没进 AgentLoopWorker 的问题。

## 下一步

1. 跑 500-sample CodeContests held-out eval：

```bash
VAL_MAX_SAMPLES=500 bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

2. 对 500-sample 输出继续跑 P0 审计脚本，确认扩大样本后仍没有环境语义回归。

3. 500-sample 也通过后，再进入 P1：2xA800 效率调优。

### 下一步：替换为 SGLang 原生 thinking budget

当前 `_handle_generating_state_with_thinking_stop`（200 行两阶段生成）原理是两次 `generate()` 调用，各自限制 `max_new_tokens`——thinking 阶段 1024，action 阶段用剩余预算。

但 SGLang 已原生支持 per-request thinking budget，通过 `Qwen3ThinkingBudgetLogitProcessor` + `custom_params.thinking_budget`。当前安装的 SGLang 版本已包含 `Qwen3ThinkingBudgetLogitProcessor`。

替换后 `verl_agent_loop.py` 可删掉 `_handle_generating_state_with_thinking_stop` 整个方法，回到 `super()._handle_generating_state()`，只靠 `sampling_params` 注入 `thinking_budget`。

实施步骤：

1. **验证 SGLang 内部 API 是否在 `sampling_params` 里认 `thinking_budget`**：
   - 在 `_sampling_params_with_turn_budget` 中临时加 `"thinking_budget": thinking_budget`
   - 跑一条样本，确认 `<\think>` 长度受限
2. **如果内部 API 不认**：patch `async_sglang_server.py` 的 `generate()` 方法，把 `custom_logit_processor` 传入 `GenerateReqInput`
3. **Server 启动参数**：确认 `--enable-custom-logit-processor --reasoning-parser qwen3` 已加
4. **成功后清理**：删除两阶段生成代码，参数从 `CODE_AGENT_*` 环境变量改为 SGLang 原生 `thinking_budget`

## 文档入口

当前应优先阅读：

- `README.md`：项目概览和常用命令。
- `AGENTS.md`：agent 操作规范和硬约束。
- `docs/specs/env_protocol.md`：OJ-like 环境协议。
- `docs/specs/eval_acceptance_criteria.md`：baseline eval P0/P1 验收标准。
- `docs/specs/verl_parquet_dataset_analysis.md`：verl parquet schema 和数据快照。
- `docs/operations/gpu_eval_tuning.md`：2xA800 运行和调参记录。

历史追溯：

- `docs/debug/verl_baseline_eval_debug_2026-04-28.md`：batch shape、agent loop、generation dump 等调试记录。
- `docs/decisions/2026-04-30-eval-time-thinking-budget.md`：per-turn budget 决策；其中 soft prompt control 已在 2026-05-01 废弃。
- `docs/references/verl_ray_agent_loop_reading_guide.md`：verl / Ray agent loop 阅读路线。
