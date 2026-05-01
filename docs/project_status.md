# 项目状态

更新日期：2026-05-01

## 当前目标

先验证 eval 环境正确性，再讨论训练路线。

- P0 验收标准见 `docs/specs/eval_acceptance_criteria.md`
- P0 未通过前，不宣称环境调通，不用指标判断模型能力
- eval 环境必须干净——不加 thinking budget、不截断、不动态改写 prompt
- 训练路线（纯 RL、SFT warm-start + RL 等）待 eval 验证通过后再讨论

## 当前主线

- 项目：OJ-like code agent
- 训练数据：`CodeContests`
- 最终测试：`LiveCodeBench`
- 环境工具：`run_public_tests`、`submit_solution`
- eval 入口：`scripts/evaluate_baseline_with_verl.sh`
- eval 路径：复用 verl `main_ppo` validation / agent loop / tool / reward 链路，使用 verl 原生 `ToolAgentLoop`
- 已废弃：全部 thinking budget 控制（per-turn token budget、两阶段生成、short-thinking prompt），详见 `docs/debug/2026-05-01-thinking-budget-detour.md`

## 代码状态

运行时代码：

- `src/verl_runtime_patch.py`：TaskRunner 侧两个 patch——numpy JSON 序列化 + validation 增量 dump
- `src/verl_dataset_adapter.py`：decode JSON-string parquet，按 chat template token 长度过滤 overlong prompt
- `src/verl_tools/oj_tools.py`：OJ 工具实现，`agent_data.code_agent_oj_tool_state` 持久化计数，accepted / submission-limit 标记 terminal
- `src/env/tools.py`：public/private judge 首错即停，`run_public_tests` 上限 15 次，`submit_solution` 上限 5 次
- `src/trajectory_parser.py`：verl decoded output → 标准 `messages` 格式

已删除：

- `src/verl_agent_loop.py`（`CodeAgentToolAgentLoop` 及所有 thinking budget 辅助函数）
- `configs/verl/code_agent_loop.yaml`（自定义 agent loop 注册）
- `tests/test_code_agent_loop.py`（仅测已删除的类）

## 验证状态

本地已验证：

- public/private judge 首错即停
- `max_public_test_calls` 不消耗 `max_submissions`
- verl tool state 跨 `create -> execute -> release` 累计
- accepted / submission-limit 写入 trajectory terminal 标记
- JSON-string prompt decode 后参与 token 长度过滤
- verl 原生 `ToolAgentLoop` 通过 `configs/verl/code_agent_loop.yaml` 注册可用（config 已删除，eval 脚本也已移除该引用）

保留的 baseline 参照：

- `outputs/verl_baseline_eval/codecontests_test_Qwen3-8B_mp4096_mr8192_20260428_201713`：原始 baseline，所有 thinking budget 实验之前的参照点

## 下一步

1. 提 `MAX_RESPONSE_LENGTH` 到 16384，给模型足够 token 预算
2. 跑 500-sample CodeContests held-out eval（不加任何 thinking budget）
3. 对输出跑 P0 审计，确认环境语义正确
4. P0 通过后，正式讨论训练路线

```bash
MAX_RESPONSE_LENGTH=16384 VAL_MAX_SAMPLES=500 \
  bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

## 文档入口

当前优先阅读：

- `README.md`：项目概览
- `AGENTS.md`：操作规范
- `docs/specs/env_protocol.md`：OJ-like 环境协议
- `docs/specs/eval_acceptance_criteria.md`：P0/P1 验收标准
- `docs/operations/gpu_eval_tuning.md`：2xA800 调参记录

决策和调试：

- `docs/decisions/2026-04-29-oj-like-two-tool-protocol.md`：两工具协议决策
- `docs/decisions/2026-04-29-verl-validation-baseline.md`：verl validation baseline 决策
- `docs/decisions/2026-04-30-eval-time-thinking-budget.md`：thinking budget 决策（已废弃）
- `docs/decisions/2026-05-01-sft-warm-start-proposal.md`：SFT warm-start 提案（待讨论）
- `docs/debug/2026-04-28-baseline-eval-batch-fix.md`：batch shape 等调试记录
- `docs/debug/2026-05-01-thinking-budget-detour.md`：thinking budget 弯路总结
