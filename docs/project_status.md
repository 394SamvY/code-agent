# 项目状态

更新日期：2026-05-01

## 当前阶段

eval 环境已验证正确，进入训练方案设计阶段。

## eval 结论

三组 smoke（mr=8192/16384/16384+94GB）跑完，结论：Qwen3-8B base model `enable_thinking=true` 下 tool-call rate = 0%。模型以 `<think>` 开始分析问题但从不闭合、从不调用工具。预算越大 thinking 越长但不改变行为。eval 环境本身正确，样本正常启停、无 OOM、0.jsonl 完整写出。

详细数据见 `docs/debug/2026-05-01-baseline-smoke-results.md`。

## 当前主线

- 项目：OJ-like code agent
- 训练数据：`CodeContests`，最终测试：`LiveCodeBench`
- 环境工具：`run_public_tests`、`submit_solution`
- eval 入口：`scripts/evaluate_baseline_with_verl.sh`，使用 verl 原生 `ToolAgentLoop`
- 已废弃：全部 thinking budget 控制，详见 `docs/debug/2026-05-01-thinking-budget-detour.md`

## 代码状态

- `src/verl_runtime_patch.py`：numpy JSON 序列化 + validation 增量 dump
- `src/verl_dataset_adapter.py`：decode JSON-string parquet，按 token 长度过滤 overlong prompt
- `src/verl_tools/oj_tools.py`：OJ 工具，state 持久化计数，accepted / submission-limit 标记 terminal
- `src/env/tools.py`：judge 首错即停，`run_public_tests` 上限 15，`submit_solution` 上限 5
- `src/trajectory_parser.py`：verl output → 标准 `messages`

## 待讨论：SFT warm-start 方案

冷启动问题：base model 从不输出 `</think>` 和 `<tool_call>`，全部 reward = 0，GRPO 无梯度可追。

当前 SFT warm-start 提案见 `docs/decisions/2026-05-01-sft-warm-start-proposal.md`，核心思路：

1. teacher 模型（更强的模型）在 CodeContests train 上按 OJ 协议生成正确 trajectory
2. 用这些 trajectory 对 Qwen3-8B 做 LoRA SFT，教会模型 think → `</think>` → `<tool_call>` 的基本行为
3. SFT 后 eval 验证 tool-call rate > 0
4. 再上 GRPO + 过程奖励精调

待定问题：

- teacher 模型选哪个？生成多少条 trajectory？
- SFT 具体怎么训？（LoRA rank、epoch、mask 策略）
- SFT 后是否需要先验证再上 GRPO？验证标准是什么？
- 是否需要先在 CodeContests valid 上试点，还是直接 train 全量生成？

## 文档入口

- `docs/debug/2026-05-01-baseline-smoke-results.md`：三组 smoke 详细分析
- `docs/debug/2026-05-01-thinking-budget-detour.md`：thinking budget 弯路总结
- `docs/decisions/2026-05-01-sft-warm-start-proposal.md`：SFT warm-start 提案
- `docs/specs/env_protocol.md`：OJ-like 环境协议
- `docs/operations/gpu_eval_tuning.md`：2xA800 调参记录
