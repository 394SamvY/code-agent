# 项目状态

更新日期：2026-05-13

## 项目背景

- 项目目标：训练 OJ-like code agent，解完整 stdin/stdout 竞赛编程题。
- 训练数据：`CodeContests`。
- 最终测试：`LiveCodeBench`。
- 环境动作：`run_public_tests` 和 `submit_solution`。
- 评测入口：`scripts/evaluate_baseline_with_verl.sh`，复用 verl 原生 validation / `ToolAgentLoop` 路径。
- baseline 评测输出统一保存在 `outputs/verl_baseline_eval/`。
- RL 链路暂定 GRPO，训练参数已先固定为 10-step 计时/跑通配置。

## 当前状态

OJ-like v1 的数据、tool、reward 和 verl validation 评测链路已经接通。此前主要 blocker 是评测效率过低；经过 response budget 和并发参数实验后，当前默认评测参数已经写入 `scripts/evaluate_baseline_with_verl.sh`，两次正式 full eval 都在 2 小时目标内完成。

评测配置和训练配置已经分离：

- baseline eval 默认使用 `configs/verl/eval_qwen3_8b.yaml`。
- GRPO 训练使用 `configs/verl/grpo_qwen3_8b.yaml`。
- `scripts/evaluate_baseline_with_verl.sh` 默认 `CONFIG_NAME=eval_qwen3_8b`，后续训练参数调整不再影响 baseline eval 默认入口。

当前默认评测参数：

```text
MAX_RESPONSE_LENGTH=8192
MAX_MODEL_LEN=12288
VAL_BATCH_SIZE=64
AGENT_WORKERS=64
MAX_NUM_SEQS=64
GPU_MEMORY_UTILIZATION=0.88
VAL_TEMPERATURE=0
VAL_DO_SAMPLE=false
ENABLE_THINKING=true
ROLLOUT_TP=1
```

评测效率实验细节、A/B/C 对比和 full eval 结果见 [`docs/eval_efficiency_experiment_plan.md`](eval_efficiency_experiment_plan.md)。

## 正式评测结果

两次正式评测使用相同默认参数和同一个 SFT checkpoint：

```text
/root/autodl-tmp/code-agent/outputs/verl_sft/qwen3_8b_oj_sft_20260505_032710/global_step_234/huggingface
```

| run | dataset | 样本数 | wall time | validation elapsed | score | acc | assistant tok/s | assistant tokens/problem | 平均 tool calls | 平均 turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_codecontests_test_mr8192_vb64_w64_s64_mem088_greedy` | `codecontests_test` | 499 | 45m02s | 43m25.7s | 0.2213 | 22.04% | 1299.7 | 6786.9 | 3.03 | 7.96 |
| `full_livecodebench_test_mr8192_vb64_w64_s64_mem088_greedy` | `livecodebench_test` | 611 | 55m33s | 52m48.5s | 0.3550 | 35.19% | 1224.3 | 6349.1 | 3.08 | 8.08 |

补充说明：

- `codecontests_test` 原始 500 条中有 1 条因 `MAX_PROMPT_LENGTH=4096` 和 `FILTER_OVERLONG_PROMPTS=true` 被过滤，实际 validation 样本数为 499。
- 两次正式评测的 `generations/0.jsonl` 与 `generations/partial_0.jsonl` 都已完整写出。
- 旧的 16 样本 smoke 外推已不再代表当前效率；当前 full eval 实测比 2 小时目标有充足余量。

## 当前结论

- 评测链路效率 blocker 已解除。完整 CodeContests / LiveCodeBench 评测不再需要按 16 样本 smoke 外推，直接以 full eval 实测为准。
- `MAX_RESPONSE_LENGTH=8192` 是当前默认预算。它显著减少无效长思考，并在固定 128 条对比中保持和 28672 budget 相同的 accepted 数；但 `response_cap_hit` 仍高，说明模型行为还没有真正学会主动短路径。
- `VAL_BATCH_SIZE=64`、`AGENT_WORKERS=64`、`MAX_NUM_SEQS=64`、`GPU_MEMORY_UTILIZATION=0.88` 是当前稳定高吞吐配置。更激进的 `mem0.95` / 高并发尝试没有证明收益，且存在线程资源风险。
- 模型已经能产生 tool call；此前“tool-call rate = 0%”是因为脚本缺少 `actor_rollout_ref.rollout.agent.default_agent_loop=code_agent_tool_agent`，误走了 verl 默认 single-turn agent。
- 当前 reward 只消费 agent loop 记录的结构化 `code_agent_tool_events`，不再解析 `solution_str`。`acc` / `acc_final` 按最后一次 `submit_solution` 事件计算，不是 `max(tool_rewards)`；如果 AC 后又 submit 失败，则 `acc_final=0`。`acc_any`、`best_submit_pass_rate` 和 `last_submit_pass_rate` 会写入 `reward_breakdown`，用于区分“不会解题”和“会解题但不会停止”。RL 训练 reward 只保留 judge outcome 和协议坏模式惩罚，避免过程启发式奖励被 hacking。

## GRPO 训练配置

当前 GRPO 训练入口为 `configs/verl/grpo_qwen3_8b.yaml`。`train_20260511_024102.log` 已完整跑通 10 个 global step，并写出 `checkpoints/global_step_10/actor/huggingface`。

显存相关配置和 OOM 调试记录见 [`docs/grpo_memory_config.md`](grpo_memory_config.md)。

关键参数：

```text
model.path=/root/autodl-tmp/code-agent/outputs/verl_sft/qwen3_8b_oj_sft_20260505_032710/global_step_234/huggingface
data.train_files=./data/verl/codecontests_train.parquet
data.train_batch_size=64
data.max_prompt_length=4096
data.max_response_length=8192
actor_rollout_ref.rollout.n=4
actor_rollout_ref.actor.ppo_mini_batch_size=16
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.ppo_epochs=1
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.rollout.gpu_memory_utilization=0.88
actor_rollout_ref.rollout.max_model_len=12288
actor_rollout_ref.rollout.max_num_batched_tokens=32768
actor_rollout_ref.rollout.max_num_seqs=64
actor_rollout_ref.rollout.agent.num_workers=64
actor_rollout_ref.actor.fsdp_config.param_offload=true
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
actor_rollout_ref.ref.fsdp_config.param_offload=true
actor_rollout_ref.actor.entropy_checkpointing=true
actor_rollout_ref.actor.entropy_from_logits_with_chunking=true
trainer.total_training_steps=10
trainer.test_freq=-1
trainer.save_freq=10
trainer.resume_mode=disable
```

当前 `codecontests_train.parquet` 重新统计为 9323 条。按当前配置：

```text
每个外层 step: 64 prompts * 4 samples = 256 trajectories
每个外层 step: ppo_mini_batch_size=16 prompts => 4 次 actor optimizer update
10-step 计时实验: 640 prompt 槽位，2560 trajectories，40 次 actor optimizer update
完整 1 epoch: floor(9323 / 64) = 145 个外层 step
```

checkpoint 策略：

- actor 只保存 `hf_model`，不保存 FSDP model shard、optimizer 或 extra state。
- 因为只保存 HF 权重，当前禁用自动 resume；如需继续训练，应把上一次产物的 `actor/huggingface/` 目录作为新的 `model.path` 重新启动。

10-step 跑通指标：

```text
平均 step time: 1582.3s = 26.4min
平均 rollout gen: 931.3s
平均 actor update: 425.2s
max_memory_allocated: 59.72GB
max_memory_reserved: 76.36GB
```

按当前 `codecontests_train.parquet` 9323 条、`train_batch_size=64` 估算：

```text
完整 1 epoch: 145 个外层 step
预计耗时: 约 64 小时
```

后续项：

- 如需全量训练，可用当前显存配置继续长跑。
- 若仍在 actor backward OOM，优先按 `docs/grpo_memory_config.md` 启用 activation offload 或临时关闭 entropy bonus；只有 rollout/SGLang 阶段 OOM 时才下调 rollout 并发和 KV-cache 参数。

## 下一步

1. 用当前配置继续长跑完整训练集，同时监控显存、step time、rollout length 和 tool 行为。
2. 长跑稳定后恢复小样本 validation，例如 `val_max_samples=128`、`test_freq` 按阶段设置，并评估训练后 checkpoint。
3. 长跑时重点监控 `acc_any - acc_final`、accepted 后继续工具调用、malformed / unknown tool、response cap hit 和 `bad_pattern_mean`，再决定是否调整 reward 权重。
