# Training Run History

每次 run 的关键配置差异和结果对照。
配置从日志中的 config dump 提取，完整 YAML 快照从 run3 起保存。

## 项目结构说明

### `configs/verl/` 目录

该目录下大量文件是 **symlink（符号链接）**，指向 verl 安装目录的内置配置，不占磁盘空间。
它们是为了让 Hydra 的 `defaults` 配置继承机制正常工作。

```
configs/verl/
├── grpo_qwen3_8b.yaml          ← 我们的训练配置（唯一需要编辑的文件）
├── tool_config.yaml             ← 工具 schema（启动时自动生成）
├── ppo_trainer.yaml → verl/...  ← symlink: Hydra defaults 继承入口
├── actor/ → verl/...            ← symlink: actor 子配置
├── rollout/ → verl/...          ← symlink: rollout 子配置
├── reward/ → verl/...           ← symlink: reward 子配置
├── ref/ → verl/...              ← symlink: ref 子配置
├── ... (其余 symlink)
```

**为什么需要这些 symlink**：`grpo_qwen3_8b.yaml` 通过 `defaults: [ppo_trainer, _self_]`
继承 verl 的完整默认配置，Hydra 要求所有被引用的配置文件都在 `--config-path` 目录下可达。
这些 symlink 把 verl 内置配置映射过来，实现零拷贝的配置继承。删除任何 symlink 会导致启动失败。

---

## Run 1 — `train_run1_oom.log`

**结果**: OOM 崩溃（SGLang 恢复显存失败）
**走到**: Step 0 的 update_policy 之后，SGLang resume_memory_occupation 时 OOM

| 配置项 | 值 |
|--------|-----|
| param_offload | **false** |
| optimizer_offload | **false** |
| use_kl_loss | **true** ❌ |
| rollout.n | **8** |
| ppo_micro_batch_size_per_gpu | **1** |
| total_epochs | **10** |
| save_freq / test_freq | **5 / 5** |
| max_actor_ckpt_to_keep | 1 |
| tool_config_path | **null** ❌ |
| reward.num_workers | **8** (默认) |
| agent.num_workers | **8** (默认) |

**死因**: 训练阶段模型+优化器占满 GPU，SGLang 无法恢复 KV cache

**额外瓶颈**: ref 模型 compute_log_prob 极慢（~7min/step），
因为需要把完整 8B ref 模型加载到 GPU 做前向，用于计算 KL 散度惩罚。
→ Run 2 通过 `use_kl_loss: false` 解决：GRPO 不依赖 KL 约束（靠 group 内 reward 排名估计 advantage），
关掉后完全跳过 ref log_prob 计算，每步省 ~7 分钟 + 一次完整前向的显存

---

## Run 2 — `train_run2.log`

**结果**: RewardLoopWorker SIGSEGV 崩溃
**走到**: 初始 validation 阶段，调用已死的 reward_loop_worker_6

| 配置项 | 值 | 相比 Run1 |
|--------|-----|-----------|
| param_offload | **true** | ✅ 修复 |
| optimizer_offload | **true** | ✅ 修复 |
| use_kl_loss | **false** | ✅ 优化 |
| rollout.n | **4** | ✅ 优化 |
| ppo_micro_batch_size_per_gpu | **2** | ✅ 优化 |
| total_epochs | **7** | ✅ 优化 |
| save_freq / test_freq | **7 / 7** | 调整 |
| max_actor_ckpt_to_keep | 1 | 不变 |
| tool_config_path | **null** ❌ | 未修 |
| reward.num_workers | **8** (默认) | 未修 |
| agent.num_workers | **8** (默认) | 未修 |

**死因**: 22+ 个进程并发 fork，reward worker SIGSEGV；且 tool_config_path 为 null

---

## Run 3 — `train_20260406_004845.log` | `config_run3.yaml`

**结果**: KeyError: 'reason'（compute_score 返回 key 不一致）
**走到**: 初始 validation，工具成功加载并调用，但 reward 后处理崩溃

| 配置项 | 值 | 相比 Run2 |
|--------|-----|-----------|
| param_offload | true | 不变 |
| optimizer_offload | true | 不变 |
| use_kl_loss | false | 不变 |
| rollout.n | 4 | 不变 |
| ppo_micro_batch_size_per_gpu | 2 | 不变 |
| total_epochs | 7 | 不变 |
| save_freq / test_freq | 7 / 7 | 不变 |
| max_actor_ckpt_to_keep | 1 | 不变 |
| tool_config_path | **configs/verl/tool_config.yaml** | ✅ 修复 |
| reward.num_workers | **2** | ✅ 修复 |
| agent.num_workers | **4** | ✅ 修复 |

**死因**: `src/reward.py` 的 `compute_score` 两个分支返回不同的 key 集合

**进展**: SIGSEGV 已消除、工具成功加载并被模型调用

---

## Run 4 — `train_20260406_005402.log` | `config_20260406_005402.yaml` | [详细分析](run4_analysis.md)

**结果**: Step 0 完成训练但保存 rollout data 时崩溃；validation reward 仅 3.3%
**走到**: validation ✅ → rollout ✅ → compute_log_prob ✅ → update_policy ✅ → _log_rollout_data ❌

| 配置项 | 值 | 相比 Run3 |
|--------|-----|-----------|
| compute_score keys | 统一 | ✅ 修复 |
| apply_chat_template_kwargs | **无** ❌ | Qwen3 thinking 吃掉所有 token |
| response_length | **512** ❌ | 不够 |

**死因1**: `TypeError: Object of type int64 is not JSON serializable`（Ray worker 进程中）
**死因2**: thinking mode 默认开启，validation reward 仅 3.3%（baseline 72.4%）

---

## Run 5 — `train_20260406_010923.log` | `config_20260406_010923.yaml`

**结果**: 训练成功运行 7 步后，保存 checkpoint 时磁盘写满崩溃
**走到**: validation ✅ (reward=73.2%) → Step 1~7 训练 ✅ → save_checkpoint ❌ 磁盘满

| 配置项 | 值 | 相比 Run4 |
|--------|-----|-----------|
| enable_thinking | **false** | ✅ 修复（reward 3.3%→73.2%）|
| response_length | **1024** | ✅ 修复（512→1024）|
| numpy JSON patch | `.pth` 全局生效 | ✅ 修复 |
| checkpoint.save_contents | `[model, optimizer, extra]` ❌ | 默认，太大 |

**初始 validation**: reward=0.732, exec_reward=0.719, num_tool_calls=1.39, num_turns=4.62
（与 baseline 72.4% 对齐，确认 enable_thinking=false 生效）

**训练趋势（7 步 rollout data）**:
```
Step 1: reward=0.776  perfect=73%  zero=22%
Step 5: reward=0.824  perfect=78%  zero=17%  ← 峰值
Step 6: reward=0.763  perfect=72%  zero=23%
```
6 步内未见显著提升，波动为主。

**死因**: `save_freq: 7` 触发 checkpoint 保存，FSDP checkpoint（model+optimizer+extra）
需要 ~60-100GB，50GB 磁盘写满 → `RuntimeError: PytorchStreamWriter failed writing file`

---

## Run 6 — `train_20260423_013404.log` | `config_20260423_013404.yaml`

**结果**: 训练 14/14 steps 完成；最终 validation reward 约 78.22%；日志收尾有 DataLoader worker killed 噪声，但随后打印 `Training complete`

**走到**: initial validation ✅ → Step 1~14 训练 ✅ → checkpoint 保存 ✅（日志显示 rank0/rank1 model 保存）→ final validation ✅ → 收尾 worker killed ⚠️

**修复**:
1. `actor.checkpoint.save_contents: [model]` — 只保存模型权重（~16GB），不保存优化器状态
2. 清理 Run 5 写坏的 checkpoint（35GB）：`rm -rf outputs/verl_grpo/checkpoints/*`

**关键日志**:
- Step 7 保存：`global_step_7/actor/model_world_size_2_rank_0.pt` 和 rank 1 保存成功
- Step 14 保存：`global_step_14/actor/model_world_size_2_rank_0.pt` 和 rank 1 保存成功
- 旧 checkpoint 清理：`Checkpoint manager remove previous save local path: .../global_step_7/actor`
- 本地未同步 `outputs/verl_grpo/checkpoints/`，需要在远端 `/root/autodl-tmp/code-agent/outputs/verl_grpo/checkpoints/global_step_14/actor/` 复核

**Validation**:
```
Initial: reward=0.7815  score=0.7815  exec_reward=0.7593  fix_reward=0.0222
Final:   reward=0.7822  score=0.7822  exec_reward=0.7556  fix_reward=0.0267
```

**训练趋势（14 步 rollout data，本地每步 512 条）**:
```
Step  1: score=0.776  perfect=72.7%  zero=21.9%  calls=1.47
Step  5: score=0.827  perfect=78.7%  zero=16.8%  calls=1.38
Step  7: score=0.758  perfect=72.3%  zero=23.6%  calls=1.49
Step 11: score=0.858  perfect=81.1%  zero=15.6%  calls=1.52
Step 14: score=0.852  perfect=80.9%  zero=15.0%  calls=1.50
```

**判断**:
- rollout 训练样本均值后半段有上升，step 11/14 的 perfect 超过 80%
- held-out validation 从 0.7815 到 0.7822，几乎没有提升；需要正式 eval 判断是否泛化
- 日志中仍有少量 `Failed to decode tool call`，但未阻断训练

**下一步**:
1. 远端确认 `global_step_14/actor` checkpoint 完整
2. 导出 HF 模型并跑 MBPP/HumanEval multi-turn 正式评测
3. 如果正式评测无提升，优先分析数据重复、reward 信号和训练集/验证集分布，而不是直接延长训练
