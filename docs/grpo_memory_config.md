# GRPO 显存配置记录

更新日期：2026-05-11

本文记录为了让 `configs/verl/grpo_qwen3_8b.yaml` 在 2xA800-80G 上跑通 GRPO 训练所做的显存相关配置。当前目标是先稳定跑完 10-step RL 链路，再基于 rollout 暴露的问题调 reward。

## 10-step 跑通结果

`train_20260511_024102.log` 已用当前配置完整跑完 10 个 global step，并写出 `checkpoints/global_step_10/actor/huggingface`。

关键指标：

```text
timing_s/step mean: 1582.3s = 26.4min
timing_s/gen mean: 931.3s
timing_s/update_actor mean: 425.2s
perf/max_memory_allocated_gb max: 59.72GB
perf/max_memory_reserved_gb max: 76.36GB
response_length/clip_ratio mean: 0.623
prompt_length/max max: 2667
```

对比此前 OOM run：

```text
entropy 优化前 max_memory_allocated: 76.58GB
entropy 优化后 max_memory_allocated: 59.72GB
```

这说明 `entropy_checkpointing` 和 `entropy_from_logits_with_chunking` 已经有效降低 actor backward 峰值，当前配置具备继续跑完整训练集的条件。

## 背景

当前训练形态：

```text
model: Qwen3-8B OJ-like SFT checkpoint
data.train_batch_size: 64
rollout.n: 4
每 step trajectories: 64 * 4 = 256
max_prompt_length: 4096
max_response_length: 8192
max_model_len: 12288
actor.ppo_mini_batch_size: 16
actor.ppo_micro_batch_size_per_gpu: 1
actor.ppo_epochs: 1
```

`train_20260510_205055.log` 和 `train_20260511_010416.log` 都在第 3 个 global step 的 actor update 反向传播阶段 OOM：

```text
verl/workers/actor/dp_actor.py:update_policy
loss.backward()
Tried to allocate 6.15 GiB
PyTorch allocated: ~73.06 GiB
```

这说明主要瓶颈不是 rollout 采样或 ref logprob，而是 actor 对长 trajectory 做 backward 时的 activation、logits/entropy 临时张量和 FSDP 峰值。

## 已启用配置

### Actor 参数和优化器 offload

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      param_offload: true
      optimizer_offload: true
      model_dtype: bf16
```

目的：

- actor 空闲或非训练阶段尽量不常驻 GPU。
- optimizer state 放到 CPU，降低训练阶段稳态显存。

日志中已经看到 actor init、compute_log_prob、update_actor 后的 offload 记录。

### Ref 参数 offload

```yaml
actor_rollout_ref:
  ref:
    fsdp_config:
      param_offload: true
      model_dtype: bf16
```

目的：

- ref 只在 rollout 后计算 KL/ref logprob，其他阶段不应长期占用 GPU。
- 减少 actor update 前后的常驻显存压力。

`train_20260511_010416.log` 中最终配置已显示 `ref.fsdp_config.param_offload: True`。但第 3 step OOM 仍发生在 actor backward，因此 ref offload 只能缓解常驻显存，不能直接解决满长 actor backward 峰值。

### Actor 最小 micro batch

```yaml
actor_rollout_ref:
  actor:
    ppo_micro_batch_size_per_gpu: 1
    ppo_mini_batch_size: 16
    ppo_epochs: 1
```

目的：

- 每张 GPU 一次只反传 1 条 trajectory，降低单次 forward/backward 显存。
- mini-batch 和 epoch 保持正式训练设置，不靠降低训练量绕过问题。

注意：即使 micro batch 是 1，单条 trajectory 接近 `prompt + response ~= 10k-12k tokens` 时，actor backward 仍可能 OOM。

### 梯度 checkpointing

```yaml
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: true
```

目的：

- transformer 层反向传播时重算部分 activation，降低保存 activation 的显存。

这是基础省显存项，已保留开启。

### Entropy 显存优化

```yaml
actor_rollout_ref:
  actor:
    entropy_coeff: 0.001
    entropy_checkpointing: true
    entropy_from_logits_with_chunking: true
```

目的：

- 保留 entropy bonus，但降低其 logits/softmax 临时显存。
- `entropy_from_logits_with_chunking=true`：分块计算 entropy，避免一次性对所有 token 的 `[tokens, vocab]` logits 做 softmax。
- `entropy_checkpointing=true`：entropy backward 时重算部分中间结果，减少 forward 阶段保存的中间张量。

触发原因：

- 满长或近满长 response 很多。`train_20260511_010416.log` 中：

```text
step1 response_length/max=8192, clip_ratio=0.5859375
step2 response_length/max=8192, clip_ratio=0.6953125
```

- Qwen3 vocab 约 151k，10k tokens 的 logits/entropy 临时张量已经是数 GiB 量级，和 OOM 中 `Tried to allocate 6.15 GiB` 的量级一致。

该配置只影响 entropy 实现，不改变采样、response length、batch size 或 GRPO group size。

### Logprob micro batch

```yaml
actor_rollout_ref:
  actor:
    ppo_micro_batch_size_per_gpu: 1
  ref:
    log_prob_micro_batch_size_per_gpu: 1
  rollout:
    log_prob_micro_batch_size_per_gpu: 1
```

目的：

- old logprob、ref logprob 和 actor update 都使用最小 per-GPU micro batch，避免 logprob 计算阶段额外 OOM。

当前 OOM 不是发生在 old/ref logprob，但这些配置保留为低风险显存保护。

### Rollout 采样高吞吐配置

```yaml
actor_rollout_ref:
  rollout:
    gpu_memory_utilization: 0.88
    max_num_batched_tokens: 32768
    max_num_seqs: 64
    agent:
      num_workers: 64
```

目的：

- 训练 rollout 采样阶段相对独立，复用 baseline eval 已验证的高吞吐配置。
- `train_20260511_010416.log` 中采样耗时从旧 run 的约 `1167-1212s` 降到 `892-1017s`。

注意：

- 这组配置主要改善采样吞吐，不解决 actor backward OOM。
- `gpu_memory_utilization=0.88` 仍留有一定余量；此前更激进的评测参数没有证明稳定收益。

## 暂未启用配置

### Activation offload

```yaml
actor_rollout_ref:
  model:
    enable_activation_offload: true
```

当前未启用。

原因：

- 这是更强的 actor backward 显存保护，但会通过 CPU/GPU 搬运 activation 明显拖慢 update。
- 目前先只开 entropy checkpoint/chunking，观察是否足够越过第 3 step。

如仍在 actor backward OOM，下一步优先启用该项。机器 CPU 内存充足，具备启用条件。

### 关闭 entropy bonus

```yaml
actor_rollout_ref:
  actor:
    entropy_coeff: 0.0
```

当前未启用。

原因：

- 关闭后会直接跳过 entropy 计算，省显存且可能更快。
- 但这会改变训练目标。当前先保留 `entropy_coeff=0.001`，只改 entropy 的实现方式。

如 entropy checkpoint/chunking 后仍 OOM，可作为比 activation offload 更快但更改训练目标的备选。

## 当前判断

- ref offload 已生效，但 OOM 根因仍是 actor backward 峰值。
- 前两步能过不代表配置稳定，因为前两步 prompt max 只有约 2k；如果后续 batch 中 prompt 更长，同时 response 接近 8192，就会逼近 `max_model_len=12288` 的 worst case。
- 当前最小侵入的修复是保留正式训练参数，只开启 entropy checkpoint/chunking。

继续长跑时重点观察：

```text
perf/max_memory_allocated_gb 是否继续维持在 60GB 左右
perf/max_memory_reserved_gb 是否长期贴近 80GB
timing_s/update_actor 是否异常上升
timing_s/gen 是否维持 900-1000s 量级
response_length/clip_ratio 和 num_turns 是否继续偏高
```
