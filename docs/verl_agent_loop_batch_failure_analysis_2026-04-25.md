# verl agent loop batch failure analysis 2026-04-25

本文记录 `scripts/evaluate_baseline_with_verl.sh codecontests_test` 在全量 baseline eval 中反复失败的原因。

结论先行：这不是显存不足，也不是 OJ tool / reward / parquet JSON 适配问题；核心是当前 `verl 0.7.1` experimental multi-turn agent loop 对同一个 worker 内多条样本的最终 prompt 长度处理不安全。只要一个 eval batch 中不同样本经过 multi-turn 后的最终 `prompt_ids` 长度不同，`AgentLoopWorker._postprocess()` 会直接 `torch.cat` 报错。

## 现象

第一次全量运行：

```text
run: outputs/verl_baseline_eval/codecontests_test_Qwen3-8B_mp2048_mr2048_20260425_231506
data.val_batch_size=4
actor_rollout_ref.rollout.agent.num_workers=2
data.filter_overlong_prompts=False
Size of val dataloader: 125
失败前完成 validation generation end: 42
```

第二次全量运行：

```text
run: outputs/verl_baseline_eval/codecontests_test_Qwen3-8B_mp2048_mr2048_20260425_233507
data.val_batch_size=8
actor_rollout_ref.rollout.agent.num_workers=4
data.filter_overlong_prompts=True
Size of val dataloader: 63
失败前完成 validation generation end: 21
```

两次最终错误相同：

```text
RuntimeError: Sizes of tensors must match except in dimension 0.
Expected size 3475 but got size 2048 for tensor number 1 in the list.
```

堆栈核心位置：

```text
verl/experimental/agent_loop/agent_loop.py
  AgentLoopWorker.generate_sequences()
  AgentLoopWorker._postprocess()
  prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
```

## 不是哪些问题

不是 GPU OOM：

- 失败前 GPU 显存约 `54GB / 80GB` 每卡。
- 失败后 GPU 正常释放。
- 错误是 tensor shape mismatch，不是 CUDA OOM。

不是 parquet JSON string 适配问题：

- 日志中已显示 `Using dataset class: OJLikeRLHFDataset`。
- `prompt` / `reward_model` / `extra_info` 已能被 verl 读取。
- tool schema 也已正常进入 agent loop。

不是初始 prompt 长度过滤完全没生效：

- 第二次运行中 `data.filter_overlong_prompts=True` 生效。
- dataloader 从 `125` 个 batch 变为 `63` 个 batch，说明过滤确实改变了 eval 集。
- 但错误仍然存在，因为错误触发点不是单纯的初始 prompt tokenize 阶段。

## 根因链路

当前 `AsyncRolloutManager.generate_sequences()` 会把一个 validation batch 按 agent worker 数切块：

```python
chunkes = prompts.chunk(len(self.agent_loop_workers))
outputs = await asyncio.gather(
    *[
        worker.generate_sequences.remote(chunk)
        for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
    ]
)
output = DataProto.concat(outputs)
```

在每个 `AgentLoopWorker.generate_sequences()` 内部，它会对 chunk 中的每条样本并发跑 agent loop：

```python
for i in range(len(batch)):
    tasks.append(asyncio.create_task(self._run_agent_loop(...)))
outputs = await asyncio.gather(*tasks)
output = self._postprocess(outputs, input_non_tensor_batch=batch.non_tensor_batch)
```

然后 `_postprocess()` 直接拼接多个 trajectory 的 tensor：

```python
prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
```

这个实现隐含假设：同一个 worker 的所有样本在 agent loop 结束后 tensor 宽度完全一致。

但 OJ-like multi-turn 任务中，这个假设不成立：

- 不同题目的初始 prompt 长度不同。
- 模型是否调用 tool、调用几次 tool、tool observation 长度都不同。
- `run_public_tests` / `submit_solution` 的 observation 会把失败样例、stdout、stderr 等追加进上下文。
- 即使初始 prompt 都小于 `max_prompt_length`，multi-turn 后最终 prompt 长度也会分化。

因此某个 chunk 里出现：

```text
sample A final prompt_ids width = 3475
sample B final prompt_ids width = 2048
```

`torch.cat(..., dim=0)` 就会报：

```text
Expected size 3475 but got size 2048
```

## 为什么提高并行度更容易触发

当前并行配置：

```text
VAL_BATCH_SIZE=8
AGENT_WORKERS=4
```

意味着每个 worker 平均处理 `8 / 4 = 2` 条样本。只要这 2 条样本最终 prompt 长度不同，就会在 worker 内 `_postprocess()` 失败。

之前配置：

```text
VAL_BATCH_SIZE=4
AGENT_WORKERS=2
```

同样是每个 worker 处理 2 条样本，所以也会失败。

把 `AGENT_WORKERS` 提到 `VAL_BATCH_SIZE` 只能避免 worker 内多样本 `_postprocess()` 失败，但仍可能在下一层 `DataProto.concat(outputs)` 处失败，因为 `DataProto.concat()` 也直接：

```python
new_batch = torch.cat(batch_lst, dim=0)
```

如果不同 worker 返回的单样本 DataProto tensor 宽度不同，仍然可能 shape mismatch。

所以“只让每个 worker 处理 1 条样本”不是严格根治，除非同时保证 worker 间输出 tensor 宽度一致，或者 patch concat/padding。

## 为什么 `filter_overlong_prompts=true` 不够

verl 的 prompt 过滤发生在 dataset 读入阶段，主要检查初始 chat prompt 是否超过 `data.max_prompt_length`。

但本错误发生在 agent loop 完成后：

```text
initial prompt
  -> model assistant text / tool call
  -> tool observation
  -> next prompt
  -> ...
  -> final prompt_ids
  -> _postprocess torch.cat
```

`filter_overlong_prompts=true` 只能减少初始超长题目，不能保证每条 trajectory 的最终 prompt 长度一致。

## 可选解决方案

### 方案 A：稳定优先，`VAL_BATCH_SIZE=1`

这是当前不改 verl 内部的最稳方案：

```bash
VAL_BATCH_SIZE=1 bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

优点：

- 避免同一个 `DataProto` 内出现不同宽度样本。
- 不需要 patch site-packages。
- 最符合“先跑出可复现 baseline”的目标。

缺点：

- 慢。
- SGLang 显存占用可以很高，但实际吞吐利用率不一定高。

### 方案 B：按 parquet 分片，多进程并行跑 `VAL_BATCH_SIZE=1`

这是“不改 verl 内部，同时保留并行”的工程方案。

做法：

- 把 `codecontests_test.parquet` 切成多个 shard。
- 每个 shard 单独启动一个 eval 进程。
- 每个进程使用 `VAL_BATCH_SIZE=1`。
- 用 `CUDA_VISIBLE_DEVICES` 把不同进程绑定到不同 GPU 或 GPU 组。
- 最后合并多个 `generations/0.jsonl` 和 metrics。

优点：

- 单个 verl agent loop 内仍保持安全。
- 可以用多进程拿回 wall-clock 并行。
- 不需要修改 verl 源码。

缺点：

- 每个进程都会加载模型和 SGLang server，资源占用高。
- 需要补 shard 脚本和 metrics merge。
- 单卡是否能承载 Qwen3-8B + hybrid actor + SGLang 要重新 smoke。2xA800 当前双卡跑是稳定的，单卡 80G 大概率可测，但不能直接假设。

### 方案 C：patch verl agent loop padding

这是理论上最干净、吞吐最高的方案。

需要 patch：

- `AgentLoopWorker._postprocess()`：对 `prompt_ids`、`response_ids`、`response_mask`、`attention_mask`、`input_ids`、`position_ids` 做 batch 内 padding 后再 concat。
- `DataProto.concat()` 或 `AsyncRolloutManager.generate_sequences()`：对多个 worker 返回的 DataProto 做跨 worker padding 后再 concat。

优点：

- 可以保留 `VAL_BATCH_SIZE>1` 和多 agent workers。
- 更符合高吞吐 eval。

缺点：

- 这是对 verl 内部语义的 patch，不是仓库 OJ 协议层问题。
- 需要非常小心 padding 方向、pad token、attention mask、position ids、response mask、logprobs 等字段一致性。
- patch 错了会产生静默指标污染，比直接失败更危险。

在没有专门单元测试覆盖 verl agent-loop tensor shape 的情况下，不建议直接把这个作为第一版 baseline 的默认路径。

## 当前建议

短期目标是先得到一份可信 baseline：

```bash
VAL_BATCH_SIZE=1 bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

如果 wall-clock 太慢，再做方案 B：

```text
生成 N 个 eval parquet shard
每个 shard 用 VAL_BATCH_SIZE=1 独立跑
最后合并 generations 和 metrics
```

如果后续确定要长期使用 verl experimental agent loop 做高吞吐在线工具评测，再单独做方案 C，并给 padding patch 加最小复现测试。

## 对脚本默认值的含义

高显存占用不等于高有效吞吐。

当前把：

```text
GPU_MEMORY_UTILIZATION=0.65
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=32
```

调高后，SGLang 每卡显存占用从约 `37GB` 提升到约 `54GB`，但只要 `VAL_BATCH_SIZE>1`，仍会撞上 agent-loop shape bug。

因此并行度的关键约束不是 GPU memory，而是 verl agent loop 对 variable-length multi-turn outputs 的 batch 后处理能力。

