# verl baseline eval 调试记录 2026-04-25

本文记录一次目标为“尽量复用 verl agent loop 做 OJ-like baseline 评测”的调试过程。

## 目标

用户已准备好：

- `/root/autodl-tmp/code-agent/data/verl/codecontests_test.parquet`
- `/root/autodl-tmp/code-agent/data/verl/livecodebench_test.parquet`
- `/root/autodl-tmp/models/Qwen3-8B`

希望得到一个一键脚本：

- 一次只评测一个 parquet。
- 尽量走 verl `main_ppo` validation 路径。
- 不自己写 agent loop。
- 不再从网上下载数据或模型。
- 当前机器是一张 RTX 5090，显存约 32GB。

## 已做改动

新增脚本：

```bash
scripts/evaluate_baseline_with_verl.sh
```

这个脚本做了以下事情：

- 接受 `codecontests_test`、`livecodebench_test`、`codecontests_valid` 三个 alias，也接受直接传 `.parquet` 路径。
- 默认模型路径为 `/root/autodl-tmp/models/Qwen3-8B`。
- 只检查目标 eval parquet 和本地模型，不自动重新生成数据。
- 每次运行前用 `python3 -m src.env.tools` 重新生成 `configs/verl/tool_config.yaml`。
- 使用 `scripts/verl_main_wrapper.py` 启动 verl `main_ppo` validation。
- 设置 `trainer.val_only=True`、`trainer.val_before_train=True`，复用 verl validation 的 multi-turn rollout / tool / reward 链路。
- 默认输出到 `outputs/verl_baseline_eval/`。

README 已加入入口示例：

```bash
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
bash scripts/evaluate_baseline_with_verl.sh livecodebench_test
```

## 关键调试过程

### 1. 本地环境状态

确认过：

- `data/verl/` 下四个标准 parquet 都存在。
- `/root/autodl-tmp/models/Qwen3-8B` 存在。
- 当前环境安装了 `verl 0.7.1`。


### 2. Hydra 配置字段检查

用 `--cfg job` 验证过脚本里的主要 override 字段能够被当前 `verl 0.7.1` 接受。

关键 override 包括：

- `trainer.val_only=True`
- `data.val_files=...`
- `data.max_prompt_length=...`
- `data.max_response_length=...`
- `actor_rollout_ref.rollout.multi_turn.enable=true`
- `actor_rollout_ref.rollout.multi_turn.tool_config_path=configs/verl/tool_config.yaml`
- `actor_rollout_ref.rollout.agent.num_workers=1`
- `actor_rollout_ref.rollout.val_kwargs.n=1`

### 3. 第一次 smoke test 失败

命令形态：

```bash
VAL_MAX_SAMPLES=1 VAL_BATCH_SIZE=1 LOG_VAL_GENERATIONS=1 AGENT_WORKERS=1 \
MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=1024 \
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

失败原因：

```text
ValueError: train_batch_size (1) must be >= actor.ppo_mini_batch_size (32)
```

结论：

即使 `val_only=True`，verl 仍会初始化 train dataloader 并校验 `train_batch_size >= ppo_mini_batch_size`。

修复：

- 脚本默认 `TRAIN_BATCH_SIZE=32`。
- 仍然用 `codecontests_valid.parquet` 作为 dummy train file。
- 真正评测只由 `data.val_files` 决定。

### 4. 第二次 smoke test 失败

修复 `TRAIN_BATCH_SIZE=32` 后，verl 进入模型加载阶段，但 actor FSDP 初始化 OOM：

```text
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 2.32 GiB.
GPU 0 has a total capacity of 31.36 GiB ...
```

日志里能看到默认 FSDP actor 用了：

```yaml
actor_rollout_ref.actor.fsdp_config.model_dtype: fp32
```

结论：

单卡 5090 上用 fp32 初始化 Qwen3-8B actor 不可行。

修复：

- 脚本默认 `FSDP_MODEL_DTYPE=bf16`。
- 覆盖：

```bash
actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
```

### 5. 第三次 smoke test 失败

bf16 后 actor 初始化能过，失败点转移到 SGLang hybrid rollout server：

```text
[torch_memory_saver.cpp] CUresult error: 2 (out of memory)
Rank 0 scheduler is dead.
EOFError
```

判断：

- 这不是工具层或数据层问题。
- 这是 verl hybrid 模式下 actor + SGLang rollout server 在同一张 32GB 5090 上并存导致的显存不足。

进一步尝试过：

- `GPU_MEMORY_UTILIZATION=0.35`
- `MAX_MODEL_LEN=3072`
- `MAX_NUM_BATCHED_TOKENS=4096`
- `MAX_NUM_SEQS=16`
- `ENFORCE_EAGER=true`

仍然在 SGLang scheduler 启动时 OOM。


中断后检查：

- 没有残留 Ray/SGLang/verl 进程。
- GPU 显存已释放。

## 当前判断

当前代码和数据链路已经接到 verl validation，但单张 5090 跑 `Qwen3-8B + verl hybrid SGLang multi-turn agent loop` 风险很高。

主要瓶颈不是 parquet、tool schema 或 reward，而是：

- actor FSDP 模型需要占显存。
- SGLang rollout server 也需要占显存。
- verl hybrid validation 会同时初始化这些组件。
- 单张 32GB 5090 没有足够余量稳定容纳 Qwen3-8B + SGLang rollout。

## 推荐卡数

如果“不降低太多配置”，建议：

- 最低建议：`2 x 48GB` 或 `2 x 80GB`。
- 更贴近仓库现有配置和历史假设：`2 x A800 80GB`。
- 单张 5090 只适合继续做极小 smoke 或改用更小模型，不适合稳定跑 Qwen3-8B verl agentic baseline。

## Prompt / Response Length 建议

`data.max_prompt_length=2048` 是当前更合理的 baseline 档位。

根据 `docs/verl_parquet_dataset_analysis.md`：

- `codecontests_test.parquet` 只有 `4/500` 条超过 2048。
- `livecodebench_test.parquet` 只有 `5/611` 条超过 2048。
- 如果用 1024，则 CodeContests test 有 `54/500` 超过，LiveCodeBench test 有 `101/611` 超过，baseline 会明显受截断影响。

`data.max_response_length=1024` 对 OJ-like agentic baseline 可能偏小。这里的 `response_length` 是整条 multi-turn trajectory 的 assistant token 总预算，不是单轮 assistant 输出上限。一条完整 Python 程序本身就可能几百到上千 token；如果模型先调用 `run_public_tests`，再根据反馈修复，最后 `submit_solution`，1024 很容易截断后续修复或提交。

建议区分：

```text
链路 smoke:     max_response_length=1024
正式 baseline: max_response_length=2048
更宽松探索:   max_response_length=3072
```

如果采用正式 baseline：

```text
data.max_prompt_length=2048
data.max_response_length=2048
actor_rollout_ref.rollout.max_model_len=4096
```

注意：`max_model_len` 应显式设为 `max_prompt_length + max_response_length`，避免 SGLang 按 Qwen3 的大上下文上限预留过多 KV/cache。

推荐 baseline 配置（2 x A800 80GB）：

```text
NUM_GPUS=2
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=2048
MAX_MODEL_LEN=4096
VAL_MAX_SAMPLES=-1
AGENT_WORKERS=2
GPU_MEMORY_UTILIZATION=0.40 到 0.45
FSDP_MODEL_DTYPE=bf16
MAX_NUM_BATCHED_TOKENS=4096 或 8192
MAX_NUM_SEQS=8 到 16
```

如果只有 `2 x 48GB`，建议更保守：

```text
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=2048
MAX_MODEL_LEN=4096
AGENT_WORKERS=1 或 2
GPU_MEMORY_UTILIZATION=0.35 到 0.40
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=4 到 8
```

如果是 `2 x 80GB`，优先跑：

```bash
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
bash scripts/evaluate_baseline_with_verl.sh livecodebench_test
```

必要时显式指定：

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=2048 MAX_MODEL_LEN=4096 \
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

## 2026-04-25 多卡继续调试结果

已在 `2 x A800 80GB` 机器上继续调试，并完成以下修复：

- `scripts/evaluate_baseline_with_verl.sh` 不再固定单卡，默认自动探测所有可见 GPU，并将 `NUM_GPUS` 接入 `trainer.n_gpus_per_node`。
- 新增 `ROLLOUT_TP` 参数，默认 `1`，并检查它必须整除 `NUM_GPUS`。
- 正式 baseline 默认改为 `MAX_PROMPT_LENGTH=2048`、`MAX_RESPONSE_LENGTH=2048`、`MAX_MODEL_LEN=4096`。
- 同时覆盖 `actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH`，避免只改 `data.max_response_length` 而 rollout 仍继承 `1024`。
- 新增 `src/verl_dataset_adapter.py` 的 `OJLikeRLHFDataset`，在 verl 读取 parquet 时把当前 v1 schema 中的 JSON string 字段 `prompt`、`reward_model`、`extra_info` 解码成 verl 0.7.1 需要的嵌套对象。
- eval 脚本通过 `data.custom_cls.path=src/verl_dataset_adapter.py` 和 `data.custom_cls.name=OJLikeRLHFDataset` 使用该 adapter，不改变现有 parquet 协议。

已跑通 smoke：

```bash
VAL_MAX_SAMPLES=1 VAL_BATCH_SIZE=1 LOG_VAL_GENERATIONS=1 AGENT_WORKERS=1 \
bash scripts/evaluate_baseline_with_verl.sh codecontests_test
```

结果：

- `trainer.n_gpus_per_node=2` 生效。
- `Using dataset class: OJLikeRLHFDataset` 生效。
- SGLang hybrid server 在两张 A800 上成功启动。
- validation 完整结束并写出 `outputs/verl_baseline_eval/codecontests_test_Qwen3-8B_mp2048_mr2048_20260425_230544/generations/0.jsonl`。
- smoke 指标为 `score=0.0`、`acc=0.0`、`num_tool_calls=0.0`，这是模型输出未触发 tool call 的样本级结果，不是脚本失败。

## 下个上下文提示词

可以把下面这段直接给下一个 Codex 上下文：

```text
我们在 /root/autodl-tmp/code-agent 做 OJ-like code agent baseline。请先读 AGENTS.md、README.md、docs/env_protocol.md、docs/verl_parquet_dataset_analysis.md、docs/verl_baseline_eval_debug_2026-04-25.md。

目标：继续把 baseline 评测脚本做成可在多卡机器上一键跑通，评测集是 data/verl/codecontests_test.parquet 和 data/verl/livecodebench_test.parquet，一次只跑一个。必须尽量复用 verl main_ppo validation 的 multi-turn agent loop、tool layer 和 src/reward.py，不要自己写 agent loop，不要重新引入 execute_code/test_list/函数补全协议，不要下载数据或模型。

当前已新增 scripts/evaluate_baseline_with_verl.sh，并更新 README。单 5090 调试结论：Hydra 配置和数据/tool schema 能过；train_batch_size 需 >= ppo_mini_batch_size，所以脚本默认 TRAIN_BATCH_SIZE=32；actor FSDP 默认 fp32 会 OOM，已覆盖 actor/ref fsdp_config.model_dtype=bf16；之后 SGLang hybrid server 在单 32GB 5090 上仍 OOM，即使 GPU_MEMORY_UTILIZATION=0.35、MAX_MODEL_LEN=3072、MAX_NUM_SEQS=16、ENFORCE_EAGER=true。判断是 Qwen3-8B + verl hybrid SGLang + actor 同卡并存显存不足。2 x A800 80GB 上已跑通 VAL_MAX_SAMPLES=1 smoke。

长度建议：max_prompt_length=2048 是合理 baseline 档位；response_length=1024 只适合 smoke，正式 OJ-like agentic baseline 更建议 max_response_length=2048，并显式设置 rollout.max_model_len=4096，避免 SGLang 按 Qwen3 大上下文预留过多 cache。

当前状态：scripts/evaluate_baseline_with_verl.sh 已支持多卡自动探测和 NUM_GPUS 覆盖，已加入 OJLikeRLHFDataset 适配 JSON-string parquet schema，并已跑通 VAL_MAX_SAMPLES=1 smoke。下一步可以直接跑 codecontests_test 全量，最后跑 livecodebench_test。模型路径默认 /root/autodl-tmp/models/Qwen3-8B。输出继续放 outputs/verl_baseline_eval/。如果需要改 README 或 docs，请同步更新。
```
