# verl Parquet 数据现状分析

本文记录当前 `data/verl/` 下四个 Parquet 文件的状态，用作后续 Qwen3-8B baseline 训练和评测的输入快照。

生成日期：2026-04-25

## 文件清单

| 文件 | 用途 | 行数 | 文件大小 |
| --- | --- | ---: | ---: |
| `data/verl/codecontests_train.parquet` | GRPO 训练集 | 9,698 | 496 MB |
| `data/verl/codecontests_valid.parquet` | 训练过程 validation | 500 | 16 MB |
| `data/verl/codecontests_test.parquet` | CodeContests held-out test | 500 | 36 MB |
| `data/verl/livecodebench_test.parquet` | LiveCodeBench final/generalization eval | 611 | 5.1 GB |

当前四个文件都已经是 verl 可读的同一套 schema：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `data_source` | string | 数据集来源，当前为 `codecontests` 或 `livecodebench`。 |
| `prompt` | string | JSON 序列化后的 chat messages。verl 读取后再交给 tokenizer/chat template。 |
| `ability` | string | verl 常用任务能力标签，当前固定为 `code`。 |
| `reward_model` | string | JSON 序列化后的 reward 配置，当前为 rule reward，只保存 `task_id` 作为 ground-truth handle。 |
| `extra_info` | string | JSON 序列化后的非 prompt 元数据，主要给 tool layer 重建每道题对应的测试和限制。 |

注意：`prompt`、`reward_model`、`extra_info` 在 Parquet 里是 string，不是嵌套 struct。这是当前导出代码的明确设计，便于 verl 和 pandas/pyarrow 稳定读取。

## 数据来源与导出方式

内部统一 schema 是 `src/data/dataset.py` 里的 `CodeProblem`：

| `CodeProblem` 字段 | 导出到 Parquet 后的位置 |
| --- | --- |
| `task_id` | `reward_model.ground_truth.task_id` 和 `extra_info.task_id` |
| `dataset` | `data_source` 和 `extra_info.dataset` |
| `problem_statement` / `title` | 经 `src/prompts.py` 拼成 `prompt` 的 user content |
| `public_tests` | `extra_info.tools_kwargs.run_public_tests.create_kwargs.public_tests` |
| `private_tests` | `extra_info.tools_kwargs.submit_solution.create_kwargs.private_tests` |
| `time_limit_seconds` | 两个 tool 的 `create_kwargs.time_limit_seconds` |
| `max_submissions` | 两个 tool 的 `create_kwargs.max_submissions` |

当前 prompt 由 `build_agentic_messages(problem)` 构造，主要包含：

| 内容 | 来源 |
| --- | --- |
| system prompt | 我们添加，用于说明模型是 Python OJ agent，并提示可以调用工具。 |
| 题目标题和题面 | 原始 dataset 字段映射到 `CodeProblem` 后再拼入。 |
| public tests | 我们从结构化 public tests 中最多展示若干个，方便模型自测。 |
| 最终写完整 Python 3 stdin/stdout 程序的指令 | 我们添加，统一输出形式。 |

`extra_info` 不会直接显示给模型。verl 会把它传给 tool layer，tool 在每条 trajectory 中根据 `tools_kwargs` 创建对应实例。

## Tool 数据布局

当前每道题会给两个工具分别保存创建参数：

| tool | 保存的测试 | 用途 |
| --- | --- | --- |
| `run_public_tests` | 只保存 `public_tests` | 给模型调试，返回 public test feedback，不作为主 reward。 |
| `submit_solution` | 只保存 `private_tests` | full judge，产生主要 correctness reward。 |

这样做避免把 private tests 放进 public tool，也避免一个 tool 同时持有不必要的数据。工具对象本身在执行时按题目重建，不是所有题共用一个全局状态对象。

## CodeContests split 状态

原始导出后，CodeContests 的 valid/test 太小：

| split | 原始行数 |
| --- | ---: |
| train | 10,481 |
| valid | 95 |
| test | 122 |

当前已从 train 中固定抽样补充 valid/test，使评估更稳定：

| split | 当前行数 | 变化 |
| --- | ---: | --- |
| train | 9,698 | 从 train 移出 783 条 |
| valid | 500 | 原 valid 95 条 + train 抽样 405 条 |
| test | 500 | 原 test 122 条 + train 抽样 378 条 |

已检查当前三个 CodeContests split：

| 检查项 | 结果 |
| --- | --- |
| train 内部 `task_id` 重复 | 0 |
| valid 内部 `task_id` 重复 | 0 |
| test 内部 `task_id` 重复 | 0 |
| train 和 valid 交集 | 0 |
| train 和 test 交集 | 0 |
| valid 和 test 交集 | 0 |

这意味着当前 train/valid/test 可以作为互斥 split 使用。

## 文件大小分析

Parquet 是按列存储的。当前四个文件体积主要由 `extra_info` 决定，因为测试数据都放在 `extra_info.tools_kwargs` 里。

| 文件 | 总大小 | `prompt` 压缩后 | `extra_info` 压缩后 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `codecontests_train.parquet` | 496 MB | 9.0 MB | 487 MB | private tests 是主要占用。 |
| `codecontests_valid.parquet` | 16 MB | 0.5 MB | 15.8 MB | 补样后仍然较小。 |
| `codecontests_test.parquet` | 36 MB | 0.5 MB | 35.7 MB | test 中有若干较大的测试输出。 |
| `livecodebench_test.parquet` | 5.1 GB | 0.6 MB | 5.1 GB | 几乎全部来自 hidden/private tests。 |

LiveCodeBench 文件大不是因为 prompt 很长，而是因为少数题目的 hidden tests 输入/输出非常大。当前样例里最大的 LiveCodeBench 测试行是：

| 文件 | row | task_id | JSON 样例大小 | 说明 |
| --- | ---: | --- | ---: | --- |
| `livecodebench_largest_tests_row_351.json` | 351 | `livecodebench/abc369_g` | 190 MB | private tests 的 input/output 总量非常大。 |

CodeContests 里当前保存过的最大测试样例是：

| 文件 | row | task_id | JSON 样例大小 | 说明 |
| --- | ---: | --- | ---: | --- |
| `codecontests_train_largest_tests_row_3537.json` | 3537 | `codecontests/31c2dc5683130c38` | 28 MB | private outputs 较大，但远小于 LiveCodeBench 最大行。 |

详细 JSON 样例保存在 `data/verl/detail_json/`，用于人工检查单行内容。

## Prompt token 统计

token 统计使用 Qwen3-8B tokenizer，并通过 chat template 渲染后计算：

```text
add_generation_prompt=True
enable_thinking=False
```

| 文件 | 行数 | p50 | mean | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `codecontests_train.parquet` | 9,698 | 622 | 672.81 | 1,168 | 1,707 | 24,416 |
| `codecontests_valid.parquet` | 500 | 672 | 712.35 | 1,230 | 1,633 | 2,657 |
| `codecontests_test.parquet` | 500 | 653 | 714.65 | 1,189 | 1,829 | 6,416 |
| `livecodebench_test.parquet` | 611 | 711 | 769.69 | 1,322 | 1,807 | 3,041 |

超过常见 `max_prompt_length` 的数量：

| 文件 | `>512` | `>1024` | `>2048` | `>4096` | `>8192` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `codecontests_train.parquet` | 6,688 | 878 | 38 | 3 | 2 |
| `codecontests_valid.parquet` | 375 | 54 | 3 | 0 | 0 |
| `codecontests_test.parquet` | 363 | 54 | 4 | 1 | 0 |
| `livecodebench_test.parquet` | 488 | 101 | 5 | 0 | 0 |

结论：

| `max_prompt_length` | 数据覆盖判断 | 训练风险 |
| --- | --- | --- |
| 512 | 明显太小，CodeContests train 有 6,688 条会超。 | 显存安全，但大量 prompt 被截断或报错。 |
| 1024 | 覆盖大多数样本，但仍有不少长题超限。 | 比较稳，适合快速 baseline。 |
| 2048 | 覆盖绝大多数样本，是当前数据更合理的目标。 | 全参 GRPO 显存压力明显增加，可能需要降低 micro batch。 |
| 4096 | 几乎覆盖全部样本。 | 对 2xA800 全参训练不建议作为第一跑。 |

## 当前训练规模估算

当前配置 `configs/verl/grpo_qwen3_8b.yaml` 仍然保留：

| 配置项 | 当前值 |
| --- | ---: |
| `train_batch_size` | 128 |
| `rollout.n` | 4 |
| `ppo_mini_batch_size` | 32 |
| `ppo_micro_batch_size_per_gpu` | 2 |
| `response_length` | 1024 |
| `total_epochs` | 7 |

每个训练 step 会生成：

```text
128 题 × 4 samples = 512 条 rollout
```

基于当前 `train=9698`：

```text
9698 / 128 ≈ 75.8
```

历史 run 中 `dataset len=374`、`train_batch_size=128`、`total_epochs=7` 最终是 `14 steps`，说明实际 step 数更接近 floor 计算：

```text
floor(9698 / 128) = 75 steps / epoch
```

因此当前只跑 1 epoch，大约已经是：

```text
75 steps × 512 rollouts = 38,400 条 rollout
```

历史 Run 6 是：

```text
14 steps × 512 rollouts = 7,168 条 rollout
```

所以当前 1 epoch 的 rollout 数量约为历史完整 7 epoch 的 5.4 倍。

## Baseline 建议

为了先测出一个可靠 baseline，建议第一跑不要继续使用 `total_epochs=7`。

建议第一版：

| 配置项 | 建议值 | 原因 |
| --- | ---: | --- |
| `trainer.total_epochs` | 1 | 当前训练集已经明显变大，1 epoch 足够先看趋势。 |
| `trainer.save_freq` | 75 | 只在 1 epoch 结束保存，避免中间 checkpoint 干扰。 |
| `trainer.test_freq` | 75 | 只做初始和结束 validation，先降低时间成本。 |
| `data.max_prompt_length` | 1024 或 2048 | 1024 更稳，2048 更符合数据但更吃显存。 |
| `data.max_response_length` | 1024 | 当前 multi-turn 总 response budget，先不放大。 |

如果使用 `max_prompt_length=1024`，会牺牲一部分长题覆盖，但更适合快速 baseline。

如果使用 `max_prompt_length=2048`，建议同时考虑：

| 配置项 | 建议值 |
| --- | ---: |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | 1 |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` | 1 |
| `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | 1 |

原因是历史 2xA800 80GB 全参训练在较短 prompt 设置下，`update_policy` 后显存已经接近 80GB 上限。放大 prompt length 不只影响 rollout，也影响 log_prob 和 update_policy 阶段。

## 当前结论

四个 Parquet 文件已经可以作为第一版 baseline 输入：

| 文件 | 状态 |
| --- | --- |
| `codecontests_train.parquet` | 可用于训练，规模足够先跑 1 epoch。 |
| `codecontests_valid.parquet` | 可用于训练过程 validation，500 题比原始 95 题稳定。 |
| `codecontests_test.parquet` | 可用于 CodeContests held-out test，500 题比原始 122 题稳定。 |
| `livecodebench_test.parquet` | 可用于 final/generalization eval，但文件大、题目偏难，不建议作为训练集。 |

接下来优先目标不是继续改数据，而是用当前四个文件跑一个可复现 baseline，并记录：

| 指标 | 用途 |
| --- | --- |
| initial validation reward | 判断 base model + 当前 prompt/tool 是否正常。 |
| final validation reward | 判断 1 epoch GRPO 是否有学习趋势。 |
| CodeContests test reward | 判断 held-out 泛化。 |
| LiveCodeBench test reward | 判断更难分布上的泛化。 |
| 平均 tool calls / num turns | 判断模型是否学会有效使用 `run_public_tests` 和 `submit_solution`。 |
| length 截断率 | 判断是否需要放大 `max_prompt_length` 或 `response_length`。 |
