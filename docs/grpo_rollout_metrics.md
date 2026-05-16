# GRPO Rollout 指标定义

更新日期：2026-05-13

本文定义 GRPO 训练时每个 `global_step` 需要统计的指标。目标不是只看 raw acc，而是同时判断训练更新是否健康、reward 是否被刷、GRPO 组内是否有有效相对信号，以及是否值得提高 `ppo_epochs`。

当前默认配置：

```text
1 global_step = 64 prompts * rollout.n=4 = 256 rollouts
1 prompt group = 同一道题的 4 条 rollout
ppo_mini_batch_size = 16 prompts = 64 rollouts
ppo_epochs = 1
1 global_step 内约 4 个 actor optimizer updates
```

指标分三层：

```text
optimization 指标：update 级别；同一 global_step 内多个 update 先取 mean
reward / correctness / behavior 指标：256 条 rollout 聚合
GRPO group 指标：64 个 prompt group 聚合
```

## Optimization 指标

这些指标来自训练更新过程，不直接从 256 条 rollout 统计。若日志保留 update 级明细，先对同一 `global_step` 内的 4 个 update 取 mean；若 verl 只落一次 step 汇总，则直接使用该 step 值。

| 指标 | 含义 | 判断 |
| --- | --- | --- |
| `actor/kl_loss` | 当前 policy 和 reference policy 的 KL penalty | 快速升高说明偏离 reference 过快 |
| `actor/ppo_kl` / `approx_kl` | 当前 update 前后 policy 的近似 KL | 衡量单次 update 是否过猛 |
| `actor/pg_clipfrac` | PPO/GRPO clip 被触发的比例 | 高说明大量 token 更新被裁剪 |
| `actor/entropy` | policy 输出分布熵 | 快速下降表示可能坍缩 |
| `actor/grad_norm` | 梯度范数 | 尖峰说明训练不稳定 |
| `actor/pg_loss` | policy gradient loss | 单独意义有限，主要看趋势 |

简化判断：

```text
KL / ppo_kl：模型变动有多大
clipfrac：更新有没有过猛
entropy：探索有没有塌缩
grad_norm：梯度是否稳定
```

`ppo_epochs=2` 的前提是：

```text
KL 稳定
clipfrac 大部分 < 0.3
entropy 没有明显塌
grad_norm 没有尖峰
```

## Reward 指标

这些指标按一个 `global_step` 的 256 条 rollout 聚合。每条 rollout 的当前 reward 为：

```text
R = R_outcome + R_bad_pattern
```

从每条 rollout 的 `score` / `reward` 和 `reward_breakdown` 读取：

| 指标 | 含义 | 聚合对象 |
| --- | --- | --- |
| `reward_mean` | 最终 reward 平均值 | 256 rollouts |
| `reward_std` | 最终 reward 标准差 | 256 rollouts |
| `outcome_reward_mean` | `R_outcome` 平均值 | 256 rollouts |
| `bad_pattern_mean` | `R_bad_pattern` 平均值 | 256 rollouts |
| `bad_pattern_nonzero_rate` | 有坏模式惩罚的轨迹比例 | 256 rollouts |

判断重点：

```text
outcome_reward_mean 上升：好
bad_pattern_mean 接近 0：坏行为减少
reward_std 太低：GRPO 组内差异不足
```

## GRPO Group 指标

这些指标按 64 个 prompt group 聚合。每个 group 有同一道题的 4 条 rollout：

```text
prompt_i: reward r1, r2, r3, r4
group_range = max(rewards) - min(rewards)
group_std = std(rewards)
```

| 指标 | 含义 | 聚合对象 |
| --- | --- | --- |
| `group_reward_range_mean` | 每组 `max_reward - min_reward` 的平均值 | 64 groups |
| `group_reward_std_mean` | 每组 reward std 的平均值 | 64 groups |
| `useful_group_rate_0.1` | 组内 reward range > 0.1 的组比例 | 64 groups |
| `useful_group_rate_0.2` | 组内 reward range > 0.2 的组比例 | 64 groups |
| `group_all_same_rate` | 4 条 reward 几乎一样的组比例 | 64 groups |
| `group_has_ac_rate` | 组内至少一条 `acc_final=1` 的比例 | 64 groups |
| `group_mixed_ac_rate` | 组内既有 AC 又有非 AC 的比例 | 64 groups |
| `group_all_bad_rate` | 组内 4 条都很差的比例，例如 `max_reward <= 0.1` | 64 groups |
| `group_all_good_rate` | 组内 4 条都很好的比例，例如 `min_reward >= 0.9` | 64 groups |

判断：

```text
useful_group_rate_0.2 高：当前 64x4 有足够组内信号
group_all_same_rate 高：组内没信号
group_mixed_ac_rate 高：AC 和非 AC 可直接比较，是强信号
```

如果：

```text
useful_group_rate_0.2 < 20%
group_all_same_rate 很高
```

再考虑从 `64x4` 切到 `32x8`。如果 `useful_group_rate_0.2 >= 30%`，通常先继续 `64x4`。

## Correctness 指标

这些指标按 256 条 rollout 聚合，来自 reward 顶层返回或 `reward_breakdown`。

| 指标 | 含义 | 聚合对象 |
| --- | --- | --- |
| `acc_final` | 最后一次 `submit_solution` 是否 accepted | 256 rollouts |
| `acc_any` | 轨迹中任意一次 `submit_solution` 是否 accepted | 256 rollouts |
| `solved_but_not_final_rate` | `acc_any=1` 且 `acc_final=0` 的比例 | 256 rollouts |
| `best_submit_pass_rate_mean` | 轨迹中最好一次 submit pass rate 的均值 | 256 rollouts |
| `last_submit_pass_rate_mean` | 最后一次 submit pass rate 的均值 | 256 rollouts |

解释：

```text
acc_final：最终是否成功，最重要
acc_any：是否曾经成功
best_submit_pass_rate：最好做到什么程度
last_submit_pass_rate：最终停在什么程度
```

典型诊断：

```text
acc_any > acc_final：模型会解，但 AC 后还继续乱动
best_submit_pass_rate 上升但 acc_any 不升：partial 解变好，但还没跨到 AC
acc_final 上升：真正目标变好
```

## Behavior 指标

这些指标按 256 条 rollout 聚合，优先从 `trajectory.tool_events`、`code_agent_trace`、`reward_breakdown` 读取。

| 指标 | 含义 | 聚合对象 |
| --- | --- | --- |
| `no_submit_rate` | 没有任何 submit 的轨迹比例 | 256 rollouts |
| `response_truncated_rate` | 输出撞 `max_response_length` / `response_length_exceeded` 的比例 | 256 rollouts |
| `public_acc_no_submit_rate` | public 过了但没 submit 的比例 | 256 rollouts |
| `accepted_then_continue_rate` | AC 后继续调用工具或继续长输出的比例 | 通常以 `acc_any=1` 的轨迹为分母 |
| `accepted_then_later_failed_rate` | AC 后又 submit 且失败的比例 | 通常以 `acc_any=1` 的轨迹为分母 |
| `avg_num_public_tests` | 平均 public test 调用次数 | 256 rollouts |
| `avg_num_submits` | 平均 submit 次数 | 256 rollouts |
| `avg_num_tool_calls` | 平均工具调用总次数 | 256 rollouts |
| `response_length_mean` | 平均 response token 数 | 256 rollouts |
| `response_length_p95` | response token 数 95 分位 | 256 rollouts |
| `tool_calls_p95` | 工具调用次数 95 分位 | 256 rollouts |
| `tool_call_parse_error_count` | parse failures 总数 | 256 rollouts |

当前最关键的是：

```text
no_submit_rate
response_truncated_rate
accepted_then_later_failed_rate
```

旧 run 里 `no_submit_rate` 和 `response_length/clip_ratio` 都偏高，因此新版 reward 已加入：

```text
response_truncated: -0.15
truncated_no_submit: -0.05
```

## Bad Pattern Top-K

这组指标按 256 条 rollout 聚合。每条 rollout 的 `reward_breakdown["bad_patterns"]` 会记录具体坏模式：

```json
{
  "no_submit": -0.25,
  "response_truncated": -0.15
}
```

每个 `global_step` 至少统计：

```text
bad_pattern_count
bad_pattern_rate = count / 256
bad_pattern_sum
bad_pattern_mean_contribution = sum / 256
```

重点关注：

| bad pattern | 含义 |
| --- | --- |
| `no_submit` | 没有 submit |
| `response_truncated` | 输出撞 response 上限 |
| `truncated_no_submit` | 撞 response 上限且没有 submit |
| `public_acc_no_submit` | public 过了但没提交 |
| `accepted_then_tool_call` | AC 后继续调用工具 |
| `accepted_then_later_failed_submit` | AC 后又提交失败 |
| `too_many_submits` | submit 过多 |
| `duplicate_submit_code` | 重复提交相同代码 |
| `malformed_tool_call` | 工具调用格式错误 |
| `unknown_tool` | 调用了 schema 外工具 |
| `tool_execution_error` | 工具 adapter 执行失败 |

这个表能直接说明坏行为主要来自哪里。比如 `bad_pattern_mean` 变差时，要用 top-k 判断是 `response_truncated` 增多，还是 submit probing / duplicate submit 增多。

## 建议报告格式

每个 `global_step` 输出一行 summary，并保留 bad pattern top-k 明细。

```text
global_step
rollouts
groups
optimizer_updates

kl_loss_mean
ppo_kl_mean
clipfrac_mean
entropy_mean
grad_norm_mean
pg_loss_mean

reward_mean
reward_std
outcome_reward_mean
bad_pattern_mean
bad_pattern_nonzero_rate

group_reward_range_mean
group_reward_std_mean
useful_group_rate_0.1
useful_group_rate_0.2
group_all_same_rate
group_has_ac_rate
group_mixed_ac_rate
group_all_bad_rate
group_all_good_rate

acc_final
acc_any
solved_but_not_final_rate
best_submit_pass_rate_mean
last_submit_pass_rate_mean

no_submit_rate
response_truncated_rate
public_acc_no_submit_rate
accepted_then_continue_rate
accepted_then_later_failed_rate
avg_num_public_tests
avg_num_submits
avg_num_tool_calls
response_length_mean
response_length_p95
tool_calls_p95
tool_call_parse_error_count

bad_pattern_topk
```

同时建议输出 first/last/rolling mean，减少单个 step 题目难度波动带来的误判。

## 决策口径

可以试 `ppo_epochs=2`，如果同时满足：

```text
KL 稳定，没有快速上升
clipfrac 大部分 < 0.3
entropy 没有明显塌缩
grad_norm 没有尖峰
useful_group_rate_0.2 >= 30%
bad_pattern_mean 没有变差
avg_num_submits 没有上升过快
response_truncated_rate / no_submit_rate 没有恶化
```

不建议试 `ppo_epochs=2`，如果出现：

```text
clipfrac 经常 > 0.4
KL 快速上升
entropy 快速下降
train reward 涨但 held-out eval 下降
bad_pattern_mean 变得更负
response_truncated_rate 或 no_submit_rate 上升
```
