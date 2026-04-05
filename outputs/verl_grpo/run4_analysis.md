# Run 4 分析 — `train_20260406_005402.log`

## 运行概况

- **配置**: `config_20260406_005402.yaml`
- **状态**: 首次通过 validation + 首次进入训练循环，但在 Step 0 完成后崩溃
- **总进展**: 这是 4 次 run 中走得最远的一次

## 时间线

| 时间 | 阶段 | 状态 |
|------|------|------|
| 00:54 | 启动 Ray + 加载模型 | ✅ |
| 00:55 | 初始 validation（tool_agent 多轮生成） | ✅ 首次通过 |
| 00:56 | Step 0: rollout 生成 | ✅ |
| 00:56:19 | compute_log_prob | ✅ 显存 17→25 GB |
| 00:56:52 | offload actor to CPU | ✅ 显存 25→2 GB |
| 00:56:53 | update_policy | ✅ 显存回到 17 GB |
| ~00:57 | _log_rollout_data → json.dumps | ❌ 崩溃 |

## 发现的两个问题

### 问题 1：`TypeError: Object of type int64 is not JSON serializable`（致命）

**错误位置**: `verl/trainer/ppo/ray_trainer.py:415` → `_dump_generations` → `json.dumps(entry)`

**根因**: verl 将 rollout 数据序列化为 JSONL 保存到 `rollout_data/` 目录，
但 entry 中包含 numpy 类型（`np.int64`, `np.float64`），Python 标准 `json` 模块无法序列化。

**修复**: 在 `site-packages/` 放置两个文件，让所有 Python 进程（含 Ray workers）自动 patch：

```
/root/miniconda3/lib/python3.12/site-packages/_json_numpy_patch.py   # patch 逻辑
/root/miniconda3/lib/python3.12/site-packages/_json_numpy_patch.pth  # 自动 import 触发
```

**为什么不能只在入口脚本 patch**: 错误发生在 `TaskRunner` Ray actor（pid=103081）里，
是 Ray spawn 的**独立 Python 进程**，不会继承主进程的 monkey-patch。
`.pth` 文件方案在 Python 进程启动时自动执行，对所有进程生效。

### 问题 2：初始 validation reward 仅 3.3%（预期 ~72%）

**现象**: validation 指标：
```
reward/mean@1:         0.033   (预期 ~0.72)
num_tool_calls/mean@1: 0.033   (几乎没调用工具)
num_turns/mean:        2.04    (几乎全是单轮)
```

**根因**: Qwen3 默认开启 **thinking mode**，模型先生成 `<think>...</think>` 块（轻松 300-500 tokens），
在 `response_length: 512` 的限制下，token 预算被思考过程耗尽，无法输出 tool call。

对比 baseline 评估（`qwen3-8b-no-think-multi-turn/`）使用了 `enable_thinking=False` + `max_new_tokens=1024`，
达到 MBPP 72.4% pass@1。

**修复（配置变更）**:
```yaml
# grpo_qwen3_8b.yaml 新增/修改

data:
  apply_chat_template_kwargs:
    enable_thinking: false       # 新增：关闭 Qwen3 思考模式

actor_rollout_ref:
  rollout:
    response_length: 1024        # 512 → 1024：与 baseline 对齐
```

传递路径：`data.apply_chat_template_kwargs` → `AgentLoopWorker` →
`tokenizer.apply_chat_template(enable_thinking=False)` → Qwen3 跳过 `<think>` 生成。

## 修改的文件清单

| 文件 | 变更 |
|------|------|
| `configs/verl/grpo_qwen3_8b.yaml` | +`enable_thinking: false`, `response_length: 512→1024` |
| `src/reward.py` | 统一两个分支的返回 key（Run 3 的 KeyError 修复） |
| `/root/miniconda3/.../site-packages/_json_numpy_patch.py` | 新建：numpy JSON 序列化 patch |
| `/root/miniconda3/.../site-packages/_json_numpy_patch.pth` | 新建：自动 import 触发 |

## 预期效果

- 初始 validation reward 应从 ~3% 提升到接近 baseline 的 ~50-70%
- Step 0 能完整跑通（含 rollout data 保存）
- 训练循环能持续进行
