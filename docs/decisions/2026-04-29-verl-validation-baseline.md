# 决策：baseline 评测复用 verl validation 路径

日期：2026-04-29

## 决策

当前 baseline 评测入口使用 `scripts/evaluate_baseline_with_verl.sh`，复用 verl `main_ppo` validation 路径，尽量和训练 rollout、agent loop、tool layer 和 reward 保持一致。

## 背景

本项目的第一阶段目标是跑出可复现 baseline，并验证 OJ-like v1 数据、tool、reward 和 verl agent loop 的主链路。为了提高评测效率，并尽可能和训练环境保持一致，baseline eval 应尽量复用训练时的 multi-turn agent loop，而不是另写本地 evaluate harness。

2026-04-25 到 2026-04-28 的调试已经确认：

- 单 5090 不适合稳定承载 Qwen3-8B + verl hybrid SGLang + actor。
- 2*A800 80GB 可以跑 2GPU 统一调度 smoke。
- Parquet `prompt` 是 JSON string，需要 `OJLikeRLHFDataset` 在读取和过滤阶段 decode。
- validation 需要 partial dump，避免长评测中断后丢失已完成 batch。
- patch 应安装在 Ray CPU TaskRunner actor 内，不能用 `sitecustomize.py` 过早 import verl/torch。

## 被拒绝的方案

- **重新写独立 agent loop eval**：短期可控，但会和训练 rollout/tool/reward 语义分叉。
- **使用 `/Users/yang/code/verl/verl/trainer/main_eval.py` 作为在线工具评测入口**：它只适合对已有 responses 做离线 reward 打分，不执行当前 OJ-like tool loop。
- **两张卡各启动一个 verl 进程做数据分片 fallback**：可以绕开部分调度问题，但不能验证单个 verl 进程统一调度多卡的训练/评测路径。
- **使用 `sitecustomize.py` 或全局启动钩子安装 patch**：会让 Ray GPU worker 在 CUDA visibility 最终确定前 import verl/torch，可能触发 NCCL `Duplicate GPU detected`。

## 后果

- baseline eval 默认走 `scripts/evaluate_baseline_with_verl.sh`。
- 评测相关修复优先保持在 verl validation / TaskRunner / dataset adapter / runtime patch 这一条链路上。
- `src/verl_dataset_adapter.py` 负责 JSON-string Parquet schema 到 verl dataset 的适配。
- `src/verl_runtime_patch.py` 负责当前必要的 validation runtime patch。
- 当前 validation 结果应优先记录到 `docs/project_status.md`，详细调试过程放到 `docs/debug/`。

## 相关文档

- `docs/project_status.md`
- `docs/debug/verl_baseline_eval_debug_2026-04-28.md`
- `docs/references/verl_ray_agent_loop_reading_guide.md`
- `scripts/evaluate_baseline_with_verl.sh`
- `scripts/verl_main_wrapper.py`
- `src/verl_dataset_adapter.py`
- `src/verl_runtime_patch.py`
