"""Runtime patches for using verl validation as OJ-like online evaluation.

The patches in this module are intentionally opt-in. They are installed by
``scripts.verl_main_wrapper.CodeAgentTaskRunner`` inside the CPU TaskRunner Ray
actor, so GPU workers do not import verl or torch before Ray finalizes their
per-worker CUDA visibility.

补丁安装点：CodeAgentTaskRunner.run() → apply_patches()
  调用链：bash evaluate_baseline_with_verl.sh → verl_main_wrapper.py → run_ppo()
    → Ray 创建 CodeAgentTaskRunner actor → .run(config) → apply_patches()

本文件包含两个 patch：
  1. _install_numpy_json_patch:    让 stdlib json 能序列化 numpy 类型
  2. _install_validation_partial_dump_patch: 替换 RayPPOTrainer._validate，
     在每个 validation batch 完成后增量写 partial_0.jsonl
"""

from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from typing import Any

import numpy as np

from src.trajectory_parser import to_messages


_PATCHED = False


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 1: numpy → JSON 序列化
# ═══════════════════════════════════════════════════════════════════════════════

def _install_numpy_json_patch() -> None:
    """Make stdlib json handle numpy scalar/array values in verl dumps.

    问题：verl 内部用 json.dumps 写 rollout/validation 记录时，部分指标来自
    numpy（如 np.int64, np.float64, np.ndarray），标准库 json 不认识这些类型，
    会抛 TypeError。

    修复：替换 json.JSONEncoder.default 方法，遇到 numpy 类型时自动转换为
    Python 原生类型。通过 _code_agent_numpy_safe 哨兵防止重复安装。

    注意：verl_main_wrapper.py 在 driver 进程也做了一次同样的 patch，
    因为 driver 在 import 阶段也需要写 JSON。那里的 patch 是 driver 侧的，
    这里的 patch 是 Ray actor 内的——两边独立，互不依赖。
    """

    if getattr(json.JSONEncoder, "_code_agent_numpy_safe", False):
        return

    original_default = json.JSONEncoder.default

    def numpy_safe_default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return original_default(self, obj)

    json.JSONEncoder.default = numpy_safe_default
    json.JSONEncoder._code_agent_numpy_safe = True


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 2: validation 增量 dump
# ═══════════════════════════════════════════════════════════════════════════════

def _append_partial_generations(
    trainer: Any,
    *,
    inputs: list[str],
    outputs: list[str],
    raw_prompts: list[Any],
    gts: list[Any],
    scores: list[float],
    reward_extra_infos: dict[str, list[Any]],
    dump_path: str | None,
    batch_index: int,
) -> None:
    """Append one completed validation batch before the full pass finishes.

    核心目的：防止长评测被中断后完全丢失已生成序列。
      - 无增量 dump 时：全部 batch 跑完才写一次 0.jsonl，中途崩溃结果全丢
      - 有增量 dump 后：每个 batch 跑完立刻追加写 partial_0.jsonl，崩溃只丢当前 batch

    写入策略：
      - 文件命名：partial_{global_steps}.jsonl（当前 global_steps=0）
      - 每 batch 的 n 条样本拆成 n 行独立 jsonl，方便增量追加和逐行读取
      - flush + fsync 确保写到磁盘，进程被 SIGKILL 也不会丢已完成 batch
    """

    if not dump_path:
        return

    os.makedirs(dump_path, exist_ok=True)
    filename = os.path.join(dump_path, f"partial_{trainer.global_steps}.jsonl")
    n = len(inputs)
    base_data = {
        "input": inputs,
        "output": outputs,
        "messages": [
            to_messages(
                output,
                initial_messages=raw_prompts[i] if i < len(raw_prompts) else None,
            )
            for i, output in enumerate(outputs)
        ],
        "gts": gts,
        "score": scores,
        "step": [trainer.global_steps] * n,
        "batch_index": [batch_index] * n,
    }

    for key, values in reward_extra_infos.items():
        if len(values) == n:
            base_data[key] = values

    with open(filename, "a", encoding="utf-8") as f:
        for i in range(n):
            entry = {key: values[i] for key, values in base_data.items()}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _dump_generations_with_structure(
    trainer: Any,
    *,
    inputs: list[str],
    outputs: list[str],
    raw_prompts: list[Any],
    gts: list[Any],
    scores: list[float],
    reward_extra_infos_dict: dict[str, list[Any]],
    dump_path: str,
) -> None:
    """Write final validation generations with parsed tool-event structure."""
    os.makedirs(dump_path, exist_ok=True)
    filename = os.path.join(dump_path, f"{trainer.global_steps}.jsonl")
    n = len(inputs)
    base_data = {
        "input": inputs,
        "output": outputs,
        "messages": [
            to_messages(
                output,
                initial_messages=raw_prompts[i] if i < len(raw_prompts) else None,
            )
            for i, output in enumerate(outputs)
        ],
        "gts": gts,
        "score": scores,
        "step": [trainer.global_steps] * n,
    }

    for key, values in reward_extra_infos_dict.items():
        if len(values) == n:
            base_data[key] = values

    with open(filename, "w", encoding="utf-8") as f:
        for i in range(n):
            entry = {key: values[i] for key, values in base_data.items()}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Dumped generations with messages to {filename}")


def _install_validation_partial_dump_patch() -> None:
    """Patch RayPPOTrainer._validate to write per-batch partial generations.

    为什么必须整体替换 _validate 方法：
      _validate 内部的 batch 循环持有局部变量（sample_inputs, sample_outputs,
      sample_scores 等），这些变量在循环体内逐步累积。如果不在循环体内直接插入
      _append_partial_generations 调用，从外部无法拿到"当前 batch 刚跑完、
      还没进入下一个 batch"的中间状态。没有办法只 patch 一两个小函数就达到
      增量 dump 的效果。

    替换后的 _validate 流程（与原版的对比）：
      原版：                         新版：
      for batch in dataloader:       for batch in dataloader:
        agent loop                     agent loop
        extract_reward                 extract_reward
        accumulate to lists            accumulate to lists
      end                            → _append_partial_generations() ← 新增
      _dump_generations()            end
                                     _dump_generations()  ← 保留

    每个 batch 的处理步骤（在循环体内）：
      1. DataProto.from_single_dict(test_data)     ← 构造 DataProto
      2. test_batch.repeat(n=1)                    ← 不做重复采样
      3. 提取 ground_truth
      4. _get_gen_batch + pad_dataproto_to_divisor ← 对齐 agent worker 数量
      5. async_rollout_manager.generate_sequences  ← 真正的 multi-turn agent loop
      6. unpad_dataproto                           ← 去掉对齐填充
      7. tokenizer.decode → output_texts + input_texts
      8. extract_reward → scores
      9. _append_partial_generations               ← 新增：增量写出

    哨兵：_code_agent_partial_dump = True，防止重复安装
    """

    from verl import DataProto
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.trainer.ppo.reward import extract_reward

    if getattr(RayPPOTrainer, "_code_agent_partial_dump", False):
        return

    def validate_with_partial_dump(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_raw_prompts = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        val_data_dir = self.config.trainer.get("validation_data_dir", None)

        # ── batch 循环 ──────────────────────────────────────────────
        for batch_index, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)

            # 如果数据中没有 uid，生成一个随机 UUID，方便后续追踪
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                    dtype=object,
                )

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            # 提取 ground_truth，用于最终 0.jsonl 写出
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                for item in test_batch
            ]
            sample_gts.extend(ground_truths)
            raw_prompts = list(test_batch.non_tensor_batch.get("raw_prompt", []))
            sample_raw_prompts.extend(raw_prompts)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad 到 agent worker 数量的整数倍，确保 batch 能被均匀分发
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(
                test_gen_batch_padded
            )

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                self.checkpoint_manager.update_weights(self.global_steps)

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # ── 增量 dump：解析 batch 结果，立即追加到 partial_0.jsonl ──
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            input_ids = test_batch.batch["prompts"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
            ]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            reward_tensor, reward_extra_info = extract_reward(test_batch)
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            batch_reward_extra_infos = {"reward": list(scores)}
            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                values_list = values.tolist() if isinstance(values, np.ndarray) else values
                values_list = values_list if isinstance(values_list, list) else [values_list]
                batch_reward_extra_infos[key] = values_list
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                reward_extra_infos_dict[key].extend(values_list)

            # ← 在这里增量写出，每个 batch 完成后立刻落盘
            _append_partial_generations(
                self,
                inputs=input_texts,
                outputs=output_texts,
                raw_prompts=raw_prompts,
                gts=ground_truths,
                scores=scores,
                reward_extra_infos=batch_reward_extra_infos,
                dump_path=val_data_dir,
                batch_index=batch_index,
            )
            # ── 增量 dump 结束 ──

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )

        # ── 所有 batch 完成后 ──

        self._maybe_log_val_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
        )

        # 写出完整的 0.jsonl，并附带标准 messages 方便后续复用。
        if val_data_dir:
            _dump_generations_with_structure(
                self,
                inputs=sample_inputs,
                outputs=sample_outputs,
                raw_prompts=sample_raw_prompts,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, values in reward_extra_infos_dict.items():
            assert len(values) == 0 or len(values) == len(sample_scores), (
                f"{key_info}: {len(values)=}, {len(sample_scores)=}"
            )

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }

        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(
            data_sources,
            sample_uids,
            reward_extra_infos_dict,
            sample_turns,
        )

    # 替换原方法 + 打哨兵防重复
    RayPPOTrainer._validate = validate_with_partial_dump
    RayPPOTrainer._code_agent_partial_dump = True


def _install_tool_agent_terminal_patch() -> None:
    """Terminate verl ToolAgentLoop after OJ tools mark a trajectory terminal."""
    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

    if getattr(ToolAgentLoop, "_code_agent_terminal_stop", False):
        return

    original_call_tool = ToolAgentLoop._call_tool
    original_handle_processing_tools_state = ToolAgentLoop._handle_processing_tools_state

    async def call_tool_with_terminal_flag(self, tool_call, tools_kwargs, agent_data):
        tool_response, tool_reward, result = await original_call_tool(
            self,
            tool_call,
            tools_kwargs,
            agent_data,
        )
        if isinstance(result, dict) and result.get("terminal"):
            setattr(agent_data, "code_agent_terminal", True)
            setattr(
                agent_data,
                "code_agent_terminal_reason",
                result.get("terminal_reason") or "tool_terminal",
            )
        return tool_response, tool_reward, result

    async def handle_processing_tools_state_with_terminal(self, agent_data):
        state = await original_handle_processing_tools_state(self, agent_data)
        if getattr(agent_data, "code_agent_terminal", False):
            return AgentState.TERMINATED
        return state

    ToolAgentLoop._call_tool = call_tool_with_terminal_flag
    ToolAgentLoop._handle_processing_tools_state = (
        handle_processing_tools_state_with_terminal
    )
    ToolAgentLoop._code_agent_terminal_stop = True


# ═══════════════════════════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════════════════════════

def apply_patches() -> None:
    """安装所有 code-agent 运行时补丁。

    调用方：CodeAgentTaskRunner.run()，在 Ray CPU actor 内调用。
    必须在 GPU worker 创建之前安装，但也不能太早（sitecustomize.py 太早）。
    当前安装点是"TaskRunner Ray CPU actor 启动后、真正 init_workers 前"。

    补丁列表：
      1. _install_numpy_json_patch:           json.dumps 兼容 numpy 类型
      2. _install_validation_partial_dump_patch: validation 增量 dump

    幂等：重复调用不会重复安装（各 patch 内部有哨兵检查）。
    """
    global _PATCHED
    if _PATCHED:
        return

    _install_numpy_json_patch()
    _install_tool_agent_terminal_patch()
    _install_validation_partial_dump_patch()
    _PATCHED = True
    print("[code-agent] verl runtime patches enabled")
