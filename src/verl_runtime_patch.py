"""Runtime patches for using verl validation as OJ-like online evaluation.

The patches in this module are intentionally opt-in. They are installed by
``scripts.verl_main_wrapper.CodeAgentTaskRunner`` inside the CPU TaskRunner Ray
actor, so GPU workers do not import verl or torch before Ray finalizes their
per-worker CUDA visibility.

补丁安装点：CodeAgentTaskRunner.run() → apply_patches()
  调用链：bash evaluate_baseline_with_verl.sh → verl_main_wrapper.py → run_ppo()
    → Ray 创建 CodeAgentTaskRunner actor → .run(config) → apply_patches()

本文件包含两个 TaskRunner 侧 patch：
  1. _install_numpy_json_patch:    让 stdlib json 能序列化 numpy 类型
  2. _install_validation_partial_dump_patch: 替换 RayPPOTrainer._validate，
     在每个 validation batch 完成后增量写 partial_0.jsonl

"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from src.trajectory_parser import to_messages


_PATCHED = False
_CODE_AGENT_DUMP_ONLY_KEYS = (
    "code_agent_trace",
    "code_agent_tool_events",
    "code_agent_terminal_reason",
    "code_agent_parse_failures",
    "code_agent_tool_tail_chars",
)
_CANONICAL_REWARD_EXTRA_KEYS = frozenset({"reward"})


def _now_for_log() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _value_at(values: Any, index: int, default: Any = None) -> Any:
    try:
        if values is None:
            return default
        return values[index]
    except Exception:
        return default


def _dataproto_len(data: Any) -> int:
    batch = getattr(data, "batch", None)
    if batch is not None:
        try:
            return len(batch)
        except Exception:
            pass
    non_tensor_batch = getattr(data, "non_tensor_batch", None)
    if isinstance(non_tensor_batch, dict):
        for values in non_tensor_batch.values():
            try:
                return len(values)
            except Exception:
                continue
    return 0


def _values_to_list(values: Any) -> list[Any]:
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, list):
        return values
    return [values]


def _append_reward_extra_infos(
    reward_extra_infos_dict: dict[str, list[Any]],
    batch_reward_extra_infos: dict[str, list[Any]],
    reward_extra_info: dict[str, Any],
) -> None:
    """Merge custom reward diagnostics without overriding canonical fields."""
    for key, values in reward_extra_info.items():
        if key in _CANONICAL_REWARD_EXTRA_KEYS:
            continue
        values_list = _values_to_list(values)
        batch_reward_extra_infos[key] = values_list
        reward_extra_infos_dict.setdefault(key, []).extend(values_list)


def _float_at(values: Any, index: int, default: float = 0.0) -> float:
    value = _value_at(values, index, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _messages_for_record(output: str, raw_prompt: Any, reward_extra_infos: dict[str, list[Any]], index: int) -> list:
    return to_messages(output, initial_messages=raw_prompt)


def _trace_for_record(reward_extra_infos: dict[str, list[Any]], index: int) -> dict[str, Any]:
    trace = _value_at(reward_extra_infos.get("code_agent_trace"), index, {})
    if not isinstance(trace, dict):
        trace = {}
    return trace


def _task_id_from_ground_truth(ground_truth: Any) -> str:
    if isinstance(ground_truth, dict):
        task_id = ground_truth.get("task_id")
        if task_id:
            return str(task_id)
    return str(ground_truth or "")


def _data_source_from_task_id(task_id: str) -> str:
    return task_id.split("/", 1)[0] if "/" in task_id else "unknown"


def _stop_reason_for_record(
    trace: dict[str, Any],
    reward_extra_infos: dict[str, list[Any]],
    index: int,
) -> str:
    reason = trace.get("terminal_reason") or _value_at(
        reward_extra_infos.get("code_agent_terminal_reason"),
        index,
    )
    # Older traces used accepted as a terminal reason.  In the normalized
    # schema accepted is a judge verdict; the episode ends naturally after the
    # model stops calling tools.
    if reason == "accepted":
        return "no_tool_call"
    return str(reason or "response_length_exceeded")


def _generation_record(
    *,
    trainer: Any,
    input_text: str,
    output_text: str,
    messages: list[dict[str, Any]],
    ground_truth: Any,
    score: float,
    reward_extra_infos: dict[str, list[Any]],
    index: int,
    assistant_token_count: int,
    output_token_count: int,
    response_length: int,
    rollout_index: int,
) -> dict[str, Any]:
    trace = _trace_for_record(reward_extra_infos, index)
    task_id = _task_id_from_ground_truth(ground_truth)
    final_submit_verdict = str(trace.get("final_submit_verdict") or "no_submission")
    final_submit_passed = trace.get("final_submit_passed")
    final_submit_total = trace.get("final_submit_total")
    acc = _float_at(reward_extra_infos.get("acc"), index, 1.0 if final_submit_verdict == "accepted" else 0.0)
    return {
        "sample": {
            "task_id": task_id,
            "data_source": _data_source_from_task_id(task_id),
        },
        "trajectory": {
            "input": input_text,
            "output": output_text,
            "messages": messages,
            "tool_events": _value_at(reward_extra_infos.get("code_agent_tool_events"), index, []),
        },
        "episode": {
            "stop_reason": _stop_reason_for_record(trace, reward_extra_infos, index),
            "assistant_token_count": int(assistant_token_count),
            "output_token_count": int(output_token_count),
            "response_length": int(response_length),
            "parse_failures": int(
                trace.get(
                    "parse_failures",
                    _value_at(reward_extra_infos.get("code_agent_parse_failures"), index, 0),
                )
                or 0
            ),
        },
        "judge": {
            "final_submit_verdict": final_submit_verdict,
            "final_submit_reward": float(trace.get("final_submit_reward") or 0.0),
            "final_submit_passed": int(final_submit_passed) if final_submit_passed is not None else None,
            "final_submit_total": int(final_submit_total) if final_submit_total is not None else None,
        },
        "behavior": {
            "num_tool_calls": int(trace.get("num_tool_calls", 0) or 0),
            "public_test_call_count": int(trace.get("public_test_call_count", 0) or 0),
            "submission_count": int(trace.get("submission_count", 0) or 0),
            "has_tool_call_after_submit_accepted": bool(
                trace.get("has_tool_call_after_submit_accepted", False)
            ),
            "assistant_chars_after_submit_accepted": int(
                trace.get("assistant_chars_after_submit_accepted", 0) or 0
            ),
            "assistant_tokens_after_submit_accepted": int(
                trace.get("assistant_tokens_after_submit_accepted", 0) or 0
            ),
        },
        "metrics": {
            "acc": acc,
            "acc_final": _float_at(reward_extra_infos.get("acc_final"), index, acc),
            "acc_any": _float_at(reward_extra_infos.get("acc_any"), index, 0.0),
            "best_submit_pass_rate": _float_at(
                reward_extra_infos.get("best_submit_pass_rate"),
                index,
                0.0,
            ),
            "last_submit_pass_rate": _float_at(
                reward_extra_infos.get("last_submit_pass_rate"),
                index,
                0.0,
            ),
            "reward": float(score),
            "outcome_reward": _float_at(reward_extra_infos.get("outcome_reward"), index, 0.0),
            "outcome_submit_policy": str(
                _value_at(reward_extra_infos.get("outcome_submit_policy"), index, "last_submit")
            ),
            "bad_pattern": _float_at(reward_extra_infos.get("bad_pattern"), index, 0.0),
            "reward_breakdown": _value_at(reward_extra_infos.get("reward_breakdown"), index, ""),
        },
        "verl": {
            "step": int(trainer.global_steps),
            "rollout_index": int(rollout_index),
        },
    }


def _message_text_for_token_count(message: dict[str, Any]) -> str:
    parts: list[str] = []
    content = message.get("content")
    if content:
        parts.append(str(content))
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        if isinstance(function, dict):
            parts.append(str(function.get("name") or ""))
            parts.append(str(function.get("arguments") or ""))
        else:
            parts.append(str(tool_call))
    return "\n".join(part for part in parts if part)


def _count_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _summarize_validation_records(
    *,
    tokenizer: Any,
    messages_per_sample: list[list[dict[str, Any]]],
    output_token_counts: list[int],
    scores: list[float],
    reward_extra_infos: dict[str, list[Any]],
    response_length: int,
) -> dict[str, Any]:
    assistant_tokens: list[int] = []
    tool_response_tokens: list[int] = []
    for messages in messages_per_sample:
        sample_assistant_tokens = 0
        sample_tool_tokens = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            text = _message_text_for_token_count(message)
            if role == "assistant":
                sample_assistant_tokens += _count_tokens(tokenizer, text)
            elif role == "tool":
                sample_tool_tokens += _count_tokens(tokenizer, text)
        assistant_tokens.append(sample_assistant_tokens)
        tool_response_tokens.append(sample_tool_tokens)

    traces = reward_extra_infos.get("code_agent_trace", [])
    terminal_reasons: Counter[str] = Counter()
    tool_calls: list[int] = []
    tool_wall_seconds: list[float] = []
    judge_runtime_seconds: list[float] = []
    for i in range(len(messages_per_sample)):
        trace = traces[i] if i < len(traces) and isinstance(traces[i], dict) else {}
        reason = trace.get("terminal_reason") or _value_at(
            reward_extra_infos.get("code_agent_terminal_reason"),
            i,
            "response_length_exceeded",
        )
        terminal_reasons[str(reason or "response_length_exceeded")] += 1
        tool_calls.append(int(trace.get("num_tool_calls", 0) or 0))
        tool_wall_seconds.append(float(trace.get("tool_wall_seconds", 0.0) or 0.0))
        judge_runtime_seconds.append(float(trace.get("judge_runtime_seconds", 0.0) or 0.0))

    response_cap_hits = sum(1 for value in output_token_counts if value >= response_length)
    assistant_cap_hits = sum(1 for value in assistant_tokens if value >= response_length)
    return {
        "samples": len(messages_per_sample),
        "score_mean": _mean([float(score) for score in scores]),
        "assistant_tokens_total": sum(assistant_tokens),
        "assistant_tokens_mean": _mean(assistant_tokens),
        "assistant_tokens_p50": _percentile(assistant_tokens, 0.50),
        "assistant_tokens_p90": _percentile(assistant_tokens, 0.90),
        "assistant_tokens_max": max(assistant_tokens) if assistant_tokens else 0,
        "tool_response_tokens_total": sum(tool_response_tokens),
        "tool_response_tokens_mean": _mean(tool_response_tokens),
        "output_tokens_total": sum(output_token_counts),
        "output_tokens_mean": _mean(output_token_counts),
        "response_cap_hits": response_cap_hits,
        "assistant_cap_hits": assistant_cap_hits,
        "response_length": response_length,
        "tool_calls_mean": _mean(tool_calls),
        "tool_calls_max": max(tool_calls) if tool_calls else 0,
        "tool_wall_seconds_total": sum(tool_wall_seconds),
        "tool_wall_seconds_mean": _mean(tool_wall_seconds),
        "judge_runtime_seconds_total": sum(judge_runtime_seconds),
        "judge_runtime_seconds_mean": _mean(judge_runtime_seconds),
        "terminal_reasons": dict(sorted(terminal_reasons.items())),
    }


def _format_summary(prefix: str, summary: dict[str, Any], elapsed_seconds: float | None = None) -> str:
    pieces = [
        f"{prefix}",
        f"ts={_now_for_log()}",
        f"samples={summary['samples']}",
    ]
    if elapsed_seconds is not None:
        pieces.append(f"elapsed_s={elapsed_seconds:.1f}")
        if elapsed_seconds > 0:
            pieces.append(f"assistant_tok_s={summary['assistant_tokens_total'] / elapsed_seconds:.1f}")
    pieces.extend(
        [
            f"score_mean={summary['score_mean']:.4f}",
            f"assistant_tok_total={summary['assistant_tokens_total']}",
            f"assistant_tok_mean={summary['assistant_tokens_mean']:.1f}",
            f"assistant_tok_p50={summary['assistant_tokens_p50']:.0f}",
            f"assistant_tok_p90={summary['assistant_tokens_p90']:.0f}",
            f"assistant_tok_max={summary['assistant_tokens_max']}",
            f"tool_resp_tok_mean={summary['tool_response_tokens_mean']:.1f}",
            f"output_tok_mean={summary['output_tokens_mean']:.1f}",
            (
                f"response_cap_hits={summary['response_cap_hits']}/{summary['samples']}"
                f"@{summary['response_length']}"
            ),
            (
                f"assistant_cap_hits={summary['assistant_cap_hits']}/{summary['samples']}"
                f"@{summary['response_length']}"
            ),
            f"tool_calls_mean={summary['tool_calls_mean']:.2f}",
            f"tool_calls_max={summary['tool_calls_max']}",
            f"tool_wall_s_total={summary['tool_wall_seconds_total']:.1f}",
            f"judge_runtime_s_total={summary['judge_runtime_seconds_total']:.1f}",
            f"terminal_reasons={json.dumps(summary['terminal_reasons'], ensure_ascii=False, sort_keys=True)}",
        ]
    )
    return "[code-agent][eval] " + " ".join(pieces)


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
    assistant_token_counts: list[int],
    output_token_counts: list[int],
    reward_extra_infos: dict[str, list[Any]],
    response_length: int,
    rollout_indices: list[int],
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
    messages = [
        _messages_for_record(
            output,
            raw_prompts[i] if i < len(raw_prompts) else None,
            reward_extra_infos,
            i,
        )
        for i, output in enumerate(outputs)
    ]

    with open(filename, "a", encoding="utf-8") as f:
        for i in range(n):
            entry = _generation_record(
                trainer=trainer,
                input_text=inputs[i],
                output_text=outputs[i],
                messages=messages[i],
                ground_truth=gts[i] if i < len(gts) else None,
                score=float(scores[i]),
                reward_extra_infos=reward_extra_infos,
                index=i,
                assistant_token_count=assistant_token_counts[i],
                output_token_count=output_token_counts[i],
                response_length=response_length,
                rollout_index=rollout_indices[i] if i < len(rollout_indices) else 0,
            )
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
    assistant_token_counts: list[int],
    output_token_counts: list[int],
    reward_extra_infos_dict: dict[str, list[Any]],
    response_length: int,
    rollout_indices: list[int],
    dump_path: str,
) -> None:
    """Write final validation generations with parsed tool-event structure."""
    os.makedirs(dump_path, exist_ok=True)
    filename = os.path.join(dump_path, f"{trainer.global_steps}.jsonl")
    n = len(inputs)
    messages = [
        _messages_for_record(
            output,
            raw_prompts[i] if i < len(raw_prompts) else None,
            reward_extra_infos_dict,
            i,
        )
        for i, output in enumerate(outputs)
    ]

    with open(filename, "w", encoding="utf-8") as f:
        for i in range(n):
            entry = _generation_record(
                trainer=trainer,
                input_text=inputs[i],
                output_text=outputs[i],
                messages=messages[i],
                ground_truth=gts[i] if i < len(gts) else None,
                score=float(scores[i]),
                reward_extra_infos=reward_extra_infos_dict,
                index=i,
                assistant_token_count=assistant_token_counts[i],
                output_token_count=output_token_counts[i],
                response_length=response_length,
                rollout_index=rollout_indices[i] if i < len(rollout_indices) else 0,
            )
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Dumped generations with messages to {filename}")


def _combined_reward_extra_infos(batch: Any, reward_extra_infos_dict: dict[str, list[Any]]) -> dict[str, list[Any]]:
    combined: dict[str, list[Any]] = {
        key: _values_to_list(values)
        for key, values in reward_extra_infos_dict.items()
    }
    non_tensor_batch = getattr(batch, "non_tensor_batch", {})
    if not isinstance(non_tensor_batch, dict):
        return combined
    for key in _CODE_AGENT_DUMP_ONLY_KEYS:
        if key in non_tensor_batch:
            combined[key] = _values_to_list(non_tensor_batch[key])
    return combined


def _install_training_rollout_dump_patch() -> None:
    """Patch training rollout JSONL dump to use the structured trajectory schema.

    verl's native ``_log_rollout_data`` only writes prompt/response/score plus
    reward extra info.  It does not include agent-loop trace/tool events or
    exact response token counts, which are required for the GRPO rollout
    metrics in ``docs/grpo_rollout_metrics.md``.
    """

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.debug import marked_timer

    if getattr(RayPPOTrainer, "_code_agent_structured_rollout_dump", False):
        return

    def log_rollout_data_with_structure(
        self,
        batch,
        reward_extra_infos_dict: dict,
        timing_raw: dict,
        rollout_data_dir: str,
    ):
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            response_length = int(self.config.actor_rollout_ref.rollout.response_length)
            rollout_n = int(self.config.actor_rollout_ref.rollout.n)
            output_ids = batch.batch["responses"]
            response_masks = batch.batch.get("response_mask", None)
            pad_token_id = self.tokenizer.pad_token_id
            output_token_counts = [
                int((ids != pad_token_id).sum().item()) for ids in output_ids
            ]
            assistant_token_counts = (
                [int(mask.sum().item()) for mask in response_masks]
                if response_masks is not None
                else output_token_counts
            )
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                for item in batch
            ]
            raw_prompts = _values_to_list(batch.non_tensor_batch.get("raw_prompt", []))
            rollout_indices = [i % rollout_n for i in range(len(outputs))]
            reward_extra_infos = _combined_reward_extra_infos(batch, reward_extra_infos_dict)

            _dump_generations_with_structure(
                self,
                inputs=inputs,
                outputs=outputs,
                raw_prompts=raw_prompts,
                gts=sample_gts,
                scores=scores,
                assistant_token_counts=assistant_token_counts,
                output_token_counts=output_token_counts,
                reward_extra_infos_dict=reward_extra_infos,
                response_length=response_length,
                rollout_indices=rollout_indices,
                dump_path=rollout_data_dir,
            )

    RayPPOTrainer._log_rollout_data = log_rollout_data_with_structure
    RayPPOTrainer._code_agent_structured_rollout_dump = True


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
        sample_messages = []
        sample_output_token_counts = []
        sample_assistant_token_counts = []
        sample_rollout_indices = []
        validation_started_at = time.perf_counter()
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        response_length = int(self.config.actor_rollout_ref.rollout.response_length)
        val_repeat_n = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        print(
            "[code-agent][eval] "
            f"validation_start ts={_now_for_log()} "
            f"val_batches={len(self.val_dataloader)} "
            f"agent_workers={self.config.actor_rollout_ref.rollout.agent.num_workers} "
            f"response_length={response_length} "
            f"validation_data_dir={val_data_dir}"
        )

        # ── batch 循环 ──────────────────────────────────────────────
        for batch_index, test_data in enumerate(self.val_dataloader):
            batch_started_at = time.perf_counter()
            test_batch = DataProto.from_single_dict(test_data)

            # 如果数据中没有 uid，生成一个随机 UUID，方便后续追踪
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                    dtype=object,
                )

            test_batch = test_batch.repeat(
                repeat_times=val_repeat_n,
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
            print(
                "[code-agent][eval] "
                f"batch_start ts={_now_for_log()} "
                f"batch_index={batch_index} "
                f"samples={_dataproto_len(test_batch)} "
                f"padded_samples={_dataproto_len(test_gen_batch_padded)} "
                f"pad_size={pad_size} "
                f"size_divisor={size_divisor}"
            )
            generation_started_at = time.perf_counter()
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(
                test_gen_batch_padded
            )
            generation_seconds = time.perf_counter() - generation_started_at

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                self.checkpoint_manager.update_weights(self.global_steps)

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print(
                "[code-agent][eval] "
                f"batch_generation_end ts={_now_for_log()} "
                f"batch_index={batch_index} "
                f"generation_s={generation_seconds:.1f}"
            )

            # ── 增量 dump：解析 batch 结果，立即追加到 partial_0.jsonl ──
            output_ids = test_output_gen_batch.batch["responses"]
            response_masks = test_output_gen_batch.batch.get("response_mask", None)
            pad_token_id = self.tokenizer.pad_token_id
            output_token_counts = [
                int((ids != pad_token_id).sum().item()) for ids in output_ids
            ]
            assistant_token_counts = (
                [int(mask.sum().item()) for mask in response_masks]
                if response_masks is not None
                else output_token_counts
            )
            sample_output_token_counts.extend(output_token_counts)
            sample_assistant_token_counts.extend(assistant_token_counts)
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
            ]
            sample_outputs.extend(output_texts)
            rollout_indices = [i % val_repeat_n for i in range(len(output_texts))]
            sample_rollout_indices.extend(rollout_indices)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            input_ids = test_batch.batch["prompts"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
            ]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            reward_started_at = time.perf_counter()
            reward_tensor, reward_extra_info = extract_reward(test_batch)
            reward_seconds = time.perf_counter() - reward_started_at
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            batch_reward_extra_infos = {"reward": list(scores)}
            reward_extra_infos_dict["reward"].extend(scores)
            _append_reward_extra_infos(
                reward_extra_infos_dict,
                batch_reward_extra_infos,
                reward_extra_info,
            )

            # These are produced inside the real AgentLoopWorker and are not
            # reward outputs. Copy them directly so generation dumps can audit
            # actual tool execution instead of reconstructing it from text.
            for key in _CODE_AGENT_DUMP_ONLY_KEYS:
                if key not in test_batch.non_tensor_batch:
                    continue
                values = test_batch.non_tensor_batch[key]
                values_list = values.tolist() if isinstance(values, np.ndarray) else values
                values_list = values_list if isinstance(values_list, list) else [values_list]
                batch_reward_extra_infos[key] = values_list
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                reward_extra_infos_dict[key].extend(values_list)

            batch_messages = [
                _messages_for_record(
                    output,
                    raw_prompts[i] if i < len(raw_prompts) else None,
                    batch_reward_extra_infos,
                    i,
                )
                for i, output in enumerate(output_texts)
            ]
            sample_messages.extend(batch_messages)
            batch_summary = _summarize_validation_records(
                tokenizer=self.tokenizer,
                messages_per_sample=batch_messages,
                output_token_counts=output_token_counts,
                scores=scores,
                reward_extra_infos=batch_reward_extra_infos,
                response_length=response_length,
            )
            print(
                _format_summary(
                    f"batch_summary batch_index={batch_index} generation_s={generation_seconds:.1f} "
                    f"reward_s={reward_seconds:.1f}",
                    batch_summary,
                    elapsed_seconds=time.perf_counter() - batch_started_at,
                )
            )

            # ← 在这里增量写出，每个 batch 完成后立刻落盘
            _append_partial_generations(
                self,
                inputs=input_texts,
                outputs=output_texts,
                raw_prompts=raw_prompts,
                gts=ground_truths,
                scores=scores,
                assistant_token_counts=assistant_token_counts,
                output_token_counts=output_token_counts,
                reward_extra_infos=batch_reward_extra_infos,
                response_length=response_length,
                rollout_indices=rollout_indices,
                dump_path=val_data_dir,
                batch_index=batch_index,
            )
            print(
                "[code-agent][eval] "
                f"partial_dump_done ts={_now_for_log()} "
                f"batch_index={batch_index} path={val_data_dir}"
            )
            # ── 增量 dump 结束 ──

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )

        # ── 所有 batch 完成后 ──
        validation_seconds = time.perf_counter() - validation_started_at
        final_summary = _summarize_validation_records(
            tokenizer=self.tokenizer,
            messages_per_sample=sample_messages,
            output_token_counts=sample_output_token_counts,
            scores=sample_scores,
            reward_extra_infos=reward_extra_infos_dict,
            response_length=response_length,
        )
        print(_format_summary("validation_summary", final_summary, elapsed_seconds=validation_seconds))

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
                assistant_token_counts=sample_assistant_token_counts,
                output_token_counts=sample_output_token_counts,
                reward_extra_infos_dict=reward_extra_infos_dict,
                response_length=response_length,
                rollout_indices=sample_rollout_indices,
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
        metric_extra_infos_dict = {
            key: values
            for key, values in reward_extra_infos_dict.items()
            if key not in _CODE_AGENT_DUMP_ONLY_KEYS
        }
        return self._val_metrics_update(
            data_sources,
            sample_uids,
            metric_extra_infos_dict,
            sample_turns,
        )

    # 替换原方法 + 打哨兵防重复
    RayPPOTrainer._validate = validate_with_partial_dump
    RayPPOTrainer._code_agent_partial_dump = True


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
      2. _install_training_rollout_dump_patch: training rollout 结构化 dump
      3. _install_validation_partial_dump_patch: validation 增量 dump

    幂等：重复调用不会重复安装（各 patch 内部有哨兵检查）。
    """
    global _PATCHED
    if _PATCHED:
        return

    _install_numpy_json_patch()
    _install_training_rollout_dump_patch()
    _install_validation_partial_dump_patch()
    _PATCHED = True
    print("[code-agent] verl runtime patches enabled")
