"""Runtime patches for using verl validation as OJ-like online evaluation.

The patches in this module are intentionally opt-in. They are installed by
``scripts.verl_main_wrapper.CodeAgentTaskRunner`` inside the CPU TaskRunner Ray
actor, so GPU workers do not import verl or torch before Ray finalizes their
per-worker CUDA visibility.
"""

from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from typing import Any

import numpy as np


_PATCHED = False


def _install_numpy_json_patch() -> None:
    """Make stdlib json handle numpy scalar/array values in verl dumps."""

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


def _append_partial_generations(
    trainer: Any,
    *,
    inputs: list[str],
    outputs: list[str],
    gts: list[Any],
    scores: list[float],
    reward_extra_infos: dict[str, list[Any]],
    dump_path: str | None,
    batch_index: int,
) -> None:
    """Append one completed validation batch before the full pass finishes."""

    if not dump_path:
        return

    os.makedirs(dump_path, exist_ok=True)
    filename = os.path.join(dump_path, f"partial_{trainer.global_steps}.jsonl")
    n = len(inputs)
    base_data = {
        "input": inputs,
        "output": outputs,
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


def _install_validation_partial_dump_patch() -> None:
    """Patch RayPPOTrainer._validate to write per-batch partial generations."""

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
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        val_data_dir = self.config.trainer.get("validation_data_dir", None)

        for batch_index, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                    dtype=object,
                )

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                for item in test_batch
            ]
            sample_gts.extend(ground_truths)

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

            _append_partial_generations(
                self,
                inputs=input_texts,
                outputs=output_texts,
                gts=ground_truths,
                scores=scores,
                reward_extra_infos=batch_reward_extra_infos,
                dump_path=val_data_dir,
                batch_index=batch_index,
            )

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )

        self._maybe_log_val_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
        )

        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
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

    RayPPOTrainer._validate = validate_with_partial_dump
    RayPPOTrainer._code_agent_partial_dump = True


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    _install_numpy_json_patch()
    _install_validation_partial_dump_patch()
    _PATCHED = True
    print("[code-agent] verl runtime patches enabled")
