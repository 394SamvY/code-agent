"""
Phase 2: Multi-turn Agentic GRPO Training
==========================================

继承 TRL 0.24.0 GRPOTrainer，override 关键方法实现多轮 agent-env 交互。

TRL 0.24.0 没有原生的 env_mask 支持，因此需要：
  1. override _generate_single_turn: 多轮 rollout 替代单次 generation
  2. override _generate_and_score_completions:
     - 缓存 test_list（按 prompt text 索引）
     - 将 env_mask 应用到 completion_mask 上（env token 不参与 loss 计算）
"""

from __future__ import annotations

import copy
import json
import sys
import yaml
from pathlib import Path
from typing import Any, Optional, Union

import torch
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.parser import parse_first_tool_call
from src.agent.prompts import build_agentic_messages
from src.data.dataset import load_mbpp, load_apps, problems_to_hf_dataset
from src.env.code_env import CodeEnvironment
from src.env.tools import TOOLS_SCHEMA
from src.reward.reward import make_multi_turn_reward


class MultiTurnGRPOTrainer(GRPOTrainer):
    """支持多轮 agentic rollout 的 GRPOTrainer.

    Override 两个方法:
    - _generate_single_turn: 多轮 rollout 替代单次 generation
    - _generate_and_score_completions: 缓存 test_list + 将 env_mask 注入 completion_mask

    env_mask 的传递路径:
      _generate_single_turn → self._current_env_masks (实例变量)
      → _generate_and_score_completions 读取并应用到 completion_mask
      → _compute_loss 中 completion_mask 已包含 env_mask 信息，无需额外修改
    """

    def __init__(self, *args, max_turns: int = 5, exec_timeout: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns = max_turns
        self.exec_timeout = exec_timeout
        self._prompt_to_problem: dict[str, dict] = {}
        # 用实例变量在 _generate_single_turn 和 _generate_and_score_completions 之间传递 env_mask
        self._current_env_masks: list[list[int]] = []

    # ------------------------------------------------------------------
    # Override: _generate_and_score_completions
    # ------------------------------------------------------------------

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """缓存 test_list + 将 env_mask 注入 completion_mask.

        流程:
        1. 从 inputs 中提取 test_list 并缓存（供 rollout 使用）
        2. 调用 super()（触发 _generate_single_turn 进行多轮 rollout）
        3. 读取 self._current_env_masks，对齐并应用到 output["completion_mask"]
        """
        # Step 1: 缓存 test_list
        for item in inputs:
            prompt = item.get("prompt")
            if prompt is None:
                continue
            problem_text = self._extract_problem_text(prompt)
            tl = item.get("test_list", [])
            if isinstance(tl, str):
                tl = json.loads(tl)
            ep = item.get("entry_point", "")
            self._prompt_to_problem[problem_text] = {
                "test_list": tl,
                "entry_point": ep,
            }

        # Step 2: 调用 super()，内部会调用 _generate_single_turn
        output = super()._generate_and_score_completions(inputs)

        # Step 3: 监控 all-zero reward group 比例
        self._log_reward_stats(output)

        # Step 4: 将 env_mask 应用到 completion_mask
        if self._current_env_masks:
            device = output["completion_mask"].device
            completion_mask = output["completion_mask"]  # (B, max_completion_len)
            max_len = completion_mask.size(1)

            for i, env_mask in enumerate(self._current_env_masks):
                if i >= completion_mask.size(0):
                    break
                # 将 env_mask pad/truncate 到 max_len
                if len(env_mask) >= max_len:
                    mask_tensor = torch.tensor(env_mask[:max_len], device=device)
                else:
                    # pad with 0（超出 env_mask 的部分是 padding，本就应该是 0）
                    padded = env_mask + [0] * (max_len - len(env_mask))
                    mask_tensor = torch.tensor(padded, device=device)
                # completion_mask 已经处理了 padding（padding 位置是 0），
                # 再乘以 env_mask 把 env token 也置为 0
                completion_mask[i] = completion_mask[i] * mask_tensor

            output["completion_mask"] = completion_mask
            self._current_env_masks = []  # 清空，防止下一轮误用

        return output

    # ------------------------------------------------------------------
    # Override: 多轮 rollout
    # ------------------------------------------------------------------

    def _generate_single_turn(self, prompts: list, images: Optional[list] = None):
        """将单次 generation 替换为多轮 agent-env 交互.

        TRL 0.24.0 签名: _generate_single_turn(self, prompts, images)
        返回: (prompt_ids, completion_ids, logprobs, forward_kwargs)

        env_mask 通过 self._current_env_masks 传递给 _generate_and_score_completions。
        """
        device = self.accelerator.device
        tokenizer = self.processing_class

        unwrapped = self.accelerator.unwrap_model(self.model)

        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_env_masks: list[list[int]] = []

        for prompt in prompts:
            problem_text = self._extract_problem_text(prompt)
            prob_data = self._find_problem_data(prompt)
            test_list = prob_data.get("test_list", [])

            env = CodeEnvironment(
                problem_description=problem_text,
                test_list=test_list,
                entry_point=prob_data.get("entry_point", ""),
                timeout=self.exec_timeout,
            )

            if isinstance(prompt, list):
                messages = copy.deepcopy(prompt)
            else:
                messages = build_agentic_messages(problem_text)

            # 初始编码（含 tools 注入）
            prompt_ids_list = self._encode_messages(
                tokenizer, messages, add_generation_prompt=True
            )
            completion_ids: list[int] = []
            env_mask: list[int] = []
            current_ids = list(prompt_ids_list)

            for _turn in range(self.max_turns):
                input_tensor = torch.tensor([current_ids], device=device)

                with torch.no_grad():
                    gen_kwargs = {
                        "max_new_tokens": self.max_completion_length,
                        "do_sample": True,
                        "temperature": self.temperature,
                        "top_p": self.top_p if self.top_p is not None else 0.95,
                        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                    }
                    outputs = unwrapped.generate(input_tensor, **gen_kwargs)

                new_ids = outputs[0][len(current_ids):].tolist()
                if not new_ids:
                    break

                completion_ids.extend(new_ids)
                env_mask.extend([1] * len(new_ids))  # agent tokens
                current_ids.extend(new_ids)

                response_text = tokenizer.decode(new_ids, skip_special_tokens=False)
                messages.append({"role": "assistant", "content": response_text})

                tool_call = parse_first_tool_call(response_text)
                if not tool_call:
                    print(f"  [Rollout] Turn {_turn+1}: {len(new_ids)} tokens, NO tool_call. "
                          f"Response (first 200): {repr(response_text[:200])}")
                else:
                    print(f"  [Rollout] Turn {_turn+1}: {len(new_ids)} tokens, "
                          f"tool_call={tool_call.name}")
                if tool_call is None:
                    break

                observation = env.execute_tool(tool_call.name, **tool_call.arguments)
                messages.append({"role": "tool", "content": observation})

                # 重新编码完整 messages 得到精确的 token ids
                full_ids = self._encode_messages(
                    tokenizer, messages,
                    add_generation_prompt=(tool_call.name != "submit"),
                )

                # 计算 observation 新增的 token（diff）
                obs_token_len = len(full_ids) - len(current_ids)
                if obs_token_len > 0:
                    new_obs_ids = full_ids[len(current_ids):]
                    completion_ids.extend(new_obs_ids)
                    env_mask.extend([0] * len(new_obs_ids))  # env tokens

                current_ids = full_ids

                # 安全对齐：重新编码可能导致长度变化
                expected_completion_len = len(current_ids) - len(prompt_ids_list)
                if len(env_mask) < expected_completion_len:
                    env_mask.extend([0] * (expected_completion_len - len(env_mask)))
                elif len(env_mask) > expected_completion_len:
                    env_mask = env_mask[:expected_completion_len]
                completion_ids = current_ids[len(prompt_ids_list):]

                if env.is_done or tool_call.name == "submit":
                    break

            if not completion_ids:
                eos_id = tokenizer.eos_token_id or 0
                completion_ids = [eos_id]
                env_mask = [1]

            all_prompt_ids.append(prompt_ids_list)
            all_completion_ids.append(completion_ids)
            all_env_masks.append(env_mask)

        # 通过实例变量传递 env_mask
        self._current_env_masks = all_env_masks

        # TRL 0.24.0 返回格式: (prompt_ids, completion_ids, logprobs, forward_kwargs)
        return all_prompt_ids, all_completion_ids, None, {}

    # ------------------------------------------------------------------
    # Monitoring: all-zero group 检测
    # ------------------------------------------------------------------

    def _log_reward_stats(self, output: dict) -> None:
        """监控 reward 的 all-zero group 比例.

        当一个 prompt 的所有 G 条生成 reward 全相同时，GRPO advantage 为 0，
        该条数据不贡献梯度。如果比例过高（>80%），训练可能停滞。
        """
        rewards = output.get("rewards")
        if rewards is None:
            return

        try:
            if isinstance(rewards, torch.Tensor):
                rewards_np = rewards.detach().cpu().float().numpy()
            else:
                return

            num_generations = self.num_generations if hasattr(self, "num_generations") else 4
            batch_size = rewards_np.shape[0]
            num_groups = batch_size // num_generations

            if num_groups == 0:
                return

            zero_groups = 0
            for g in range(num_groups):
                start = g * num_generations
                end = start + num_generations
                group_rewards = rewards_np[start:end]
                if group_rewards.std() < 1e-8:  # 所有 reward 相同
                    zero_groups += 1

            zero_ratio = zero_groups / num_groups
            mean_reward = float(rewards_np.mean())

            # 输出到日志
            if hasattr(self, "log"):
                self.log({
                    "reward/all_zero_group_ratio": zero_ratio,
                    "reward/mean": mean_reward,
                    "reward/std": float(rewards_np.std()),
                })

            if zero_ratio > 0.8:
                print(f"  [WARNING] all-zero group ratio = {zero_ratio:.2f} "
                      f"({zero_groups}/{num_groups}). Training may stagnate!")
        except Exception:
            pass  # 监控不应影响训练

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_messages(
        tokenizer, messages: list[dict], add_generation_prompt: bool = True
    ) -> list[int]:
        """用 Qwen chat template + tools 编码 messages."""
        encoded = tokenizer.apply_chat_template(
            messages,
            tools=TOOLS_SCHEMA,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
        if isinstance(encoded, torch.Tensor):
            return encoded.squeeze(0).tolist()
        if hasattr(encoded, "input_ids"):
            ids = encoded.input_ids
            return ids[0] if isinstance(ids[0], list) else list(ids)
        return list(encoded) if not isinstance(encoded, list) else encoded

    def _extract_problem_text(self, prompt) -> str:
        """从 prompt 中提取题目描述."""
        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
        return str(prompt)

    def _find_problem_data(self, prompt) -> dict:
        """从缓存中查找题目的 test_list 等数据."""
        if self._prompt_to_problem:
            problem_text = self._extract_problem_text(prompt)
            data = self._prompt_to_problem.get(problem_text)
            if data:
                return data
        return {"test_list": [], "entry_point": ""}


# ======================================================================
# main: 启动多轮训练
# ======================================================================


def main(config_path: str = "configs/multi_turn.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    grpo_cfg = cfg["grpo"]
    agent_cfg = cfg.get("agent", {})

    print("Loading dataset...")
    max_samples = data_cfg.get("max_train_samples")
    train_problems = load_mbpp(
        version="full", split=data_cfg["split_train"], max_samples=max_samples
    )
    val_problems = load_mbpp(
        version="full", split=data_cfg["split_val"], max_samples=max_samples
    )

    # 可选：追加 APPS 数据集扩充训练数据
    if data_cfg.get("use_apps"):
        apps_cfg = data_cfg.get("apps", {})
        print("  Loading APPS dataset for augmentation...")
        apps_problems = load_apps(
            split="train",
            difficulty=apps_cfg.get("difficulty", "introductory"),
            max_samples=apps_cfg.get("max_samples"),
        )
        train_problems.extend(apps_problems)
        print(f"  APPS added {len(apps_problems)} problems, total: {len(train_problems)}")

    def to_chat_messages(prompt_text):
        return build_agentic_messages(prompt_text)

    train_ds = problems_to_hf_dataset(train_problems, prompt_formatter=to_chat_messages)
    val_ds = problems_to_hf_dataset(val_problems, prompt_formatter=to_chat_messages)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    max_response_length = data_cfg.get("max_response_length", 2048)

    training_args = GRPOConfig(
        output_dir=cfg["output_dir"],
        num_generations=grpo_cfg["num_generations"],
        max_completion_length=max_response_length,
        beta=grpo_cfg["beta"],
        learning_rate=grpo_cfg["learning_rate"],
        per_device_train_batch_size=grpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo_cfg["gradient_accumulation_steps"],
        num_train_epochs=grpo_cfg["num_train_epochs"],
        save_steps=grpo_cfg["save_steps"],
        logging_steps=grpo_cfg["logging_steps"],
        bf16=torch.cuda.is_available(),
        use_cpu=cfg.get("use_cpu", False),
        remove_unused_columns=False,
        report_to=cfg.get("report_to", "wandb"),
    )

    peft_config = None
    if model_cfg.get("use_lora"):
        peft_config = LoraConfig(
            r=model_cfg["lora_r"],
            lora_alpha=model_cfg["lora_alpha"],
            target_modules=model_cfg["lora_target_modules"],
            task_type="CAUSAL_LM",
        )

    # ---- Reward 配置 ----
    reward_cfg = cfg.get("reward", {})
    reward_func = make_multi_turn_reward(
        format_weight=reward_cfg.get("format_weight", 0.1),
        behavioral_weight=reward_cfg.get("behavioral_weight", 0.05),
    )

    resume_from = model_cfg.get("resume_from") or model_cfg["name"]

    print("Initializing MultiTurnGRPOTrainer...")
    trainer = MultiTurnGRPOTrainer(
        model=resume_from,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        reward_funcs=reward_func,
        peft_config=peft_config,
        max_turns=agent_cfg.get("max_turns", 5),
        exec_timeout=reward_cfg.get("execution_timeout", 5),
    )

    print("Starting multi-turn training...")
    trainer.train()
    trainer.save_model()
    print(f"Model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/multi_turn.yaml"
    main(config)
