"""
Phase 1: One-shot GRPO Training
================================

标准 TRL GRPOTrainer：模型一次性生成代码，
用真实代码执行作为 reward（通过=1, 失败=0）。
"""

from __future__ import annotations

import sys
import yaml
from pathlib import Path

import torch
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import load_mbpp, load_apps, problems_to_hf_dataset
from src.agent.prompts import SYSTEM_PROMPT_ONE_SHOT, build_one_shot_prompt
from src.reward.reward import code_execution_reward


def build_prompt_column(problem_prompt: str) -> list[dict[str, str]]:
    """将题目描述转成 chat messages 格式（TRL 需要）."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_ONE_SHOT},
        {"role": "user", "content": build_one_shot_prompt(problem_prompt)},
    ]


def main(config_path: str = "configs/one_shot.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    grpo_cfg = cfg["grpo"]

    # ---- 数据 ----
    print("Loading dataset...")
    max_samples = data_cfg.get("max_train_samples")
    train_problems = load_mbpp(
        version="full", split=data_cfg["split_train"], max_samples=max_samples
    )
    val_problems = load_mbpp(
        version="full", split=data_cfg["split_val"], max_samples=max_samples
    )

    # 可选：追加 APPS 数据集
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
        return [
            {"role": "system", "content": SYSTEM_PROMPT_ONE_SHOT},
            {"role": "user", "content": build_one_shot_prompt(prompt_text)},
        ]

    train_ds = problems_to_hf_dataset(train_problems, prompt_formatter=to_chat_messages)
    val_ds = problems_to_hf_dataset(val_problems, prompt_formatter=to_chat_messages)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ---- 训练配置 ----
    training_args = GRPOConfig(
        output_dir=cfg["output_dir"],
        num_generations=grpo_cfg["num_generations"],
        max_completion_length=data_cfg["max_completion_length"],
        beta=grpo_cfg["beta"],
        learning_rate=grpo_cfg["learning_rate"],
        per_device_train_batch_size=grpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo_cfg["gradient_accumulation_steps"],
        num_train_epochs=grpo_cfg["num_train_epochs"],
        save_steps=grpo_cfg["save_steps"],
        logging_steps=grpo_cfg["logging_steps"],
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to=cfg.get("report_to", "wandb"),
    )

    # ---- LoRA ----
    peft_config = None
    if model_cfg.get("use_lora"):
        peft_config = LoraConfig(
            r=model_cfg["lora_r"],
            lora_alpha=model_cfg["lora_alpha"],
            target_modules=model_cfg["lora_target_modules"],
            task_type="CAUSAL_LM",
        )

    # ---- Trainer ----
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model_cfg["name"],
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        reward_funcs=code_execution_reward,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model()
    print(f"Model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/one_shot.yaml"
    main(config)
