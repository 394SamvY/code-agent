"""verl FSDP checkpoint / PEFT adapter 转换工具。

三种模式：
  adapter    — 从 FSDP checkpoint 导出 PEFT adapter，用于 verl GRPO 续训
  merge      — 从 FSDP checkpoint 合并为完整 HF 模型，用于 SGLang 推理
  merge-adapter — 从已导出的 PEFT adapter 合并为完整 HF 模型（无需 FSDP checkpoint）

用法:
    # 导出 PEFT adapter（训练后常规操作）
    python scripts/convert_checkpoint.py \
        --base_model /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct \
        --ckpt_path outputs/sft_ckpt/global_step_310/model_world_size_1_rank_0.pt \
        --output_dir /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct-SFT-LoRA

    # 从 FSDP checkpoint 合并为完整模型
    python scripts/convert_checkpoint.py --mode merge \
        --base_model /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct \
        --ckpt_path outputs/sft_ckpt/global_step_310/model_world_size_1_rank_0.pt \
        --output_dir /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct-SFT

    # 从 adapter 合并为完整模型（只需 base + adapter，无需 FSDP checkpoint）
    python scripts/convert_checkpoint.py --mode merge-adapter \
        --base_model /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct \
        --adapter_path /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct-SFT-LoRA \
        --output_dir /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct-SFT
"""

import argparse
import json
import torch
from pathlib import Path
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_fsdp_checkpoint(base_model: str, ckpt_path: str, lora_meta: dict):
    """加载基座模型，应用 LoRA 配置，加载 FSDP checkpoint 权重。"""
    print(f"Base model: {base_model}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"LoRA meta: r={lora_meta['r']}, alpha={lora_meta['lora_alpha']}")

    print("\nLoading base model (CPU, bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    print("Loading FSDP checkpoint state dict...")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    target_modules = sorted({
        k.split(".")[k.split(".").index("lora_A") - 1]
        for k in state_dict if "lora_A" in k
    })
    print(f"Detected LoRA target_modules from checkpoint: {target_modules}")

    lora_config = LoraConfig(
        r=lora_meta["r"],
        lora_alpha=lora_meta["lora_alpha"],
        target_modules=target_modules,
        task_type=lora_meta["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.load_state_dict(state_dict, strict=True)
    print(f"  Loaded {len(state_dict)} keys")
    del state_dict

    return model


def export_adapter(model, output_path: Path):
    """保存 PEFT adapter（adapter_config.json + adapter_model.safetensors）。"""
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving PEFT adapter to {output_path}...")
    model.save_pretrained(output_path)

    adapter_size = sum(f.stat().st_size for f in output_path.iterdir() if f.is_file()) / 1024 / 1024
    print(f"Done! Adapter size: {adapter_size:.1f} MB")


def merge_and_save(model, base_model: str, output_path: Path):
    """合并 LoRA 到基座模型，保存为完整 HF 模型。"""
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("Merging LoRA into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done!")


def merge_adapter_and_save(base_model: str, adapter_path: str, output_path: Path):
    """从已导出的 PEFT adapter 直接合并为完整 HF 模型。"""
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {base_model}")
    print(f"Adapter: {adapter_path}")

    if torch.cuda.is_available():
        device_map, device_label = "auto", "GPU"
    else:
        device_map, device_label = "cpu", "CPU（需要 ≥30GB 内存）"
    print(f"\nLoading base model ({device_label}, bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging LoRA into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path, max_shard_size="5GB")
    tokenizer.save_pretrained(output_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="verl FSDP checkpoint / PEFT adapter 转换工具")
    parser.add_argument("--mode", choices=["adapter", "merge", "merge-adapter"], default="adapter",
                        help="adapter: 导出 PEFT adapter | merge: 从 FSDP ckpt 合并 | merge-adapter: 从 adapter 合并")
    parser.add_argument("--base_model", type=str, required=True,
                        help="基座模型路径")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="FSDP checkpoint 文件路径（adapter/merge 模式必需）")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="已导出的 PEFT adapter 目录（merge-adapter 模式必需）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--lora_meta", type=str, default=None,
                        help="lora_train_meta.json 路径（默认从 ckpt 目录自动检测）")
    args = parser.parse_args()

    output_path = Path(args.output_dir)

    if args.mode == "merge-adapter":
        if not args.adapter_path:
            parser.error("--adapter_path is required for merge-adapter mode")
        merge_adapter_and_save(args.base_model, args.adapter_path, output_path)
    else:
        if not args.ckpt_path:
            parser.error("--ckpt_path is required for adapter/merge mode")

        ckpt_dir = Path(args.ckpt_path).parent
        if args.lora_meta is None:
            args.lora_meta = str(ckpt_dir / "lora_train_meta.json")

        with open(args.lora_meta) as f:
            lora_meta = json.load(f)

        model = load_fsdp_checkpoint(args.base_model, args.ckpt_path, lora_meta)

        if args.mode == "adapter":
            export_adapter(model, output_path)
        else:
            merge_and_save(model, args.base_model, output_path)


if __name__ == "__main__":
    main()
