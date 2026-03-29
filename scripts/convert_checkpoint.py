"""verl FSDP checkpoint 转换工具。

两种模式：
  adapter — 导出 PEFT adapter（~39MB），用于 verl GRPO 续训 / PeftModel 加载
  merge   — 合并为完整 HF 模型（~15GB），用于 SGLang 推理

用法:
    # 导出 PEFT adapter（训练后常规操作）
    python scripts/convert_checkpoint.py \
        --base_model /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct \
        --ckpt_path outputs/sft_ckpt/global_step_63/model_world_size_1_rank_0.pt \
        --output_dir /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct-SFT-LoRA

    # 合并为完整模型（评估前临时操作）
    python scripts/convert_checkpoint.py --mode merge \
        --base_model /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct \
        --ckpt_path outputs/sft_ckpt/global_step_63/model_world_size_1_rank_0.pt \
        --output_dir /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct-SFT
"""

import argparse
import json
import torch
from pathlib import Path
from peft import LoraConfig, get_peft_model
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

    lora_config = LoraConfig(
        r=lora_meta["r"],
        lora_alpha=lora_meta["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=lora_meta["task_type"],
    )
    model = get_peft_model(model, lora_config)

    print("Loading FSDP checkpoint state dict...")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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


def main():
    parser = argparse.ArgumentParser(description="verl FSDP checkpoint 转换工具")
    parser.add_argument("--mode", choices=["adapter", "merge"], default="adapter",
                        help="adapter: 导出 PEFT adapter (39MB) | merge: 合并为完整 HF 模型 (15GB)")
    parser.add_argument("--base_model", type=str, required=True,
                        help="基座模型路径")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="FSDP checkpoint 文件路径 (model_world_size_*_rank_*.pt)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--lora_meta", type=str, default=None,
                        help="lora_train_meta.json 路径（默认从 ckpt 目录自动检测）")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_path).parent
    if args.lora_meta is None:
        args.lora_meta = str(ckpt_dir / "lora_train_meta.json")

    with open(args.lora_meta) as f:
        lora_meta = json.load(f)

    model = load_fsdp_checkpoint(args.base_model, args.ckpt_path, lora_meta)
    output_path = Path(args.output_dir)

    if args.mode == "adapter":
        export_adapter(model, output_path)
    else:
        merge_and_save(model, args.base_model, output_path)


if __name__ == "__main__":
    main()
