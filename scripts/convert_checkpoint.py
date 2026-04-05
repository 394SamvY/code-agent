"""verl FSDP checkpoint 转换工具。

支持全量微调和 LoRA 两种模式。

模式：
  full-merge    — 从全量微调的 FSDP checkpoint 导出为完整 HF 模型（用于 SGLang 推理）
  adapter       — 从 LoRA FSDP checkpoint 导出 PEFT adapter
  merge         — 从 LoRA FSDP checkpoint 合并为完整 HF 模型
  merge-adapter — 从已导出的 PEFT adapter 合并为完整 HF 模型

用法:
    # 全量微调 → 导出完整模型（最常用）
    python scripts/convert_checkpoint.py --mode full-merge \
        --ckpt_path outputs/verl_grpo/checkpoints/global_step_50 \
        --output_dir /root/autodl-tmp/models/Qwen3-8B-GRPO

    # LoRA → 导出 adapter
    python scripts/convert_checkpoint.py --mode adapter \
        --base_model /root/autodl-tmp/models/Qwen3-8B \
        --ckpt_path outputs/sft_ckpt/global_step_310/model_world_size_1_rank_0.pt \
        --output_dir /root/autodl-tmp/models/Qwen3-8B-SFT-LoRA

    # LoRA → 合并为完整模型
    python scripts/convert_checkpoint.py --mode merge \
        --base_model /root/autodl-tmp/models/Qwen3-8B \
        --ckpt_path outputs/sft_ckpt/global_step_310/model_world_size_1_rank_0.pt \
        --output_dir /root/autodl-tmp/models/Qwen3-8B-SFT

    # 从 adapter 合并（无需 FSDP checkpoint）
    python scripts/convert_checkpoint.py --mode merge-adapter \
        --base_model /root/autodl-tmp/models/Qwen3-8B \
        --adapter_path /root/autodl-tmp/models/Qwen3-8B-SFT-LoRA \
        --output_dir /root/autodl-tmp/models/Qwen3-8B-SFT
"""

import argparse
import json
import glob
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Full fine-tuning: FSDP checkpoint → HF model
# ---------------------------------------------------------------------------

def full_merge_and_save(ckpt_path: str, output_path: Path):
    """从全量微调的 verl FSDP checkpoint 目录导出为完整 HF 模型。

    verl 保存全量微调 checkpoint 时，每个 rank 一个文件：
      model_world_size_N_rank_0.pt, model_world_size_N_rank_1.pt, ...
    这些是 FSDP flat_param 格式，需要通过 verl 的工具合并。
    """
    ckpt_dir = Path(ckpt_path)
    output_path.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(glob.glob(str(ckpt_dir / "model_world_size_*_rank_*.pt")))
    if not pt_files:
        pt_files = sorted(glob.glob(str(ckpt_dir / "*.pt")))

    if not pt_files:
        raise FileNotFoundError(
            f"No checkpoint files found in {ckpt_dir}. "
            f"Expected model_world_size_*_rank_*.pt files."
        )

    print(f"Found {len(pt_files)} checkpoint shard(s) in {ckpt_dir}")

    if len(pt_files) == 1:
        print(f"Loading single shard: {pt_files[0]}")
        state_dict = torch.load(pt_files[0], map_location="cpu", weights_only=False)
    else:
        print("Loading and merging multiple FSDP shards...")
        state_dict = {}
        for f in pt_files:
            print(f"  Loading {Path(f).name}...")
            shard = torch.load(f, map_location="cpu", weights_only=False)
            state_dict.update(shard)
            del shard

    has_lora = any("lora_A" in k or "lora_B" in k for k in state_dict)
    if has_lora:
        raise ValueError(
            "Checkpoint contains LoRA parameters. Use --mode adapter or --mode merge instead."
        )

    print(f"Total keys: {len(state_dict)}")

    hf_config_file = ckpt_dir / "config.json"
    if hf_config_file.exists():
        print("Found config.json in checkpoint dir, loading model from checkpoint config...")
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir), torch_dtype=torch.bfloat16, device_map="cpu",
        )
    else:
        print("Loading state dict directly (no config.json in checkpoint dir)...")
        model = AutoModelForCausalLM.from_config(
            AutoModelForCausalLM.config_class.from_dict(state_dict.get("config", {}))
        )

    try:
        model.load_state_dict(state_dict, strict=True)
        print("State dict loaded (strict=True)")
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
        print("State dict loaded (strict=False, some keys may be missing/extra)")

    del state_dict

    tokenizer_dir = ckpt_dir / "tokenizer"
    if not tokenizer_dir.exists():
        tokenizer_dir = ckpt_dir
    if (tokenizer_dir / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    else:
        tokenizer = None
        print("Warning: No tokenizer found in checkpoint. Copy tokenizer manually.")

    print(f"Saving HF model to {output_path}...")
    model.save_pretrained(output_path, max_shard_size="5GB")
    if tokenizer:
        tokenizer.save_pretrained(output_path)
    print("Done!")


# ---------------------------------------------------------------------------
# LoRA: FSDP checkpoint → adapter / merged model
# ---------------------------------------------------------------------------

def load_fsdp_lora_checkpoint(base_model: str, ckpt_path: str, lora_meta: dict):
    """加载基座模型，应用 LoRA 配置，加载 FSDP checkpoint 权重。"""
    from peft import LoraConfig, get_peft_model

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
    from peft import PeftModel

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
    parser = argparse.ArgumentParser(description="verl FSDP checkpoint 转换工具")
    parser.add_argument("--mode", choices=["full-merge", "adapter", "merge", "merge-adapter"],
                        default="full-merge",
                        help="full-merge: 全量微调导出 | adapter: 导出 LoRA adapter | "
                             "merge: 从 LoRA FSDP ckpt 合并 | merge-adapter: 从 adapter 合并")
    parser.add_argument("--base_model", type=str, default=None,
                        help="基座模型路径（LoRA 模式必需）")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="FSDP checkpoint 路径（full-merge/adapter/merge 必需）")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="已导出的 PEFT adapter 目录（merge-adapter 模式必需）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--lora_meta", type=str, default=None,
                        help="lora_train_meta.json 路径（LoRA 模式，默认从 ckpt 目录自动检测）")
    args = parser.parse_args()

    output_path = Path(args.output_dir)

    if args.mode == "full-merge":
        if not args.ckpt_path:
            parser.error("--ckpt_path is required for full-merge mode")
        full_merge_and_save(args.ckpt_path, output_path)

    elif args.mode == "merge-adapter":
        if not args.adapter_path or not args.base_model:
            parser.error("--adapter_path and --base_model are required for merge-adapter mode")
        merge_adapter_and_save(args.base_model, args.adapter_path, output_path)

    else:
        if not args.ckpt_path or not args.base_model:
            parser.error("--ckpt_path and --base_model are required for adapter/merge mode")

        ckpt_dir = Path(args.ckpt_path).parent
        if args.lora_meta is None:
            args.lora_meta = str(ckpt_dir / "lora_train_meta.json")

        with open(args.lora_meta) as f:
            lora_meta = json.load(f)

        model = load_fsdp_lora_checkpoint(args.base_model, args.ckpt_path, lora_meta)

        if args.mode == "adapter":
            export_adapter(model, output_path)
        else:
            merge_and_save(model, args.base_model, output_path)


if __name__ == "__main__":
    main()
