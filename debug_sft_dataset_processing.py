#!/usr/bin/env python3
"""
模拟 verl SFT dataset 对第一条样本的完整处理流程。
展示 tokenization、loss mask、attention mask 等所有输出。

使用方法：
  在有 verl 环境的服务器上运行：
  python3 debug_sft_dataset_processing.py
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

# 添加项目路径以便 import verl
project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir))

# 导入 verl 的 dataset 类
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


def main():
    # 1. 配置参数（对应 train_sft_with_verl.sh 的配置）
    data_file = project_dir / "data/verl/sft/sft_accepted_train.parquet"

    # 模型路径：优先使用环境变量，否则使用默认路径
    import os
    model_path = os.getenv("MODEL_PATH", "/root/autodl-tmp/models/Qwen3-8B")

    print("=" * 80)
    print("verl SFT Dataset 处理流程模拟")
    print("=" * 80)
    print(f"\n数据文件: {data_file}")
    print(f"模型路径: {model_path}")

    # 2. 加载 tokenizer
    print("\n[1] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    print(f"    tokenizer 类型: {type(tokenizer).__name__}")
    print(f"    vocab_size: {tokenizer.vocab_size}")
    print(f"    pad_token_id: {tokenizer.pad_token_id}")
    print(f"    eos_token_id: {tokenizer.eos_token_id}")

    # 3. 构造 data_config（对应脚本中的 data.* 参数）
    data_config = DictConfig({
        "messages_key": "messages",
        "tools_key": "tools",
        "enable_thinking_key": "enable_thinking",
        "enable_thinking_default": False,
        "pad_mode": "no_padding",  # 脚本默认值
        "max_length": 20480,       # 脚本默认值
        "truncation": "right",     # 脚本默认值
        "ignore_input_ids_mismatch": True,  # 脚本默认值
        "apply_chat_template_kwargs": {},
    })

    print("\n[2] 创建 MultiTurnSFTDataset...")
    print(f"    配置:")
    print(f"      - messages_key: {data_config.messages_key}")
    print(f"      - tools_key: {data_config.tools_key}")
    print(f"      - enable_thinking_key: {data_config.enable_thinking_key}")
    print(f"      - enable_thinking_default: {data_config.enable_thinking_default}")
    print(f"      - pad_mode: {data_config.pad_mode}")
    print(f"      - max_length: {data_config.max_length}")
    print(f"      - truncation: {data_config.truncation}")

    dataset = MultiTurnSFTDataset(
        parquet_files=str(data_file),
        tokenizer=tokenizer,
        config=data_config,
        processor=None,  # 纯文本，不需要 processor
        max_samples=-1,
    )

    print(f"\n    数据集大小: {len(dataset)}")

    # 4. 读取原始第一条样本
    print("\n[3] 原始第一条样本内容:")
    df = pd.read_parquet(data_file)
    first_row = df.iloc[0]
    print(f"    task_id: {first_row['task_id']}")
    print(f"    title: {first_row['title']}")
    print(f"    enable_thinking: {first_row['enable_thinking']}")
    print(f"    turns: {first_row['turns']}")
    print(f"    messages 数量: {len(first_row['messages'])}")

    print("\n    messages 内容:")
    for i, msg in enumerate(first_row['messages']):
        role = msg['role']
        content = msg.get('content', '')
        tool_calls = msg.get('tool_calls', [])

        print(f"\n      Turn {i} [{role}]:")
        if content:
            content_preview = content[:200] + "..." if len(content) > 200 else content
            print(f"        content: {content_preview}")
        if tool_calls:
            print(f"        tool_calls: {len(tool_calls)} calls")
            for tc in tool_calls[:2]:  # 只显示前2个
                print(f"          - {tc.get('function', {}).get('name', 'unknown')}")

    print(f"\n    tools 数量: {len(first_row['tools']) if first_row['tools'] else 0}")
    if first_row['tools']:
        print(f"    tools 列表:")
        for tool in first_row['tools'][:3]:  # 只显示前3个
            print(f"      - {tool.get('function', {}).get('name', 'unknown')}")

    # 5. 获取处理后的样本
    print("\n" + "=" * 80)
    print("[4] 经过 MultiTurnSFTDataset 处理后的输出:")
    print("=" * 80)

    processed_sample = dataset[0]

    print(f"\n返回的字段: {list(processed_sample.keys())}")

    # 展示各个字段
    input_ids = processed_sample['input_ids']
    loss_mask = processed_sample['loss_mask']
    position_ids = processed_sample['position_ids']

    print(f"\n[4.1] input_ids:")
    print(f"      shape: {input_ids.shape}")
    print(f"      dtype: {input_ids.dtype}")
    print(f"      序列长度: {len(input_ids)}")
    print(f"      前 50 个 token IDs: {input_ids[:50].tolist()}")

    print(f"\n[4.2] loss_mask (1=计算loss, 0=不计算):")
    print(f"      shape: {loss_mask.shape}")
    print(f"      dtype: {loss_mask.dtype}")
    print(f"      需要计算 loss 的 token 数: {loss_mask.sum().item()}")
    print(f"      不计算 loss 的 token 数: {(loss_mask == 0).sum().item()}")
    print(f"      loss 覆盖率: {loss_mask.sum().item() / len(loss_mask) * 100:.2f}%")
    print(f"      前 50 个 mask 值: {loss_mask[:50].tolist()}")

    print(f"\n[4.3] position_ids:")
    print(f"      shape: {position_ids.shape}")
    print(f"      dtype: {position_ids.dtype}")
    print(f"      前 50 个位置: {position_ids[:50].tolist()}")

    # 6. 解码展示
    print("\n" + "=" * 80)
    print("[5] 解码后的文本内容:")
    print("=" * 80)

    decoded_full = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"\n完整序列 (前 2000 字符):")
    print(decoded_full[:2000])
    if len(decoded_full) > 2000:
        print(f"\n... (总长度: {len(decoded_full)} 字符)")

    # 7. 分析 loss mask 的分布
    print("\n" + "=" * 80)
    print("[6] Loss Mask 分布分析:")
    print("=" * 80)

    # 找出所有需要计算 loss 的区间
    loss_regions = []
    in_loss_region = False
    start_idx = 0

    for i, mask_val in enumerate(loss_mask):
        if mask_val == 1 and not in_loss_region:
            start_idx = i
            in_loss_region = True
        elif mask_val == 0 and in_loss_region:
            loss_regions.append((start_idx, i))
            in_loss_region = False

    if in_loss_region:
        loss_regions.append((start_idx, len(loss_mask)))

    print(f"\n找到 {len(loss_regions)} 个需要计算 loss 的区间:")
    for idx, (start, end) in enumerate(loss_regions[:5]):  # 只显示前5个
        region_tokens = input_ids[start:end]
        region_text = tokenizer.decode(region_tokens, skip_special_tokens=False)
        print(f"\n  区间 {idx + 1}: [{start}:{end}] (长度 {end - start})")
        print(f"    文本预览: {region_text[:200]}...")

    if len(loss_regions) > 5:
        print(f"\n  ... 还有 {len(loss_regions) - 5} 个区间")

    # 8. 统计信息
    print("\n" + "=" * 80)
    print("[7] 统计信息:")
    print("=" * 80)
    print(f"\n  总 token 数: {len(input_ids)}")
    print(f"  训练 token 数 (loss_mask=1): {loss_mask.sum().item()}")
    print(f"  忽略 token 数 (loss_mask=0): {(loss_mask == 0).sum().item()}")
    print(f"  训练比例: {loss_mask.sum().item() / len(loss_mask) * 100:.2f}%")

    # 9. 验证 enable_thinking 的影响
    print("\n" + "=" * 80)
    print("[8] 验证 enable_thinking 参数:")
    print("=" * 80)
    print(f"\n  当前样本 enable_thinking: {first_row['enable_thinking']}")
    print(f"  检查解码文本中是否包含 thinking 标签:")
    print(f"    包含 '<thinking>': {'<thinking>' in decoded_full}")
    print(f"    包含 '</thinking>': {'</thinking>' in decoded_full}")
    print(f"    包含 '<think>': {'<think>' in decoded_full}")
    print(f"    包含 '</think>': {'</think>' in decoded_full}")

    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
