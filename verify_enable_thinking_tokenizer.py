#!/usr/bin/env python3
"""
在服务器上运行此脚本，验证 enable_thinking 对 Qwen3 tokenizer 的实际影响。

用法：
    python3 verify_enable_thinking_tokenizer.py

目的：
    确认 enable_thinking=False 是否真的会删除 <think>...</think> 标签
"""

from transformers import AutoTokenizer

def main():
    print("=" * 80)
    print("验证 enable_thinking 对 Qwen3-8B tokenizer 的影响")
    print("=" * 80)

    # 1. 加载 tokenizer
    model_path = "/root/autodl-tmp/models/Qwen3-8B"
    print(f"\n1. 加载 tokenizer: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("   ✓ tokenizer 加载成功")
    except Exception as e:
        print(f"   ✗ 加载失败: {e}")
        return

    # 2. 准备测试消息（从实际 SFT 数据中提取的典型格式）
    message = {
        "role": "assistant",
        "content": "<think>The problem: The bear can choose a day d (1 ≤ d < n). On day d, he borrows a barrel of honey, sells it at price x_d. On day d+1, he buys a barrel at price x_{d+1} and returns it. His profit is x_d - x_{d+1} - c. We want to find the maximum profit.</think>\n\n<tool_call>{\"name\": \"run_public_tests\", \"arguments\": {\"code\": \"print('hello')\"}}</tool_call>"
    }

    print(f"\n2. 测试消息")
    print(f"   role: {message['role']}")
    print(f"   content 长度: {len(message['content'])} chars")
    print(f"   包含 <think>: True")
    print(f"   包含 </think>: True")
    print(f"   包含 <tool_call>: True")

    # 3. 场景 A: enable_thinking=False (当前训练配置)
    print(f"\n" + "=" * 80)
    print(f"场景 A: enable_thinking=False (当前 SFT 训练配置)")
    print("=" * 80)

    try:
        # 渲染为文本
        text_false = tokenizer.apply_chat_template(
            [message],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )
        print(f"\n渲染后的文本:")
        print(text_false)
        print()

        # tokenize
        tokens_false = tokenizer.apply_chat_template(
            [message],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False
        )
        decoded_false = tokenizer.decode(tokens_false)

        print(f"token 数量: {len(tokens_false)}")
        print(f"decode 后包含 '<think>': {'<think>' in decoded_false}")
        print(f"decode 后包含 '</think>': {'</think>' in decoded_false}")
        print(f"decode 后包含 '<tool_call>': {'<tool_call>' in decoded_false}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    # 4. 场景 B: enable_thinking=True (评测配置)
    print(f"\n" + "=" * 80)
    print(f"场景 B: enable_thinking=True (评测配置)")
    print("=" * 80)

    try:
        # 渲染为文本
        text_true = tokenizer.apply_chat_template(
            [message],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True
        )
        print(f"\n渲染后的文本:")
        print(text_true)
        print()

        # tokenize
        tokens_true = tokenizer.apply_chat_template(
            [message],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=True
        )
        decoded_true = tokenizer.decode(tokens_true)

        print(f"token 数量: {len(tokens_true)}")
        print(f"decode 后包含 '<think>': {'<think>' in decoded_true}")
        print(f"decode 后包含 '</think>': {'</think>' in decoded_true}")
        print(f"decode 后包含 '<tool_call>': {'<tool_call>' in decoded_true}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    # 5. 对比
    print(f"\n" + "=" * 80)
    print(f"对比结果")
    print("=" * 80)

    try:
        print(f"enable_thinking=False: {len(tokens_false)} tokens")
        print(f"enable_thinking=True:  {len(tokens_true)} tokens")
        print(f"差异: {len(tokens_true) - len(tokens_false)} tokens")

        if '<think>' not in decoded_false and '<think>' in decoded_true:
            print(f"\n✓ 确认：enable_thinking=False 会删除 <think> 标签")
            print(f"\n结论：")
            print(f"  - 训练时 enable_thinking=False → 模型从未见过 <think>")
            print(f"  - 评测时 enable_thinking=True → 模型不认识 <think>")
            print(f"  - 这就是模型无法正确调用工具的根本原因")
            print(f"\n修复方案：")
            print(f"  修改 scripts/train_sft_with_verl.sh，添加：")
            print(f"    data.apply_chat_template_kwargs.enable_thinking=true")
        elif '<think>' in decoded_false and '<think>' in decoded_true:
            print(f"\n✗ 意外：enable_thinking=False 并未删除 <think> 标签")
            print(f"\n需要重新排查其他原因")
        else:
            print(f"\n? 未知情况，需要进一步分析")

    except Exception as e:
        print(f"对比失败: {e}")

if __name__ == '__main__':
    main()
