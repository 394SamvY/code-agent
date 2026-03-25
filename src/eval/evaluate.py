"""
评估脚本
========

支持三种评估模式：
1. one_shot: 模型一次性生成代码（本地 HuggingFace）
2. multi_turn: 模型多轮与环境交互（通过 SGLang OpenAI API，与训练完全一致）
3. baseline: 等同于 one_shot（裸模型，不加载 LoRA）

在 MBPP test 和 HumanEval 上评估 pass@1。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import Counter

from src.data.dataset import load_mbpp, load_humaneval, CodeProblem
from src.prompts import (
    SYSTEM_PROMPT_ONE_SHOT,
    build_one_shot_prompt,
    build_agentic_messages,
)
from src.env.code_env import CodeEnvironment
from src.env.sandbox import execute_with_tests
from src.env.tools import TOOLS_SCHEMA


def extract_code_from_completion(text: str) -> str:
    """从 one-shot completion 中提取代码.

    提取策略：
    1. 优先匹配 ```python ... ``` 代码块
    2. fallback：返回整段文本
    """
    import re

    pattern = re.compile(r"```python\s*\n?(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3].rstrip()
    return text.strip()


def load_model_and_tokenizer(
    model_name: str,
    lora_path: str | None = None,
    device: str = "auto",
):
    """加载模型和 tokenizer，可选加载 LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device == "auto":
        if torch.cuda.is_available():
            device_map = "auto"
        else:
            device_map = "cpu"
    else:
        device_map = device

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device_map == "cpu" else torch.bfloat16,
        device_map=device_map,
    )

    if lora_path:
        print(f"Loading LoRA from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def evaluate_one_shot(
    model,
    tokenizer,
    problems: list[CodeProblem],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    """One-shot 评估：模型一次性生成代码."""
    passed = 0
    total = len(problems)
    results: list[dict] = []

    for i, prob in enumerate(problems):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_ONE_SHOT},
            {"role": "user", "content": build_one_shot_prompt(prob.prompt)},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
        input_ids = input_ids.to(model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = model.generate(input_ids, **gen_kwargs)

        new_ids = outputs[0][input_ids.shape[1]:]
        completion = tokenizer.decode(new_ids, skip_special_tokens=True)
        code = extract_code_from_completion(completion)

        is_correct = _run_tests(code, prob.test_list)
        if is_correct:
            passed += 1

        results.append({
            "task_id": prob.task_id,
            "passed": is_correct,
            "code": code[:500],
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}] pass@1 so far: {passed}/{i+1} = {passed/(i+1):.3f}")

    return {
        "pass@1": passed / total if total > 0 else 0.0,
        "passed": passed,
        "total": total,
        "results": results,
    }


def evaluate_multi_turn(
    sglang_url: str,
    problems: list[CodeProblem],
    max_turns: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    timeout: int = 5,
) -> dict:
    """Multi-turn 评估：通过 SGLang OpenAI 兼容 API，与训练使用相同的推理+解析栈。

    除 pass@1 外，还收集 agentic 专属指标：
    - avg_turns: 平均交互轮数
    - fix_rate: 修复率（首次失败但最终通过的比例）
    """
    from openai import OpenAI

    client = OpenAI(base_url=sglang_url, api_key="EMPTY")

    passed = 0
    total = len(problems)
    results: list[dict] = []

    all_turn_counts: list[int] = []
    fix_count = 0
    first_attempt_correct = 0

    for i, prob in enumerate(problems):
        env = CodeEnvironment(
            problem_description=prob.prompt,
            test_list=prob.test_list,
            entry_point=prob.entry_point,
            timeout=timeout,
        )

        messages = build_agentic_messages(prob.prompt)
        num_turns = 0
        first_test_passed = None

        for _turn in range(max_turns):
            response = client.chat.completions.create(
                model="default",
                messages=messages,
                tools=TOOLS_SCHEMA,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

            choice = response.choices[0]
            assistant_msg = choice.message
            num_turns += 1

            msg_dict: dict = {"role": "assistant", "content": assistant_msg.content or ""}
            if assistant_msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in assistant_msg.tool_calls
                ]
            messages.append(msg_dict)

            if not assistant_msg.tool_calls:
                break

            tc = assistant_msg.tool_calls[0]
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            observation = env.execute_tool(tool_name, **tool_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": observation,
            })

            # 追踪首次测试结果（用于 fix_rate 计算）
            if first_test_passed is None and "tests passed" in observation:
                first_test_passed = "All tests passed" in observation

        is_correct = env.is_all_passed
        if is_correct:
            passed += 1

        if first_test_passed is False and is_correct:
            fix_count += 1
        if first_test_passed is True and is_correct:
            first_attempt_correct += 1

        all_turn_counts.append(num_turns)

        results.append({
            "task_id": prob.task_id,
            "passed": is_correct,
            "num_turns": num_turns,
            "num_executions": len(env.test_results_history),
            "first_test_passed": first_test_passed,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}] pass@1 so far: {passed}/{i+1} = {passed/(i+1):.3f}")

    avg_turns = sum(all_turn_counts) / total if total > 0 else 0.0

    failed_first = sum(1 for r in results if r["first_test_passed"] is False)
    fix_rate = fix_count / failed_first if failed_first > 0 else 0.0

    agentic_metrics = {
        "avg_turns": round(avg_turns, 2),
        "avg_executions": round(sum(r["num_executions"] for r in results) / total, 2) if total > 0 else 0.0,
        "fix_rate": round(fix_rate, 4),
        "fix_count": fix_count,
        "first_attempt_correct": first_attempt_correct,
    }

    return {
        "pass@1": passed / total if total > 0 else 0.0,
        "passed": passed,
        "total": total,
        "agentic_metrics": agentic_metrics,
        "results": results,
    }


def _run_tests(code: str, test_list: list[str], timeout: int = 5) -> bool:
    """执行测试用例，全部通过返回 True."""
    for test in test_list:
        result = execute_with_tests(code, test, timeout=timeout)
        if not result.success:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Evaluate code agent")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=["mbpp_test", "humaneval"],
                        choices=["mbpp_test", "mbpp_val", "humaneval"])
    parser.add_argument("--mode", type=str, default="one_shot",
                        choices=["one_shot", "multi_turn", "baseline"])
    parser.add_argument("--output_dir", type=str, default="./outputs/eval")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="本地数据集目录（无外网环境使用），内含 mbpp_full/ humaneval/ 子目录")
    parser.add_argument("--sglang_url", type=str, default="http://localhost:30000/v1",
                        help="SGLang server 的 OpenAI 兼容 API 地址（multi_turn 模式使用）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # multi_turn 通过 SGLang API，不需要本地加载模型
    model, tokenizer = None, None
    if args.mode != "multi_turn":
        print(f"Loading model: {args.model}")
        lora = args.lora_path if args.mode != "baseline" else None
        model, tokenizer = load_model_and_tokenizer(args.model, lora, args.device)
    else:
        print(f"Using SGLang backend at {args.sglang_url}")

    all_results: dict[str, dict] = {}

    for ds_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating on {ds_name} (mode={args.mode})")
        print(f"{'='*60}")

        mbpp_local = os.path.join(args.data_dir, "mbpp_full") if args.data_dir else None
        humaneval_local = os.path.join(args.data_dir, "humaneval") if args.data_dir else None

        if ds_name == "mbpp_test":
            problems = load_mbpp(version="full", split="test", max_samples=args.max_samples, local_path=mbpp_local)
        elif ds_name == "mbpp_val":
            problems = load_mbpp(version="full", split="validation", max_samples=args.max_samples, local_path=mbpp_local)
        elif ds_name == "humaneval":
            problems = load_humaneval(max_samples=args.max_samples, local_path=humaneval_local)
        else:
            raise ValueError(f"Unknown dataset: {ds_name}")

        print(f"  {len(problems)} problems")

        if args.mode == "multi_turn":
            result = evaluate_multi_turn(
                sglang_url=args.sglang_url,
                problems=problems,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature or 0.7,
            )
        else:
            result = evaluate_one_shot(
                model, tokenizer, problems,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

        all_results[ds_name] = {
            "pass@1": result["pass@1"],
            "passed": result["passed"],
            "total": result["total"],
        }
        if "agentic_metrics" in result:
            all_results[ds_name]["agentic_metrics"] = result["agentic_metrics"]

        print(f"\n  {ds_name} pass@1 = {result['pass@1']:.4f} ({result['passed']}/{result['total']})")

        if "agentic_metrics" in result:
            am = result["agentic_metrics"]
            print(f"  avg_turns = {am['avg_turns']}, avg_executions = {am['avg_executions']}")
            print(f"  fix_rate = {am['fix_rate']:.4f} ({am['fix_count']} fixes)")

        detail_path = output_dir / f"{ds_name}_{args.mode}_details.json"
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(detail_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for ds_name, r in all_results.items():
        print(f"  {ds_name}: pass@1 = {r['pass@1']:.4f} ({r['passed']}/{r['total']})")

    summary_path = output_dir / f"summary_{args.mode}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "lora_path": args.lora_path,
            "mode": args.mode,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
