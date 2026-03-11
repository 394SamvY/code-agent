"""
评估脚本
========

支持三种评估模式：
1. one_shot: 模型一次性生成代码
2. multi_turn: 模型多轮与环境交互（Qwen 原生 tool calling）
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
from src.agent.prompts import (
    SYSTEM_PROMPT_ONE_SHOT,
    build_one_shot_prompt,
    build_agentic_messages,
)
from src.agent.parser import parse_first_tool_call, extract_code_from_completion
from src.env.code_env import CodeEnvironment
from src.env.sandbox import execute_with_tests
from src.env.tools import TOOLS_SCHEMA


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
    model,
    tokenizer,
    problems: list[CodeProblem],
    max_turns: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    timeout: int = 5,
    few_shot: bool = False,
) -> dict:
    """Multi-turn 评估：Qwen 原生 tool calling 格式.

    除 pass@1 外，还收集 agentic 专属指标：
    - avg_turns: 平均交互轮数
    - fix_rate: 修复率（初始代码错误但最终通过的比例）
    - tool_distribution: 工具调用频次分布
    - workflow_patterns: 工具调用序列模式统计
    """
    passed = 0
    total = len(problems)
    results: list[dict] = []

    # Agentic 指标收集
    all_turn_counts: list[int] = []
    all_tool_sequences: list[list[str]] = []
    fix_count = 0  # 初始错误但最终修复的数量
    first_attempt_correct = 0  # 第一次 write_code 就正确的数量

    for i, prob in enumerate(problems):
        env = CodeEnvironment(
            problem_description=prob.prompt,
            test_list=prob.test_list,
            entry_point=prob.entry_point,
            timeout=timeout,
        )

        messages = build_agentic_messages(prob.prompt, few_shot=few_shot)
        num_turns = 0
        tool_sequence: list[str] = []
        raw_responses: list[str] = []  # 保存模型原始输出
        first_write_tested = False  # 第一次 write+test 的结果
        first_test_passed = None

        for _turn in range(max_turns):
            input_ids = tokenizer.apply_chat_template(
                messages,
                tools=TOOLS_SCHEMA,
                add_generation_prompt=True,
                return_tensors="pt",
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
            response = tokenizer.decode(new_ids, skip_special_tokens=True)
            raw_responses.append(response[:1000])  # 保存前 1000 字符用于诊断
            messages.append({"role": "assistant", "content": response})
            num_turns += 1

            tool_call = parse_first_tool_call(response)
            if tool_call is None:
                break

            tool_sequence.append(tool_call.name)
            observation = env.execute_tool(tool_call.name, **tool_call.arguments)
            messages.append({"role": "tool", "content": observation})

            # 追踪第一次 run_tests 的结果（用于计算修复率）
            if tool_call.name == "run_tests" and first_test_passed is None:
                first_test_passed = "All tests passed" in observation

            if tool_call.name == "submit" or env.is_done:
                break

        if not env.is_done:
            env.execute_tool("submit")

        is_correct = env.final_reward > 0.5
        if is_correct:
            passed += 1

        # 修复率统计：第一次测试失败但最终通过
        if first_test_passed is False and is_correct:
            fix_count += 1
        if first_test_passed is True and is_correct:
            first_attempt_correct += 1

        all_turn_counts.append(num_turns)
        all_tool_sequences.append(tool_sequence)

        results.append({
            "task_id": prob.task_id,
            "passed": is_correct,
            "num_turns": num_turns,
            "tool_sequence": tool_sequence,
            "first_test_passed": first_test_passed,
            "raw_responses": raw_responses,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}] pass@1 so far: {passed}/{i+1} = {passed/(i+1):.3f}")

    # ---- 汇总 agentic 指标 ----
    avg_turns = sum(all_turn_counts) / total if total > 0 else 0.0

    # 工具调用分布
    tool_counter: Counter = Counter()
    for seq in all_tool_sequences:
        tool_counter.update(seq)

    # 工作流模式统计（将工具序列归类）
    pattern_counter: Counter = Counter()
    for seq in all_tool_sequences:
        pattern = " -> ".join(seq) if seq else "(no tools)"
        pattern_counter[pattern] += 1

    # 修复率：在有过测试的题目中，初始失败但最终修复的比例
    tested_count = sum(1 for r in results if r["first_test_passed"] is not None)
    failed_first = sum(1 for r in results if r["first_test_passed"] is False)
    fix_rate = fix_count / failed_first if failed_first > 0 else 0.0

    agentic_metrics = {
        "avg_turns": round(avg_turns, 2),
        "fix_rate": round(fix_rate, 4),
        "fix_count": fix_count,
        "first_attempt_correct": first_attempt_correct,
        "tested_problems": tested_count,
        "tool_distribution": dict(tool_counter.most_common()),
        "top_workflow_patterns": dict(pattern_counter.most_common(10)),
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
    parser.add_argument("--few_shot", action="store_true",
                        help="multi_turn 模式下是否使用 few-shot 示例")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    lora = args.lora_path if args.mode != "baseline" else None
    model, tokenizer = load_model_and_tokenizer(args.model, lora, args.device)

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
                model, tokenizer, problems,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature or 0.7,
                few_shot=args.few_shot,
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
            print(f"  avg_turns = {am['avg_turns']}, fix_rate = {am['fix_rate']:.4f} "
                  f"({am['fix_count']} fixes / {am['fix_count'] + am['first_attempt_correct']} tested)")
            print(f"  tool_distribution: {am['tool_distribution']}")
            top_patterns = list(am["top_workflow_patterns"].items())[:5]
            for pattern, count in top_patterns:
                print(f"    {pattern}: {count}")

        detail_path = output_dir / f"{ds_name}_{args.mode}_details.json"
        with open(detail_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for ds_name, r in all_results.items():
        print(f"  {ds_name}: pass@1 = {r['pass@1']:.4f} ({r['passed']}/{r['total']})")

    summary_path = output_dir / f"summary_{args.mode}.json"
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
