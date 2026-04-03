"""
评估脚本
========

支持两种评估模式（均通过 SGLang completions API）：
1. one_shot: 模型一次性生成代码
2. multi_turn: 模型多轮与环境交互

在 MBPP test 和 HumanEval 上评估 pass@1。

使用前需先启动 SGLang server:
  bash scripts/start_sglang.sh
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import load_mbpp, load_humaneval, load_apps, CodeProblem
from src.prompts import (
    SYSTEM_PROMPT_ONE_SHOT,
    build_one_shot_prompt,
    build_agentic_messages,
)
from src.env.code_env import CodeEnvironment
from src.env.sandbox import execute_with_tests
from src.env.tools import TOOLS_SCHEMA


def _strip_thinking(text: str) -> str:
    """剥离 Qwen3 的 <think>...</think> 块，返回剩余文本。"""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _parse_tool_call(text: str) -> tuple[str, dict | None]:
    """从模型输出中解析工具调用，只认 Qwen 标准格式。

    支持 Qwen3 的 <think> 块：先剥离再解析 <tool_call>。
    兼容 Qwen2.5（arguments 为 stringified JSON）和 Qwen3（arguments 为原生 dict）。

    Returns:
        (content, tool_call): content 是工具调用之前的文本（不含 think 块），
        tool_call 是 {"name": ..., "arguments": {...}} 或 None.
    """
    cleaned = _strip_thinking(text)
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", cleaned, re.DOTALL)
    if match:
        content = cleaned[: match.start()].strip()
        try:
            data = json.loads(match.group(1))
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if name and isinstance(arguments, dict):
                return content, {"name": name, "arguments": arguments}
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    return cleaned, None


def extract_code_from_completion(text: str) -> str:
    """从 one-shot completion 中提取代码.

    提取策略：
    0. 剥离 Qwen3 <think>...</think> 块
    1. 优先匹配 ```python ... ``` 代码块
    2. fallback：返回整段文本
    """
    text = _strip_thinking(text)
    pattern = re.compile(r"```python\s*\n?(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3].rstrip()
    return text.strip()


def _sglang_complete(
    client,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: list[str] | None = None,
) -> str:
    """调用 SGLang completions API 并返回生成文本。"""
    if stop is None:
        stop = ["<|im_end|>", "<|endoftext|>"]
    response = client.completions.create(
        model="default",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
    )
    return response.choices[0].text


def evaluate_one_shot(
    client,
    tokenizer,
    problems: list[CodeProblem],
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> dict:
    """One-shot 评估：通过 SGLang，模型一次性生成代码。"""
    passed = 0
    total = len(problems)
    results: list[dict] = []

    for i, prob in enumerate(problems):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_ONE_SHOT},
            {"role": "user", "content": build_one_shot_prompt(prob.prompt)},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        completion = _sglang_complete(client, prompt, max_new_tokens, temperature)
        completion = _strip_thinking(completion)
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
    client,
    tokenizer,
    problems: list[CodeProblem],
    max_turns: int = 5,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    timeout: int = 5,
    output_path: str | None = None,
    enable_thinking: bool = False,
) -> dict:
    """Multi-turn 评估：通过 SGLang completions API + apply_chat_template 注入 tools.

    流程：
    1. 用 tokenizer.apply_chat_template(messages, tools=TOOLS_SCHEMA) 构建完整 prompt
    2. 用 /v1/completions 端点做原始文本补全
    3. 手动解析 <tool_call> 标签

    除 pass@1 外，还收集 agentic 专属指标：
    - avg_turns: 平均交互轮数
    - fix_rate: 修复率（首次失败但最终通过的比例）
    """
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
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=TOOLS_SCHEMA,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )

            if i == 0 and _turn == 0:
                print(f"\n[DEBUG] full prompt:\n{prompt}")

            raw_text = _sglang_complete(client, prompt, max_new_tokens, temperature)
            assistant_text = _strip_thinking(raw_text)
            num_turns += 1

            content, tool_call = _parse_tool_call(assistant_text)

            if i == 0:
                print(f"[DEBUG] turn={_turn+1}, tool_call={'yes' if tool_call else 'no'}")
                print(f"[DEBUG] raw output:\n{assistant_text}")
                if tool_call:
                    print(f"[DEBUG] parsed tool_call={tool_call['name']}, args_len={len(json.dumps(tool_call['arguments']))}")

            if tool_call:
                messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["arguments"]),
                        },
                    }],
                })

                observation = env.execute_tool(tool_call["name"], **tool_call["arguments"])
                messages.append({"role": "tool", "content": observation})

                if first_test_passed is None and "tests passed" in observation:
                    first_test_passed = "All tests passed" in observation
            else:
                messages.append({"role": "assistant", "content": assistant_text})
                break

        is_correct = env.is_all_passed
        if is_correct:
            passed += 1

        if first_test_passed is False and is_correct:
            fix_count += 1
        if first_test_passed is True and is_correct:
            first_attempt_correct += 1

        all_turn_counts.append(num_turns)

        serialized_msgs = []
        for m in messages:
            entry = {"role": m["role"], "content": m.get("content", "")}
            if "tool_calls" in m:
                entry["tool_calls"] = m["tool_calls"]
            serialized_msgs.append(entry)

        results.append({
            "task_id": prob.task_id,
            "passed": is_correct,
            "num_turns": num_turns,
            "num_executions": len(env.test_results_history),
            "first_test_passed": first_test_passed,
            "final_code": env.current_code,
            "test_results_history": env.test_results_history,
            "messages": serialized_msgs,
        })

        if output_path:
            with open(output_path, "w") as f:
                json.dump({"results": results, "progress": f"{i+1}/{total}"}, f, indent=2, ensure_ascii=False)

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
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径（用于加载 tokenizer）")
    parser.add_argument("--datasets", nargs="+", default=["mbpp_test", "humaneval"],
                        choices=["mbpp_test", "mbpp_val", "humaneval", "apps_intro_test", "apps_intro_train"])
    parser.add_argument("--mode", type=str, default="one_shot",
                        choices=["one_shot", "multi_turn"])
    parser.add_argument("--output_dir", type=str, default="./outputs/eval")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="本地数据集目录（无外网环境使用），内含 mbpp_full/ humaneval/ 子目录")
    parser.add_argument("--sglang_url", type=str, default="http://localhost:30000/v1",
                        help="SGLang server 的 OpenAI 兼容 API 地址")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="启用 Qwen3 thinking mode（模型会先推理再回答）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from openai import OpenAI
    client = OpenAI(base_url=args.sglang_url, api_key="EMPTY")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Using SGLang backend at {args.sglang_url}")
    print(f"Tokenizer: {args.model}")

    all_results: dict[str, dict] = {}

    for ds_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating on {ds_name} (mode={args.mode})")
        print(f"{'='*60}")

        mbpp_local = os.path.join(args.data_dir, "mbpp_full") if args.data_dir else None
        humaneval_local = os.path.join(args.data_dir, "humaneval") if args.data_dir else None
        apps_local = os.path.join(args.data_dir, "apps", f"{ds_name.split('_')[-1]}.jsonl") if args.data_dir and ds_name.startswith("apps_") else None

        if ds_name == "mbpp_test":
            problems = load_mbpp(version="full", split="test", max_samples=args.max_samples, local_path=mbpp_local)
        elif ds_name == "mbpp_val":
            problems = load_mbpp(version="full", split="validation", max_samples=args.max_samples, local_path=mbpp_local)
        elif ds_name == "humaneval":
            problems = load_humaneval(max_samples=args.max_samples, local_path=humaneval_local)
        elif ds_name == "apps_intro_test":
            problems = load_apps(split="test", difficulty="introductory", max_samples=args.max_samples, local_path=apps_local)
        elif ds_name == "apps_intro_train":
            problems = load_apps(split="train", difficulty="introductory", max_samples=args.max_samples, local_path=apps_local)
        else:
            raise ValueError(f"Unknown dataset: {ds_name}")

        print(f"  {len(problems)} problems")

        if args.mode == "multi_turn":
            result = evaluate_multi_turn(
                client=client,
                tokenizer=tokenizer,
                problems=problems,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature or 0.7,
                output_path=str(output_dir / f"{ds_name}_{args.mode}_progress.json"),
                enable_thinking=args.enable_thinking,
            )
        else:
            result = evaluate_one_shot(
                client=client,
                tokenizer=tokenizer,
                problems=problems,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                enable_thinking=args.enable_thinking,
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
            "mode": args.mode,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
