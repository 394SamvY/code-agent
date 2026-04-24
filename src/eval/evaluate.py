"""
OJ-like evaluation entrypoint.

Supports one-shot code generation and multi-turn tool use on the unified
CodeProblem schema.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import CodeProblem, load_codecontests, load_livecodebench
from src.env.code_env import CodeEnvironment
from src.env.tools import (
    TERMINAL_VERDICTS,
    TOOLS_SCHEMA,
    VERDICT_ACCEPTED,
    VERDICT_SUBMISSION_LIMIT_EXCEEDED,
)
from src.prompts import (
    SYSTEM_PROMPT_ONE_SHOT,
    build_agentic_messages,
    build_one_shot_prompt,
)


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _parse_tool_call(text: str) -> tuple[str, dict[str, Any] | None]:
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
        except (json.JSONDecodeError, TypeError):
            pass
    return cleaned, None


def extract_code_from_completion(text: str) -> str:
    text = _strip_thinking(text)
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", text, flags=re.DOTALL)
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


def _serialize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        item = {"role": message["role"], "content": message.get("content", "")}
        if "tool_calls" in message:
            item["tool_calls"] = message["tool_calls"]
        serialized.append(item)
    return serialized


def _submit_code(
    problem: CodeProblem,
    code: str,
    timeout: int | float | None = None,
    max_submissions: int = 5,
) -> tuple[CodeEnvironment, dict[str, Any]]:
    env = CodeEnvironment(problem, timeout=timeout, max_submissions=max_submissions)
    result = env.submit_solution(code)
    return env, result


def evaluate_one_shot(
    client,
    tokenizer,
    problems: list[CodeProblem],
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: int | float | None = None,
    max_submissions: int = 5,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    passed = 0
    total = len(problems)
    results: list[dict[str, Any]] = []

    for index, problem in enumerate(problems):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_ONE_SHOT},
            {"role": "user", "content": build_one_shot_prompt(problem)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        completion = _sglang_complete(client, prompt, max_new_tokens, temperature)
        code = extract_code_from_completion(completion)
        env, submission = _submit_code(
            problem,
            code,
            timeout=timeout,
            max_submissions=max_submissions,
        )
        accepted = submission["verdict"] == VERDICT_ACCEPTED
        if accepted:
            passed += 1

        results.append(
            {
                "task_id": problem.task_id,
                "dataset": problem.dataset,
                "passed": accepted,
                "verdict": submission["verdict"],
                "final_code": code,
                "submission_history": env.submission_history,
                "public_results_history": env.public_results_history,
                "messages": _serialize_messages(messages),
            }
        )

        if (index + 1) % 10 == 0:
            print(f"  [{index + 1}/{total}] pass@1 so far: {passed}/{index + 1} = {passed/(index + 1):.3f}")

    return {
        "pass@1": passed / total if total else 0.0,
        "passed": passed,
        "total": total,
        "results": results,
    }


def _append_tool_call_message(
    messages: list[dict[str, Any]],
    content: str,
    tool_call: dict[str, Any],
) -> None:
    messages.append(
        {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["arguments"]),
                    },
                }
            ],
        }
    )


def evaluate_multi_turn(
    client,
    tokenizer,
    problems: list[CodeProblem],
    max_tool_calls: int = 32,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    timeout: int | float | None = None,
    max_submissions: int = 5,
    output_path: str | None = None,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    passed = 0
    total = len(problems)
    results: list[dict[str, Any]] = []
    tool_call_counts: list[int] = []

    for index, problem in enumerate(problems):
        env = CodeEnvironment(problem, timeout=timeout, max_submissions=max_submissions)
        messages = build_agentic_messages(problem)
        fallback_submitted = False

        for tool_call_count in range(max_tool_calls):
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=TOOLS_SCHEMA,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            raw_text = _sglang_complete(client, prompt, max_new_tokens, temperature)
            assistant_text = _strip_thinking(raw_text)
            content, tool_call = _parse_tool_call(assistant_text)

            if tool_call is None:
                messages.append({"role": "assistant", "content": assistant_text})
                code = extract_code_from_completion(assistant_text)
                if code:
                    env.submit_solution(code)
                    fallback_submitted = True
                break

            _append_tool_call_message(messages, content, tool_call)
            observation = env.execute_tool(tool_call["name"], **tool_call["arguments"])
            messages.append({"role": "tool", "content": observation})

            if env.last_submission_verdict in TERMINAL_VERDICTS:
                break
        else:
            tool_call_count = max_tool_calls - 1

        accepted = env.is_accepted
        if accepted:
            passed += 1

        tool_call_counts.append(tool_call_count + 1 if max_tool_calls else 0)
        results.append(
            {
                "task_id": problem.task_id,
                "dataset": problem.dataset,
                "passed": accepted,
                "verdict": env.last_submission_verdict,
                "num_tool_calls": len(env.tool_history),
                "fallback_submitted": fallback_submitted,
                "final_code": env.current_code,
                "submission_history": env.submission_history,
                "public_results_history": env.public_results_history,
                "messages": _serialize_messages(messages),
            }
        )

        if output_path:
            with open(output_path, "w") as f:
                json.dump({"results": results, "progress": f"{index + 1}/{total}"}, f, indent=2, ensure_ascii=False)

        if (index + 1) % 10 == 0:
            print(f"  [{index + 1}/{total}] pass@1 so far: {passed}/{index + 1} = {passed/(index + 1):.3f}")

    avg_tool_calls = sum(tool_call_counts) / total if total else 0.0
    return {
        "pass@1": passed / total if total else 0.0,
        "passed": passed,
        "total": total,
        "agentic_metrics": {
            "avg_tool_calls": round(avg_tool_calls, 2),
            "max_tool_calls": max_tool_calls,
            "max_submissions": max_submissions,
        },
        "results": results,
    }


def _local_path(data_dir: str | None, dataset_name: str) -> str | None:
    if data_dir is None:
        return None
    candidate = Path(data_dir) / dataset_name
    return str(candidate) if candidate.exists() else None


def load_eval_dataset(
    dataset_name: str,
    max_samples: int | None = None,
    data_dir: str | None = None,
    codecontests_split: str = "valid",
    livecodebench_version_tag: str = "release_v6",
) -> list[CodeProblem]:
    if dataset_name == "livecodebench":
        return load_livecodebench(
            max_samples=max_samples,
            local_path=_local_path(data_dir, "livecodebench"),
            version_tag=livecodebench_version_tag,
        )
    if dataset_name == "codecontests":
        return load_codecontests(
            split=codecontests_split,  # type: ignore[arg-type]
            max_samples=max_samples,
            local_path=_local_path(data_dir, "codecontests"),
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OJ-like code agent")
    parser.add_argument("--model", type=str, required=True, help="Tokenizer/model path")
    parser.add_argument("--datasets", nargs="+", default=["livecodebench"], choices=["livecodebench", "codecontests"])
    parser.add_argument("--mode", type=str, default="multi_turn", choices=["one_shot", "multi_turn"])
    parser.add_argument("--output_dir", type=str, default="./outputs/eval")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_tool_calls", type=int, default=None)
    parser.add_argument("--max_turns", type=int, default=None, help="Deprecated alias for --max_tool_calls")
    parser.add_argument("--max_submissions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--codecontests_split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--livecodebench_version_tag", type=str, default="release_v6")
    parser.add_argument("--sglang_url", type=str, default="http://localhost:30000/v1")
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    args = parser.parse_args()

    from openai import OpenAI
    from transformers import AutoTokenizer

    max_tool_calls = args.max_tool_calls if args.max_tool_calls is not None else args.max_turns
    if max_tool_calls is None:
        max_tool_calls = 32
    temperature = args.temperature
    if temperature is None:
        temperature = 0.0 if args.mode == "one_shot" else 0.7

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.sglang_url, api_key="EMPTY")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Using SGLang backend at {args.sglang_url}")
    print(f"Tokenizer: {args.model}")
    print(f"Mode: {args.mode}, max_tool_calls={max_tool_calls}, max_submissions={args.max_submissions}")

    all_results: dict[str, dict[str, Any]] = {}
    for dataset_name in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {dataset_name} (mode={args.mode})")
        print(f"{'=' * 60}")
        problems = load_eval_dataset(
            dataset_name,
            max_samples=args.max_samples,
            data_dir=args.data_dir,
            codecontests_split=args.codecontests_split,
            livecodebench_version_tag=args.livecodebench_version_tag,
        )
        print(f"  {len(problems)} problems")

        if args.mode == "one_shot":
            result = evaluate_one_shot(
                client=client,
                tokenizer=tokenizer,
                problems=problems,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                timeout=args.timeout,
                max_submissions=args.max_submissions,
                enable_thinking=args.enable_thinking,
            )
        else:
            result = evaluate_multi_turn(
                client=client,
                tokenizer=tokenizer,
                problems=problems,
                max_tool_calls=max_tool_calls,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                timeout=args.timeout,
                max_submissions=args.max_submissions,
                output_path=str(output_dir / f"{dataset_name}_{args.mode}_progress.json"),
                enable_thinking=args.enable_thinking,
            )

        all_results[dataset_name] = {
            "pass@1": result["pass@1"],
            "passed": result["passed"],
            "total": result["total"],
        }
        if "agentic_metrics" in result:
            all_results[dataset_name]["agentic_metrics"] = result["agentic_metrics"]

        print(f"\n  {dataset_name} pass@1 = {result['pass@1']:.4f} ({result['passed']}/{result['total']})")

        detail_path = output_dir / f"{dataset_name}_{args.mode}_details.json"
        with open(detail_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    summary_path = output_dir / f"summary_{args.mode}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "mode": args.mode,
                "results": all_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
