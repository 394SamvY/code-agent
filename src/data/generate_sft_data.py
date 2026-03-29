"""
SFT 轨迹数据生成
==================

用强模型 API + CodeEnvironment 真实执行，生成多轮工具调用轨迹。
输出 verl MultiTurnSFTDataset 所需的 parquet 格式。

流程：
  强模型收到题目 (with tools)
    → 产出 tool_call (OpenAI 原生格式)
    → CodeEnvironment 真实执行
    → 把结果喂回强模型
    → 循环直到通过或达到 max_turns
    → 保存完整 messages 轨迹

用法:
  export OPENAI_API_KEY=sk-...
  python -m src.data.generate_sft_data --output data/sft/train.parquet
  python -m src.data.generate_sft_data --dataset humaneval --output data/sft/val.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import load_mbpp, load_humaneval, load_apps, CodeProblem
from src.env.code_env import CodeEnvironment
from src.env.tools import TOOLS_SCHEMA
from src.prompts import SYSTEM_PROMPT_AGENTIC_PLAIN, USER_PROMPT_TEMPLATE


def _call_api(client, model, messages, tools, temperature=0.7, max_tokens=1024):
    """封装 API 调用，统一错误处理。"""
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"  API error: {e}")
        return None


def _append_tool_call_and_execute(messages, msg, env):
    """处理 tool_call：追加 assistant message，执行工具，追加 tool response。

    Returns:
        (observation, num_executions_added)
    """
    assistant_msg = {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [],
    }
    for tc in msg.tool_calls:
        assistant_msg["tool_calls"].append({
            "type": "function",
            "id": tc.id,
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        })
    messages.append(assistant_msg)

    tc = msg.tool_calls[0]
    try:
        args = json.loads(tc.function.arguments)
    except json.JSONDecodeError:
        args = {"code": tc.function.arguments}

    observation = env.execute_tool(tc.function.name, **args)
    messages.append({
        "role": "tool",
        "content": observation,
        "tool_call_id": tc.id,
    })
    return observation, 1


def _save_parquet(records: list[dict], output_path: Path):
    """将轨迹数据保存为 parquet。

    输出两列：
    - messages: 完整多轮对话 [{role, content, tool_calls?, tool_call_id?}, ...]
                这是模型的实际交互记录（谁说了什么、调了什么工具）
    - tools:    工具 schema 描述 [{"type":"function", "function":{name, description, parameters}}]
                这不是调用记录，而是告诉模型"有哪些工具可用"的菜单
                apply_chat_template(messages, tools=tools) 会把它注入到 system prompt 中
                我们只有 execute_code 一个工具，所以每行内容相同（parquet 列压缩会去重）

    Parquet 嵌套 struct 会统一 schema，缺失字段补 None（如 system 消息会多出
    tool_calls=None）。经测试 Qwen 的 apply_chat_template 能正确处理 None 字段，
    不影响 tokenization 结果，因此直接用原生嵌套结构存储，和 verl 生态保持一致。
    """
    df = pd.DataFrame({
        "messages": [r["messages"] for r in records],
        "tools": [r["tools"] for r in records],
    })
    df.to_parquet(output_path, index=False)
    return df


def generate_trajectory(
    client: OpenAI,
    problem: CodeProblem,
    model: str = "gpt-5.1-2025-11-13",
    max_turns: int = 5,
    timeout: int = 5,
) -> dict | None:
    """用强模型为一道题生成多轮工具调用轨迹。

    Returns:
        dict with messages/tools/task_id/passed/num_turns/num_executions,
        或 None（API 错误时）。
    """
    env = CodeEnvironment(
        problem_description=problem.prompt,
        test_list=problem.test_list,
        entry_point=problem.entry_point,
        timeout=timeout,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_AGENTIC_PLAIN},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            problem_description=problem.prompt
        )},
    ]

    num_turns = 0
    num_executions = 0

    for _turn in range(max_turns):
        response = _call_api(client, model, messages, TOOLS_SCHEMA)
        if response is None:
            return None

        msg = response.choices[0].message
        num_turns += 1

        if msg.tool_calls:
            _, n = _append_tool_call_and_execute(messages, msg, env)
            num_executions += n

            if env.is_all_passed:
                # 全部通过，让模型输出总结
                final = _call_api(client, model, messages, TOOLS_SCHEMA,
                                  temperature=0.3, max_tokens=256)
                if final:
                    final_msg = final.choices[0].message
                    if not final_msg.tool_calls:
                        messages.append({
                            "role": "assistant",
                            "content": final_msg.content or "All tests passed.",
                        })
                        num_turns += 1
                else:
                    messages.append({
                        "role": "assistant",
                        "content": "All tests passed.",
                    })
                break
        else:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
            })
            break

    return {
        "messages": messages,
        "tools": TOOLS_SCHEMA,
        "task_id": problem.task_id,
        "passed": env.is_all_passed,
        "num_turns": num_turns,
        "num_executions": num_executions,
    }


def generate_dataset(
    problems: list[CodeProblem],
    model: str = "gpt-5.1-2025-11-13",
    max_turns: int = 5,
    output_path: str = "data/sft/train.parquet",
    progress_path: str | None = None,
    api_base: str | None = None,
    delay: float = 0.5,
) -> Path:
    """为一批题目生成 SFT 轨迹并保存为 parquet。"""
    client = OpenAI(
        base_url=api_base,
    )

    results = []
    stats = {"total": 0, "passed": 0, "failed": 0, "error": 0,
             "total_turns": 0, "total_executions": 0}

    for i, problem in enumerate(problems):
        stats["total"] += 1

        traj = generate_trajectory(
            client, problem, model=model, max_turns=max_turns,
        )

        if traj is None:
            stats["error"] += 1
            print(f"  [{i+1}/{len(problems)}] {problem.task_id}: ERROR")
            time.sleep(2)
            continue

        if traj["passed"]:
            stats["passed"] += 1
        else:
            stats["failed"] += 1

        stats["total_turns"] += traj["num_turns"]
        stats["total_executions"] += traj["num_executions"]

        results.append(traj)

        if (i + 1) % 10 == 0:
            n = stats["passed"] + stats["failed"]
            pass_rate = stats["passed"] / n if n > 0 else 0
            avg_turns = stats["total_turns"] / n if n > 0 else 0
            avg_exec = stats["total_executions"] / n if n > 0 else 0
            print(
                f"  [{i+1}/{len(problems)}] "
                f"pass={stats['passed']}/{n} ({pass_rate:.1%}), "
                f"avg_turns={avg_turns:.1f}, avg_exec={avg_exec:.1f}, "
                f"errors={stats['error']}"
            )

            # 保存进度
            if progress_path:
                _save_progress(results, progress_path, stats)

        if delay > 0:
            time.sleep(delay)

    # 保存最终结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 只保留最终通过且有工具调用的轨迹
    valid = [r for r in results if r["num_executions"] > 0 and r["passed"]]

    _save_parquet(valid, output_path)

    # 同时保存 JSON 格式，方便人工查看
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"task_id": r["task_id"], "passed": r["passed"],
              "num_turns": r["num_turns"], "num_executions": r["num_executions"],
              "messages": r["messages"]}
             for r in valid],
            f, indent=2, ensure_ascii=False,
        )

    # 打印统计
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"{'='*60}")
    print(f"  Total problems: {stats['total']}")
    print(f"  Passed: {stats['passed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Errors: {stats['error']}")
    n = stats["passed"] + stats["failed"]
    if n > 0:
        print(f"  Pass rate: {stats['passed']/n:.1%}")
        print(f"  Avg turns: {stats['total_turns']/n:.1f}")
        print(f"  Avg executions: {stats['total_executions']/n:.1f}")
    print(f"  Valid trajectories: {len(valid)}")
    print(f"  Saved to: {output_path}")
    print(f"  JSON:     {json_path}")

    # 保存详细统计
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            **stats,
            "valid_count": len(valid),
            "model": model,
            "max_turns": max_turns,
        }, f, indent=2)

    return output_path


def _save_progress(results: list, path: str, stats: dict):
    """保存生成进度，方便中断后查看。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "stats": stats,
            "count": len(results),
            "last_task_id": results[-1]["task_id"] if results else None,
        }, f, indent=2)


def rebuild_parquet_from_json(json_path: str, parquet_path: str | None = None):
    """从已有的 JSON 文件重新生成干净的 parquet（不需要调 API）。"""
    json_path = Path(json_path)
    if parquet_path is None:
        parquet_path = json_path.with_suffix(".parquet")
    else:
        parquet_path = Path(parquet_path)

    with open(json_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} trajectories from {json_path}")

    # 为 JSON 数据补上 tools 字段（JSON 里可能没有）
    for r in data:
        if "tools" not in r:
            r["tools"] = TOOLS_SCHEMA

    _save_parquet(data, parquet_path)
    print(f"Saved to {parquet_path} ({len(data)} rows)")
    return parquet_path


def main():
    parser = argparse.ArgumentParser(description="Generate SFT trajectories")
    sub = parser.add_subparsers(dest="command")

    # --- rebuild 子命令：从 JSON 重建干净 parquet ---
    p_rebuild = sub.add_parser("rebuild", help="Rebuild clean parquet from existing JSON")
    p_rebuild.add_argument("json_files", nargs="+", help="JSON file paths to rebuild")

    # --- generate 子命令（默认）---
    p_gen = sub.add_parser("generate", help="Generate trajectories via API")
    p_gen.add_argument("--model", default="gpt-5.1-2025-11-13")
    p_gen.add_argument("--dataset", default="mbpp_train",
                       choices=["mbpp_train", "mbpp_val", "humaneval", "apps"])
    p_gen.add_argument("--output", default="data/sft/train.parquet")
    p_gen.add_argument("--max_turns", type=int, default=5)
    p_gen.add_argument("--max_samples", type=int, default=None)
    p_gen.add_argument("--data_dir", type=str, default=None)
    p_gen.add_argument("--delay", type=float, default=0.5)
    p_gen.add_argument("--api_base", type=str,
                       default="http://thirdpart-proxy-prod.xaminim.com/v1/proxy/openai")

    args = parser.parse_args()

    if args.command == "rebuild":
        for json_file in args.json_files:
            rebuild_parquet_from_json(json_file)
        return

    # generate (default / 无子命令时兼容旧用法)
    if args.command is None:
        parser.print_help()
        print("\nHint: use 'rebuild' to regenerate parquet from JSON,")
        print("      or 'generate' to create new trajectories via API.")
        return

    mbpp_local = os.path.join(args.data_dir, "mbpp_full") if args.data_dir else None
    humaneval_local = os.path.join(args.data_dir, "humaneval") if args.data_dir else None

    if args.dataset == "mbpp_train":
        problems = load_mbpp(version="full", split="train",
                             max_samples=args.max_samples, local_path=mbpp_local)
    elif args.dataset == "mbpp_val":
        problems = load_mbpp(version="full", split="validation",
                             max_samples=args.max_samples, local_path=mbpp_local)
    elif args.dataset == "humaneval":
        problems = load_humaneval(max_samples=args.max_samples,
                                  local_path=humaneval_local)
    elif args.dataset == "apps":
        problems = load_apps(split="train", difficulty="introductory",
                             max_samples=args.max_samples or 500)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Loaded {len(problems)} problems from {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Max turns: {args.max_turns}")
    print(f"Output: {args.output}")
    print()

    progress_path = Path(args.output).with_suffix(".progress.json")

    generate_dataset(
        problems=problems,
        model=args.model,
        max_turns=args.max_turns,
        output_path=args.output,
        progress_path=str(progress_path),
        api_base=args.api_base,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
