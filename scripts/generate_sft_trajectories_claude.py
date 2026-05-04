"""用 Claude (Anthropic API) 生成 SFT 训练 trajectory。

对 CodeContests train 中的每道题，用 Claude 在 OJ-like 两工具协议下交互解题。
输出 JSONL 格式与 sft_trajectories.jsonl 一致，
包含 messages/verdict/task_id/turns/submissions/public_calls/title 字段。

用法：
  python3 scripts/generate_sft_trajectories_claude.py \
      --input data/verl/codecontests_train.parquet \
      --output data/verl/sft_trajectories_claude.jsonl \
      --max-samples 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import anthropic

# ── API client ────────────────────────────────────────


ANTHROPIC_BASE_URL = "https://cc-vibe.com"
ANTHROPIC_AUTH_TOKEN = "sk-ad7a4ef36b65a40d3b7c87a6c3d114200dd7315d34d9952e9863d1fe1f61b4b7"


def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=ANTHROPIC_BASE_URL, api_key=ANTHROPIC_AUTH_TOKEN)


# ── Judge ─────────────────────────────────────────────
from src.env.tools import (
    OJTestCase,
    VERDICT_NO_TESTS,
    format_judge_observation,
    parse_oj_tests,
    run_oj_judge,
)

# ── Tool schemas（Anthropic 格式）─────────────────────
TOOLS_ANTHROPIC = [
    {
        "name": "run_public_tests",
        "description": (
            "Run your code against public test cases. "
            "Use this to check if your solution passes the example tests. "
            "Returns per-test pass/fail results. Stops at the first failure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python program reading from stdin and writing to stdout.",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "submit_solution",
        "description": (
            "Submit your final solution for full judging. "
            "You have at most 5 submissions total. "
            "Only call this when public tests pass."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python program reading from stdin and writing to stdout.",
                }
            },
            "required": ["code"],
        },
    },
]

# 提示词追加：要求工具调用 + 每步先思考
TOOL_CALL_REQUIRED_PROMPT = (
    "\n\n# Important\n"
    "You MUST call a tool in every response. "
    "Start by calling run_public_tests to test your solution. "
    "When all public tests pass, call submit_solution.\n\n"
    "# Thinking Required\n"
    "Before EVERY tool call, you MUST first analyze the problem, your approach, "
    "any feedback from previous tests, and what you plan to do next. "
    "Output your reasoning inside <think>...</think> tags before each tool call. "
    "Do this for EVERY turn, even if you are just fixing a small bug or resubmitting."
)

MAX_SUBMISSIONS = 5
MAX_PUBLIC_CALLS = 15
MAX_ASSISTANT_TURNS = 20

DEFAULT_MAX_TOKENS = 16384
DEFAULT_THINKING_BUDGET = 2048  # Claude extended thinking token 预算


def call_claude(
    client: anthropic.Anthropic,
    system_prompt: str,
    messages: list[dict],
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> dict | None:
    """调 Claude API，返回兼容格式的 dict。

    Returns:
        dict with: reasoning_content (str), content (str), tool_calls (list or None)
        None if API error after retries
    """
    for attempt in range(3):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "tools": TOOLS_ANTHROPIC,
                "max_tokens": max_tokens,
                "timeout": 120,
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            # extended thinking: 只有非 thinking 模型不需要，这里默认开启
            if thinking_budget > 0:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

            resp = client.messages.create(**kwargs)

            # 解析 Anthropic 响应为统一格式
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[dict] = []

            for block in resp.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "thinking":
                    thinking_parts.append(block.thinking)
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input, ensure_ascii=False),
                        },
                    })

            return {
                "content": "\n".join(text_parts) if text_parts else "",
                "reasoning_content": "\n".join(thinking_parts) if thinking_parts else "",
                "tool_calls": tool_calls if tool_calls else None,
            }
        except Exception as e:
            print(f"  [API retry {attempt + 1}/3] {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return None


def parse_tool_call(msg: dict) -> tuple[str | None, dict | None, str | None]:
    """从统一格式的 assistant message 中提取 tool call。"""
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        return None, None, None
    tc = tool_calls[0]
    func = tc.get("function", {})
    name = func.get("name")
    call_id = tc.get("id")
    try:
        args = json.loads(func.get("arguments", "{}"))
    except json.JSONDecodeError:
        args = {}
    return name, args, call_id


def run_problem(
    client: anthropic.Anthropic,
    model: str,
    row: pd.Series,
    index: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> dict | None:
    """对单道题跑 multi-turn agent loop，返回结果 dict（API 错误时返回 None）。"""
    prompt_str = row["prompt"]
    messages = json.loads(prompt_str) if isinstance(prompt_str, str) else list(prompt_str)

    # 解析测试用例
    extra = json.loads(row["extra_info"]) if isinstance(row["extra_info"], str) else row["extra_info"]
    tools_kwargs = extra.get("tools_kwargs", {})
    public_tests = parse_oj_tests(
        tools_kwargs.get("run_public_tests", {}).get("create_kwargs", {}).get("public_tests", [])
    )
    private_tests = parse_oj_tests(
        tools_kwargs.get("submit_solution", {}).get("create_kwargs", {}).get("private_tests", [])
    )

    title = ""
    for m in messages:
        if m.get("role") == "user":
            for line in m.get("content", "").split("\n"):
                if line.startswith("Title:"):
                    title = line
            break

    # ── 分离 system prompt 和消息 ──
    system_prompt = ""
    api_messages: list[dict] = []  # Anthropic 格式
    sft_messages: list[dict] = []  # 输出用，文本 <tool_call> 格式

    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        else:
            api_messages.append({"role": m["role"], "content": m["content"]})
        sft_messages.append({"role": m["role"], "content": m["content"]})

    # 在 system prompt 末尾追加工具调用要求
    system_prompt += TOOL_CALL_REQUIRED_PROMPT

    submissions = 0
    public_calls = 0

    for turn in range(MAX_ASSISTANT_TURNS):
        msg = call_claude(client, system_prompt, api_messages, model,
                          max_tokens=max_tokens, thinking_budget=thinking_budget)
        if msg is None:
            return None  # API 错误

        reasoning = msg.get("reasoning_content", "") or ""
        content = msg.get("content", "") or ""
        tool_name, tool_args, tool_call_id = parse_tool_call(msg)

        # ── 构建 Anthropic assistant 消息块 ──
        anthropic_blocks: list[dict] = []
        if reasoning:
            anthropic_blocks.append({"type": "thinking", "thinking": reasoning})
        if content:
            anthropic_blocks.append({"type": "text", "text": content})
        if tool_name:
            anthropic_blocks.append({
                "type": "tool_use",
                "id": tool_call_id,
                "name": tool_name,
                "input": tool_args,
            })

        # ── 构建 SFT assistant 消息 ──
        if not tool_name:
            # 没有 tool call，可能是纯文本回复
            sft_text = content
            if reasoning:
                sft_text = "<think>" + reasoning + "</think>\n\n" + content
            sft_messages.append({"role": "assistant", "content": sft_text})
            return {
                "messages": sft_messages,
                "verdict": "no_tool_call",
                "task_id": extra.get("task_id", f"sample_{index}"),
                "turns": turn + 1,
                "submissions": submissions,
                "public_calls": public_calls,
                "title": title,
            }

        tool_call_json = json.dumps({"name": tool_name, "arguments": tool_args}, ensure_ascii=False)
        if reasoning:
            sft_text = "<think>" + reasoning + "</think>\n\n<tool_call>" + tool_call_json + "</tool_call>"
        else:
            sft_text = "<tool_call>" + tool_call_json + "</tool_call>"
        sft_messages.append({"role": "assistant", "content": sft_text})

        # Anthropic 格式：assistant 消息包含 content blocks
        api_messages.append({"role": "assistant", "content": anthropic_blocks})

        # ── 执行工具 ──
        code = tool_args.get("code", "")
        if tool_name == "run_public_tests":
            public_calls += 1
            if public_calls > MAX_PUBLIC_CALLS:
                result = {
                    "action": "run_public_tests",
                    "verdict": "public_test_limit_exhausted",
                    "passed": 0, "total": 0, "first_failed": None, "tests": [],
                }
            else:
                result = run_oj_judge(code, public_tests, action="run_public_tests")
        elif tool_name == "submit_solution":
            submissions += 1
            pub_result = run_oj_judge(code, public_tests, action="run_public_tests")
            if pub_result["verdict"] != "accepted":
                observation = (
                    "[ERROR] Your code did not pass the public test cases. "
                    "Please fix the code and ensure it passes public tests before submitting.\n\n"
                    + format_judge_observation(pub_result, include_all_failures=False)
                )
                # Anthropic 格式：tool_result 放在 user 消息中
                api_messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": observation}],
                })
                sft_messages.append({"role": "tool", "content": observation})
                continue
            result = run_oj_judge(code, private_tests, action="submit_solution")
            result["submission_index"] = submissions
            result["max_submissions"] = MAX_SUBMISSIONS
            if submissions >= MAX_SUBMISSIONS:
                result["verdict"] = result.get("verdict", "submission_limit_exhausted")
        else:
            observation = f"Unknown tool: {tool_name}"
            api_messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": observation}],
            })
            sft_messages.append({"role": "tool", "content": observation})
            continue

        observation = format_judge_observation(result, include_all_failures=False)
        api_messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": observation}],
        })
        sft_messages.append({"role": "tool", "content": observation})

        verdict = result.get("verdict", "unknown")
        terminal = False
        if verdict == "accepted" and tool_name == "submit_solution":
            terminal = True
        elif submissions >= MAX_SUBMISSIONS and tool_name == "submit_solution":
            terminal = True

        if terminal:
            return {
                "messages": sft_messages,
                "verdict": verdict,
                "task_id": extra.get("task_id", f"sample_{index}"),
                "turns": turn + 1,
                "submissions": submissions,
                "public_calls": public_calls,
                "title": title,
            }

    return {
        "messages": sft_messages,
        "verdict": "max_turns",
        "task_id": extra.get("task_id", f"sample_{index}"),
        "turns": MAX_ASSISTANT_TURNS,
        "submissions": submissions,
        "public_calls": public_calls,
        "title": title,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate SFT trajectories with Claude")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--max-samples", type=int, default=0, help="Max problems to process (0=all)")
    parser.add_argument("--target-accepted", type=int, default=0,
                        help="Stop after this many accepted (0=process all --max-samples)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent API workers")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Claude model name")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max tokens per API call (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET,
                        help=f"Extended thinking token budget (default: {DEFAULT_THINKING_BUDGET})")
    parser.add_argument("--resume", default=None, help="Resume from existing JSONL, skipping already processed task_ids")
    args = parser.parse_args()

    # ── 断点续跑：读取已处理的 task_id ──
    done_ids = set()
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path, encoding="utf-8") as rf:
                for line in rf:
                    try:
                        done_ids.add(json.loads(line)["task_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
            print(f"Resume: {len(done_ids)} task_ids already processed from {args.resume}")

    df = pd.read_parquet(args.input)
    total = len(df)
    if args.max_samples > 0:
        df = df.head(args.max_samples)
    n = len(df)
    target = args.target_accepted
    print(f"Loaded {total} problems, processing {n}, workers={args.workers}"
          + (f", target accepted={target}" if target else "")
          + f"\nmodel={args.model}, max_tokens={args.max_tokens}, thinking_budget={args.thinking_budget}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    accepted_count = 0
    failed_count = 0
    completed_count = 0
    # 续跑时追加写入，全新运行时覆盖写入
    write_mode = "a" if args.resume else "w"
    out_f = open(output_path, write_mode, encoding="utf-8")

    def process_one(idx_row):
        nonlocal accepted_count, failed_count, completed_count
        idx, row = idx_row
        client = get_client()
        try:
            traj = run_problem(
                client, args.model, row, idx,
                max_tokens=args.max_tokens,
                thinking_budget=args.thinking_budget,
            )
        except Exception:
            with lock:
                failed_count += 1
                completed_count += 1
                print(f"[{completed_count}/{n}] ERROR  (accepted={accepted_count}, failed={failed_count})")
            traceback.print_exc()
            return

        with lock:
            completed_count += 1
            if traj:
                verdict = traj.get("verdict", "?")
                if verdict == "accepted":
                    accepted_count += 1
                out_f.write(json.dumps(traj, ensure_ascii=False) + "\n")
                out_f.flush()
                verb = "ACCEPTED" if verdict == "accepted" else verdict.upper()
                print(f"[{completed_count}/{n}] {verb}  turns={traj['turns']}  "
                      f"{traj['title'][:50]}")
            else:
                failed_count += 1
                print(f"[{completed_count}/{n}] API_ERROR  (saved={completed_count - failed_count})")

    tasks = [(i, row) for i, (_, row) in enumerate(df.iterrows())]
    # 断点续跑：跳过已处理的 task_id
    if done_ids:
        def _get_task_id(row):
            extra = row["extra_info"]
            if isinstance(extra, str):
                extra = json.loads(extra)
            return extra.get("task_id", "")
        tasks = [(i, row) for i, row in tasks if _get_task_id(row) not in done_ids]
        print(f"Skipping {n - len(tasks)} already processed, {len(tasks)} remaining")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, t): t for t in tasks}
        for future in as_completed(futures):
            future.result()
            if target and accepted_count >= target:
                for f in futures:
                    f.cancel()
                break

    out_f.close()

    print(f"\nDone. {accepted_count} accepted, {failed_count} failed out of {completed_count}")
    print(f"JSONL: {output_path}")


if __name__ == "__main__":
    main()
