"""Teacher 模型生成 SFT 训练 trajectory。

对 CodeContests train 中的每道题，用强 teacher 模型（DeepSeek V4 Pro）
在 OJ-like 两工具协议下交互解题。只保留 accepted 的 trajectory，
输出为 JSONL（供人工检查）和 parquet（供 verl MultiTurnSFTDataset 直接加载）。

用法：
  先 export 环境变量 DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL，然后：
  python3 scripts/generate_sft_trajectories.py \
      --input data/verl/codecontests_train.parquet \
      --output data/verl/sft_trajectories.jsonl \
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

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

# ── API client ────────────────────────────────────────
from openai import OpenAI


def get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        print("[ERROR] DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=base_url)


# ── Judge ─────────────────────────────────────────────
from src.env.tools import (
    OJTestCase,
    VERDICT_NO_TESTS,
    format_judge_observation,
    parse_oj_tests,
    run_oj_judge,
)

# ── Tool schemas（和 eval 环境完全一致）───────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_public_tests",
            "description": "Run your code against public test cases. "
            "Use this to check if your solution passes the example tests. "
            "Returns per-test pass/fail results. Stops at the first failure.",
            "parameters": {
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
    },
    {
        "type": "function",
        "function": {
            "name": "submit_solution",
            "description": "Submit your final solution for full judging. "
            "You have at most 5 submissions total. "
            "Only call this when public tests pass.",
            "parameters": {
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
    },
]

MAX_SUBMISSIONS = 5
MAX_PUBLIC_CALLS = 15
MAX_ASSISTANT_TURNS = 20


def call_teacher(client: OpenAI, messages: list[dict], model: str) -> dict | None:
    """调 teacher API，返回 assistant message（含 reasoning_content 作为 thinking）。"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                temperature=0.6,
                max_tokens=16384,
                timeout=120,
            )
            choice = resp.choices[0]
            msg = choice.message.model_dump(exclude_none=True)
            # DeepSeek 的思考过程在 reasoning_content 字段，Pydantic v2 用 model_extra 存储
            raw = getattr(choice.message, "model_extra", None) or {}
            reasoning = raw.get("reasoning_content", "") if isinstance(raw, dict) else ""
            if not reasoning:
                # 也尝试从 model_dump 后的 dict 直接取
                reasoning = choice.message.model_dump().get("reasoning_content", "") or ""
            if reasoning:
                msg["reasoning_content"] = reasoning
            return msg
        except Exception as e:
            print(f"  [API retry {attempt+1}/3] {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return None


def parse_tool_call(msg: dict) -> tuple[str | None, dict | None, str | None]:
    """从 assistant message 中提取 tool call name、arguments 和 tool_call_id。"""
    tool_calls = msg.get("tool_calls", [])
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
    client: OpenAI,
    model: str,
    row: pd.Series,
    index: int,
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

    submissions = 0
    public_calls = 0
    api_messages = list(messages)  # 发 API 用，OpenAI 原生 tool_calls 格式
    sft_messages = list(messages)  # 存 SFT 数据用，文本 <tool_call> 格式

    for turn in range(MAX_ASSISTANT_TURNS):
        msg = call_teacher(client, api_messages, model)
        if msg is None:
            return None  # API 错误，跳过这道题

        reasoning = msg.get("reasoning_content", "") or ""
        content = msg.get("content", "") or ""
        tool_name, tool_args, tool_call_id = parse_tool_call(msg)

        if not tool_name:
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

        # ── assistant 消息 ──
        # API 格式：保留原始 structured tool_calls
        api_messages.append(msg)
        # SFT 格式：<think>思考过程</think> + <tool_call>
        tool_call_json = json.dumps({"name": tool_name, "arguments": tool_args}, ensure_ascii=False)
        if reasoning:
            sft_text = "<think>" + reasoning + "</think>\n\n<tool_call>" + tool_call_json + "</tool_call>"
        else:
            sft_text = "<tool_call>" + tool_call_json + "</tool_call>"
        sft_messages.append({"role": "assistant", "content": sft_text})

        # ── 执行工具 ──
        code = tool_args.get("code", "")
        if tool_name == "run_public_tests":
            public_calls += 1
            if public_calls > MAX_PUBLIC_CALLS:
                result = {"action": "run_public_tests", "verdict": "public_test_limit_exhausted",
                          "passed": 0, "total": 0, "first_failed": None, "tests": []}
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
                api_messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": observation})
                sft_messages.append({"role": "tool", "content": observation})
                continue
            result = run_oj_judge(code, private_tests, action="submit_solution")
            result["submission_index"] = submissions
            result["max_submissions"] = MAX_SUBMISSIONS
            if submissions >= MAX_SUBMISSIONS:
                result["verdict"] = result.get("verdict", "submission_limit_exhausted")
        else:
            observation = f"Unknown tool: {tool_name}"
            api_messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": observation})
            sft_messages.append({"role": "tool", "content": observation})
            continue

        observation = format_judge_observation(result, include_all_failures=False)
        api_messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": observation})
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
    parser = argparse.ArgumentParser(description="Generate SFT trajectories with teacher model")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--parquet-output", default=None, help="Also output parquet file")
    parser.add_argument("--max-samples", type=int, default=0, help="Max problems to process (0=all)")
    parser.add_argument("--target-accepted", type=int, default=0,
                        help="Stop after this many accepted (0=process all --max-samples)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent API workers")
    parser.add_argument("--model", default="deepseek-v4-pro", help="Teacher model name")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    total = len(df)
    if args.max_samples > 0:
        df = df.head(args.max_samples)
    n = len(df)
    target = args.target_accepted
    print(f"Loaded {total} problems, processing {n}, workers={args.workers}"
          + (f", target accepted={target}" if target else ""))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 共享状态
    lock = threading.Lock()
    accepted_count = 0
    failed_count = 0
    completed_count = 0
    out_f = open(output_path, "w", encoding="utf-8")

    def process_one(idx_row):
        """单个题目的处理函数（在线程池中执行）。"""
        nonlocal accepted_count, failed_count, completed_count
        idx, row = idx_row
        client = get_client()
        try:
            traj = run_problem(client, args.model, row, idx)
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
                print(f"[{completed_count}/{n}] API_ERROR  (saved={completed_count-failed_count})")

    # 准备任务列表
    tasks = [(i, row) for i, (_, row) in enumerate(df.iterrows())]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, t): t for t in tasks}
        for future in as_completed(futures):
            future.result()  # 获取异常（如果有）
            # 如果达到目标数量，取消剩余任务
            if target and accepted_count >= target:
                for f in futures:
                    f.cancel()
                break

    out_f.close()

    print(f"\nDone. {accepted_count} accepted, {failed_count} failed out of {completed_count}")
    print(f"JSONL: {output_path}")

    # 可选：写 parquet
    if args.parquet_output:
        rows = []
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                rows.append({"messages": json.loads(line)["messages"],
                             "data_source": "codecontests_sft"})
        pd.DataFrame(rows).to_parquet(args.parquet_output, index=False)
        print(f"Parquet: {args.parquet_output}")


if __name__ == "__main__":
    main()
