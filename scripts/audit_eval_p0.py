#!/usr/bin/env python3
"""Audit P0 correctness signals in OJ-like verl generation dumps."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_KNOWN_TOOLS = {"run_public_tests", "submit_solution"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _tool_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    trace = record.get("code_agent_trace") or {}
    events = trace.get("tool_calls") if isinstance(trace, dict) else None
    return events if isinstance(events, list) else []


def _valid_tool_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in _tool_events(record) if event.get("executed")]


def _valid_raw_tool_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for match in _TOOL_CALL_RE.finditer(record.get("output", "") or ""):
        try:
            payload = json.loads(match.group(1).strip())
        except Exception:
            continue
        arguments = payload.get("arguments")
        if (
            payload.get("name") in _KNOWN_TOOLS
            and isinstance(arguments, dict)
            and isinstance(arguments.get("code"), str)
        ):
            calls.append(payload)
    return calls


def _message_tool_calls(record: dict[str, Any]) -> int:
    total = 0
    for message in record.get("messages") or []:
        if isinstance(message, dict):
            total += len(message.get("tool_calls") or [])
    return total


def _has_think_then_tool(record: dict[str, Any]) -> bool:
    for turn in (record.get("code_agent_trace") or {}).get("assistant_turns", []):
        text = turn.get("text", "") if isinstance(turn, dict) else ""
        if "<think>" in text and "</think>" in text and "<tool_call>" in text:
            return True
    output = record.get("output", "")
    return "<think>" in output and "</think>" in output and "<tool_call>" in output


def _terminal_violation(record: dict[str, Any]) -> bool:
    terminal_seen = False
    for event in _valid_tool_events(record):
        if terminal_seen:
            return True
        result = event.get("result") or {}
        if event.get("terminal") or result.get("verdict") in {"accepted", "submission_limit_exceeded"}:
            terminal_seen = True
    return False


def _observation_mismatch(record: dict[str, Any]) -> bool:
    for event in _valid_tool_events(record):
        result = event.get("result") or {}
        observation = event.get("observation") or ""
        verdict = result.get("verdict")
        if verdict and verdict not in observation:
            return True
        if result.get("action") and result["action"] not in observation:
            return True
        if "passed" in result and "total" in result:
            count = f"{result['passed']}/{result['total']}"
            if count not in observation:
                return True
    return False


def audit(records: list[dict[str, Any]], *, expected_rows: int | None) -> list[str]:
    failures: list[str] = []
    if expected_rows is not None and len(records) != expected_rows:
        failures.append(f"expected {expected_rows} records, found {len(records)}")

    max_tool_calls = max((len(_valid_tool_events(record)) for record in records), default=0)
    if max_tool_calls >= 50:
        failures.append(f"max valid tool calls is suspiciously high: {max_tool_calls}")

    for index, record in enumerate(records):
        trace_tool_calls = len(_valid_tool_events(record))
        raw_tool_calls = len(_valid_raw_tool_calls(record))
        if raw_tool_calls != trace_tool_calls:
            failures.append(
                f"row {index}: raw output has {raw_tool_calls} valid tool calls but trace executed {trace_tool_calls}"
            )

        reported_tool_calls = record.get("num_tool_calls")
        if reported_tool_calls is not None and int(reported_tool_calls) != trace_tool_calls:
            failures.append(
                f"row {index}: num_tool_calls={reported_tool_calls} but trace has {trace_tool_calls}"
            )

        message_tool_calls = _message_tool_calls(record)
        if message_tool_calls and message_tool_calls != trace_tool_calls:
            failures.append(
                f"row {index}: messages have {message_tool_calls} tool calls but trace has {trace_tool_calls}"
            )

        if _terminal_violation(record):
            failures.append(f"row {index}: valid tool call appears after terminal event")

        if _observation_mismatch(record):
            failures.append(f"row {index}: observation does not match structured tool result")

    think_tool = [i for i, record in enumerate(records) if _has_think_then_tool(record)]
    if len(think_tool) < 3:
        failures.append(f"only {len(think_tool)} records show </think> followed by tool call")

    failed_feedback = []
    for i, record in enumerate(records):
        for event in _valid_tool_events(record):
            verdict = (event.get("result") or {}).get("verdict")
            if verdict and verdict != "accepted":
                failed_feedback.append(i)
                break
    if len(failed_feedback) < 3:
        failures.append(f"only {len(failed_feedback)} records contain failed public/submit feedback")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("final", type=Path, help="Final generations/0.jsonl path")
    parser.add_argument("--partial", type=Path, default=None, help="Optional partial_0.jsonl path")
    parser.add_argument("--expected-rows", type=int, default=32)
    args = parser.parse_args()

    records = _load_jsonl(args.final)
    failures = audit(records, expected_rows=args.expected_rows)

    if args.partial:
        partial_records = _load_jsonl(args.partial)
        if len(partial_records) != len(records):
            failures.append(f"partial rows {len(partial_records)} != final rows {len(records)}")

    max_tool_calls = max((len(_valid_tool_events(record)) for record in records), default=0)
    think_tool_rows = [i for i, record in enumerate(records) if _has_think_then_tool(record)][:3]
    failed_rows = []
    for i, record in enumerate(records):
        if any((event.get("result") or {}).get("verdict") not in {None, "accepted"} for event in _valid_tool_events(record)):
            failed_rows.append(i)
        if len(failed_rows) >= 3:
            break

    print(f"records: {len(records)}")
    print(f"max_valid_tool_calls: {max_tool_calls}")
    print(f"sample_think_then_tool_rows: {think_tool_rows}")
    print(f"sample_failed_feedback_rows: {failed_rows}")

    if failures:
        print("P0 audit failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("P0 audit passed")


if __name__ == "__main__":
    main()
