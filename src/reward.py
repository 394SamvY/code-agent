"""Structured-event reward for OJ-like two-action GRPO training.

The reward input is the real tool execution event stream recorded by
``CodeAgentToolAgentLoop``.  ``solution_str`` is intentionally ignored.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from typing import Any


EVENTS_KEY = "code_agent_tool_events"
TRACE_KEY = "code_agent_trace"
KNOWN_TOOLS = {"run_public_tests", "submit_solution"}

OUTCOME_SUBMIT_POLICY = "last_submit"

BAD_PATTERN_MIN = -0.60
BAD_PATTERN_MAX = 0.0
FINAL_MIN = -0.50
FINAL_MAX = 1.00
NON_ACCEPTED_MAX = 0.00
NON_ACCEPTED_SUBMIT_FLOOR = -0.20
NO_SUBMIT_FLOOR = -0.25
PUBLIC_ACC_NO_SUBMIT_FLOOR = -0.35
PROTOCOL_ERROR_FLOOR = -0.30
TRUNCATED_NO_SUBMIT_FLOOR = -0.35
AC_PROTOCOL_ERROR_MAX = 0.85


@dataclass
class RewardEvent:
    index: int
    tool: str
    verdict: str
    passed: int
    total: int
    pass_rate: float
    code: str | None
    code_hash: str | None
    semantic_hash: str | None
    observation: str
    first_failed: dict[str, Any] | None
    error_kind: str | None


def _stable_float(value: float) -> float:
    return round(float(value), 10)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def normalize_code(code: str) -> str:
    lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def code_hash(code: Any) -> str | None:
    if not isinstance(code, str):
        return None
    normalized = normalize_code(code)
    if not normalized:
        return None
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def semantic_code_hash(code: Any) -> str | None:
    if not isinstance(code, str):
        return None
    normalized = normalize_code(code)
    if not normalized:
        return None
    try:
        tree = ast.parse(normalized)
    except SyntaxError:
        return "syntax:" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    dumped = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return "ast:" + hashlib.sha1(dumped.encode("utf-8")).hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _event_error_kind(raw: dict[str, Any]) -> str | None:
    explicit = raw.get("error_kind")
    if explicit:
        return str(explicit)

    verdict = str(raw.get("verdict") or "")
    first_failed = _as_dict(raw.get("first_failed"))
    stderr = str(first_failed.get("stderr") or raw.get("stderr") or raw.get("observation") or "")

    if "IndexError" in stderr or "index out of range" in stderr:
        return "index_error"
    if "KeyError" in stderr:
        return "key_error"
    if "RecursionError" in stderr or "maximum recursion depth exceeded" in stderr:
        return "recursion_error"
    if verdict == "time_limit_exceeded" or "Time Limit Exceeded" in stderr or "TLE" in stderr:
        return "time_limit_exceeded"
    if verdict == "syntax_error":
        return "syntax_error"
    if verdict == "runtime_error":
        return "runtime_error"
    if verdict == "wrong_answer":
        return "wrong_answer"
    return verdict or None


def _normalize_event(raw: Any, index: int) -> RewardEvent | None:
    if not isinstance(raw, dict):
        return None
    tool = str(raw.get("tool") or raw.get("action") or "")
    verdict = str(raw.get("verdict") or "unknown")
    passed = int(raw.get("passed") or 0)
    total = int(raw.get("total") or 0)
    pass_rate = float(raw.get("pass_rate")) if raw.get("pass_rate") is not None else (passed / total if total else 0.0)
    code = raw.get("code")
    if code is not None and not isinstance(code, str):
        code = str(code)
    normalized_hash = raw.get("code_hash") or code_hash(code)
    normalized_semantic_hash = raw.get("semantic_hash") or semantic_code_hash(code)
    first_failed = raw.get("first_failed") if isinstance(raw.get("first_failed"), dict) else None
    return RewardEvent(
        index=int(raw.get("index", index)),
        tool=tool,
        verdict=verdict,
        passed=passed,
        total=total,
        pass_rate=pass_rate,
        code=code,
        code_hash=str(normalized_hash) if normalized_hash else None,
        semantic_hash=str(normalized_semantic_hash) if normalized_semantic_hash else None,
        observation=str(raw.get("observation") or ""),
        first_failed=first_failed,
        error_kind=_event_error_kind(raw),
    )


def load_structured_events(extra_info: dict[str, Any] | None) -> list[RewardEvent]:
    extra = extra_info or {}
    if EVENTS_KEY not in extra:
        raise ValueError(f"Missing required structured reward input: extra_info[{EVENTS_KEY!r}]")
    events: list[RewardEvent] = []
    for index, raw in enumerate(_as_list(extra.get(EVENTS_KEY))):
        event = _normalize_event(raw, index)
        if event is not None:
            events.append(event)
    return events


def _choose_submit(submit_events: list[RewardEvent], policy: str) -> RewardEvent | None:
    if not submit_events:
        return None
    if policy == "best_submit":
        return max(submit_events, key=lambda event: (event.verdict == "accepted", event.pass_rate, event.index))
    if policy == "any_ac_else_best":
        accepted = [event for event in submit_events if event.verdict == "accepted"]
        if accepted:
            return accepted[0]
        return max(submit_events, key=lambda event: (event.pass_rate, event.index))
    if policy != "last_submit":
        raise ValueError(f"Unknown outcome submit policy: {policy}")
    return submit_events[-1]


def _compute_outcome_reward(
    submit_events: list[RewardEvent],
    *,
    policy: str,
) -> tuple[float, str, float]:
    submit = _choose_submit(submit_events, policy)
    if submit is None:
        return 0.0, "no_submission", 0.0
    if submit.verdict == "accepted":
        return 1.0, submit.verdict, submit.pass_rate
    # Non-AC submissions are objective but non-positive: higher private pass rate is
    # less bad, while only accepted can cross above zero.
    return NON_ACCEPTED_SUBMIT_FLOOR * (1.0 - submit.pass_rate), submit.verdict, submit.pass_rate


def _submit_diagnostics(submit_events: list[RewardEvent], *, policy: str) -> dict[str, float | str]:
    last_submit = submit_events[-1] if submit_events else None
    best_submit = (
        max(submit_events, key=lambda event: (event.verdict == "accepted", event.pass_rate, event.index))
        if submit_events
        else None
    )
    acc_final = 1.0 if last_submit is not None and last_submit.verdict == "accepted" else 0.0
    acc_any = 1.0 if any(event.verdict == "accepted" for event in submit_events) else 0.0
    return {
        "acc_final": acc_final,
        "acc_any": acc_any,
        "best_submit_pass_rate": best_submit.pass_rate if best_submit is not None else 0.0,
        "last_submit_pass_rate": last_submit.pass_rate if last_submit is not None else 0.0,
        "outcome_submit_policy": policy,
    }


def _count_duplicate_semantic_hashes(events: list[RewardEvent]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for event in events:
        if not event.semantic_hash:
            continue
        if event.semantic_hash in seen:
            duplicates += 1
        else:
            seen.add(event.semantic_hash)
    return duplicates


def _count_fail_same_code_submit(events: list[RewardEvent], prev_tool: str) -> int:
    count = 0
    for prev, current in zip(events, events[1:]):
        if current.tool != "submit_solution" or prev.tool != prev_tool:
            continue
        if prev.verdict == "accepted":
            continue
        if prev.semantic_hash and prev.semantic_hash == current.semantic_hash:
            count += 1
    return count


def _accepted_then_tool_call(events: list[RewardEvent]) -> bool:
    accepted_index = None
    for index, event in enumerate(events):
        if event.tool == "submit_solution" and event.verdict == "accepted":
            accepted_index = index
            break
    return accepted_index is not None and accepted_index < len(events) - 1


def _accepted_then_later_failed_submit(events: list[RewardEvent]) -> bool:
    seen_accepted = False
    for event in events:
        if event.tool != "submit_solution":
            continue
        if seen_accepted and event.verdict != "accepted":
            return True
        if event.verdict == "accepted":
            seen_accepted = True
    return False


def _compute_bad_pattern_penalty(
    events: list[RewardEvent],
    *,
    trace: dict[str, Any],
    parse_failures: int,
) -> tuple[float, dict[str, float]]:
    penalties: dict[str, float] = {}

    def add(key: str, value: float) -> None:
        if value:
            penalties[key] = penalties.get(key, 0.0) + value

    public_events = [event for event in events if event.tool == "run_public_tests"]
    submit_events = [event for event in events if event.tool == "submit_solution"]
    malformed_events = [
        event
        for event in events
        if event.tool == "malformed_tool_call"
        or event.verdict == "tool_parse_error"
        or event.error_kind == "malformed_tool_call"
    ]
    unknown_tool_events = [
        event
        for event in events
        if event.tool not in KNOWN_TOOLS and event.tool != "malformed_tool_call"
    ]
    tool_execution_errors = [
        event
        for event in events
        if event.tool in KNOWN_TOOLS
        and (event.verdict == "tool_execution_error" or event.error_kind == "tool_execution_error")
    ]
    no_submit = not submit_events
    public_accepted = any(event.verdict == "accepted" for event in public_events)
    response_truncated = trace.get("terminal_reason") == "response_length_exceeded"
    parse_error_count = max(parse_failures, len(malformed_events))

    if no_submit:
        add("no_submit", -0.25)
    if no_submit and public_accepted:
        add("public_acc_no_submit", -0.10)
    if response_truncated:
        add("response_truncated", -0.15)
    if response_truncated and no_submit:
        add("truncated_no_submit", -0.05)

    add("public_fail_same_code_submit", -min(0.12, 0.06 * _count_fail_same_code_submit(events, "run_public_tests")))
    add("submit_fail_same_code_submit", -min(0.16, 0.08 * _count_fail_same_code_submit(events, "submit_solution")))
    add("duplicate_public_code", -min(0.08, 0.02 * _count_duplicate_semantic_hashes(public_events)))
    add("duplicate_submit_code", -min(0.10, 0.03 * _count_duplicate_semantic_hashes(submit_events)))

    add("too_many_public_tests", -min(0.05, 0.01 * max(0, len(public_events) - 5)))
    add("too_many_submits", -min(0.09, 0.03 * max(0, len(submit_events) - 2)))

    accepted_then_tool_call = _accepted_then_tool_call(events) or bool(trace.get("has_tool_call_after_submit_accepted"))
    if accepted_then_tool_call:
        add("accepted_then_tool_call", -0.15)
    if _accepted_then_later_failed_submit(events):
        add("accepted_then_later_failed_submit", -0.25)

    if any(event.verdict == "public_test_limit_exceeded" for event in public_events):
        add("public_test_limit_exceeded", -0.08)
    if any(event.verdict == "submission_limit_exceeded" for event in submit_events):
        add("submission_limit_exceeded", -0.12)

    add("malformed_tool_call", -min(0.30, 0.15 * parse_error_count))
    add("unknown_tool", -min(0.30, 0.20 * len(unknown_tool_events)))
    add("tool_execution_error", -min(0.30, 0.20 * len(tool_execution_errors)))

    accepted_text_chars = int(trace.get("assistant_chars_after_submit_accepted") or 0)
    if accepted_text_chars > 1500:
        add("accepted_then_long_text", -0.05)
    elif accepted_text_chars > 500:
        add("accepted_then_long_text", -0.02)

    empty_think_count = int(trace.get("empty_think_after_submit_accepted_count") or 0)
    consecutive_empty_think = bool(trace.get("consecutive_empty_think_after_submit_accepted"))
    if not accepted_then_tool_call and (consecutive_empty_think or empty_think_count >= 2):
        add("empty_think_after_submit_accepted", -min(0.08, 0.02 * max(1, empty_think_count - 1)))

    total = _clamp(sum(penalties.values()), BAD_PATTERN_MIN, BAD_PATTERN_MAX)
    return total, penalties


def _parse_failures(extra_info: dict[str, Any], trace: dict[str, Any]) -> int:
    for value in (trace.get("parse_failures"), extra_info.get("code_agent_parse_failures")):
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return 0


def _breakdown_json(
    *,
    outcome_reward: float,
    bad_pattern: float,
    final: float,
    acc: float,
    outcome_verdict: str,
    outcome_pass_rate: float,
    submit_policy: str,
    bad_patterns: dict[str, float],
    event_count: int,
    parse_failures: int,
    submit_diagnostics: dict[str, float | str],
) -> str:
    return json.dumps(
        {
            "acc": _stable_float(acc),
            "acc_any": _stable_float(float(submit_diagnostics["acc_any"])),
            "acc_final": _stable_float(float(submit_diagnostics["acc_final"])),
            "bad_pattern": _stable_float(bad_pattern),
            "bad_patterns": {key: _stable_float(value) for key, value in bad_patterns.items()},
            "best_submit_pass_rate": _stable_float(float(submit_diagnostics["best_submit_pass_rate"])),
            "event_count": int(event_count),
            "final": _stable_float(final),
            "last_submit_pass_rate": _stable_float(float(submit_diagnostics["last_submit_pass_rate"])),
            "outcome_pass_rate": _stable_float(outcome_pass_rate),
            "outcome_reward": _stable_float(outcome_reward),
            "outcome_submit_policy": str(submit_diagnostics["outcome_submit_policy"]),
            "outcome_verdict": outcome_verdict,
            "parse_failures": int(parse_failures),
            "reward_formula": "outcome + bad_pattern",
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Compute reward from structured tool execution events only."""
    del data_source, solution_str, ground_truth, kwargs
    extra = extra_info or {}
    events = load_structured_events(extra)
    known_events = [event for event in events if event.tool in KNOWN_TOOLS]
    trace = _as_dict(extra.get(TRACE_KEY))
    parse_failures = _parse_failures(extra, trace)

    submit_events = [event for event in known_events if event.tool == "submit_solution"]
    last_submit = submit_events[-1] if submit_events else None
    acc = 1.0 if last_submit is not None and last_submit.verdict == "accepted" else 0.0

    submit_policy = str(extra.get("outcome_submit_policy") or OUTCOME_SUBMIT_POLICY)
    submit_diagnostics = _submit_diagnostics(submit_events, policy=submit_policy)
    outcome_reward, outcome_verdict, outcome_pass_rate = _compute_outcome_reward(
        submit_events,
        policy=submit_policy,
    )
    bad_pattern, bad_patterns = _compute_bad_pattern_penalty(
        events,
        trace=trace,
        parse_failures=parse_failures,
    )

    public_events = [event for event in known_events if event.tool == "run_public_tests"]
    public_accepted = any(event.verdict == "accepted" for event in public_events)
    has_malformed = parse_failures > 0 or any(
        event.tool == "malformed_tool_call"
        or event.verdict == "tool_parse_error"
        or event.error_kind == "malformed_tool_call"
        for event in events
    )
    has_unknown_or_tool_error = any(
        (event.tool not in KNOWN_TOOLS and event.tool != "malformed_tool_call")
        or event.verdict == "tool_execution_error"
        or event.error_kind == "tool_execution_error"
        for event in events
    )

    accepted_for_outcome = outcome_verdict == "accepted"
    final = outcome_reward + bad_pattern
    if not accepted_for_outcome:
        final = min(final, NON_ACCEPTED_MAX)
        if not submit_events:
            final = min(final, NO_SUBMIT_FLOOR)
        if not submit_events and public_accepted:
            final = min(final, PUBLIC_ACC_NO_SUBMIT_FLOOR)
        if not submit_events and trace.get("terminal_reason") == "response_length_exceeded":
            final = min(final, TRUNCATED_NO_SUBMIT_FLOOR)
        if has_malformed or has_unknown_or_tool_error:
            final = min(final, PROTOCOL_ERROR_FLOOR)
    elif has_malformed or has_unknown_or_tool_error:
        final = min(final, AC_PROTOCOL_ERROR_MAX)
    final = _stable_float(_clamp(final, FINAL_MIN, FINAL_MAX))

    breakdown = _breakdown_json(
        outcome_reward=outcome_reward,
        bad_pattern=bad_pattern,
        final=final,
        acc=acc,
        outcome_verdict=outcome_verdict,
        outcome_pass_rate=outcome_pass_rate,
        submit_policy=submit_policy,
        bad_patterns=bad_patterns,
        event_count=len(events),
        parse_failures=parse_failures,
        submit_diagnostics=submit_diagnostics,
    )
    return {
        "score": final,
        "reward": final,
        "acc": acc,
        "acc_final": _stable_float(float(submit_diagnostics["acc_final"])),
        "acc_any": _stable_float(float(submit_diagnostics["acc_any"])),
        "best_submit_pass_rate": _stable_float(float(submit_diagnostics["best_submit_pass_rate"])),
        "last_submit_pass_rate": _stable_float(float(submit_diagnostics["last_submit_pass_rate"])),
        "num_tool_calls": len(events),
        "reward_breakdown": breakdown,
        "outcome_reward": _stable_float(outcome_reward),
        "outcome_submit_policy": str(submit_diagnostics["outcome_submit_policy"]),
        "bad_pattern": _stable_float(bad_pattern),
    }
