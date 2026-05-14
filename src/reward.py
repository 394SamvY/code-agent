"""Structured-event reward for OJ-like two-action GRPO training.

The reward input is the real tool execution event stream recorded by
``CodeAgentToolAgentLoop``.  ``solution_str`` is intentionally ignored.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
import re
from typing import Any


EVENTS_KEY = "code_agent_tool_events"
TRACE_KEY = "code_agent_trace"

OUTCOME_SUBMIT_POLICY = "last_submit"

DEBUG_MIN = -0.10
DEBUG_MAX = 0.15
WEAK_WA_LOGIC_CHANGED_BONUS = 0.005
BAD_PATTERN_MIN = -0.40
BAD_PATTERN_MAX = 0.0
FINAL_MIN = -0.50
FINAL_MAX = 1.00
NON_ACCEPTED_MAX = 0.60


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
    if tool not in {"run_public_tests", "submit_solution"}:
        return None
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
    return min(0.5, 0.4 * submit.pass_rate), submit.verdict, submit.pass_rate


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


def _state_rank(event: RewardEvent) -> int:
    if event.verdict == "accepted":
        return 6 if event.tool == "submit_solution" else 4
    if event.tool == "submit_solution" and event.pass_rate > 0:
        return 5
    if event.verdict == "syntax_error":
        return 1
    if event.verdict in {"runtime_error", "time_limit_exceeded"}:
        return 2
    if event.verdict == "wrong_answer":
        return 3
    return 0


def _added(old: str, new: str, pattern: str) -> bool:
    return pattern not in old and pattern in new


def _changed_range_expr(old: str, new: str) -> bool:
    return old.count("range(") != new.count("range(") or bool(re.search(r"range\([^)]*[+\-][^)]*\)", new))


def _added_boundary_comparison(old: str, new: str) -> bool:
    patterns = (r"\b\w+\s*[+\-]\s*1\s*[<>]=?\s*\w+", r"\b\w+\s*[<>]=?\s*\w+\s*[+\-]\s*1", r"\b0\s*<=\s*\w+\s*<")
    return any(re.search(pattern, new) and not re.search(pattern, old) for pattern in patterns)


def _added_empty_guard(old: str, new: str) -> bool:
    return bool(re.search(r"\bif\s+(not\s+)?\w+\s*:", new)) and old != new


def _aligned_index_error(old: str, new: str) -> bool:
    return (
        _added(old, new, "len(")
        or _added_empty_guard(old, new)
        or _changed_range_expr(old, new)
        or _added_boundary_comparison(old, new)
    )


def _aligned_key_error(old: str, new: str) -> bool:
    return _added(old, new, ".get(") or _added(old, new, "defaultdict") or bool(re.search(r"\b\w+\s+in\s+\w+", new) and not re.search(r"\b\w+\s+in\s+\w+", old))


def _aligned_recursion_error(old: str, new: str) -> bool:
    return (
        _added(old, new, "setrecursionlimit")
        or _added(old, new, "visited")
        or _added(old, new, "memo")
        or _added(old, new, "deque(")
    )


def _aligned_tle(old: str, new: str) -> bool:
    return any(
        _added(old, new, token)
        for token in ("lru_cache", "cache", "memo", "set(", "dict(", "prefix", "suffix", "precompute", "bisect", "heapq")
    )


def _logic_changed(old: str, new: str) -> bool:
    return semantic_code_hash(old) != semantic_code_hash(new)


def _repair_aligned(prev: RewardEvent, next_event: RewardEvent) -> bool:
    if not prev.code or not next_event.code:
        return False
    old = normalize_code(prev.code)
    new = normalize_code(next_event.code)
    kind = prev.error_kind
    if kind == "index_error":
        return _aligned_index_error(old, new)
    if kind == "key_error":
        return _aligned_key_error(old, new)
    if kind == "recursion_error":
        return _aligned_recursion_error(old, new)
    if kind == "time_limit_exceeded":
        return _aligned_tle(old, new)
    return False


def _code_diff_ratio(old: str, new: str) -> float:
    old_norm = normalize_code(old)
    new_norm = normalize_code(new)
    if not old_norm and not new_norm:
        return 0.0
    return 1.0 - SequenceMatcher(None, old_norm, new_norm).ratio()


def _add_signal(signals: dict[str, float], key: str, value: float) -> None:
    if value:
        signals[key] = signals.get(key, 0.0) + value


def _progress_bonus(
    prev: RewardEvent,
    next_event: RewardEvent,
    weight: float,
    *,
    best_rank: int,
    best_pass_rate: float,
) -> tuple[float, str | None]:
    rank_delta = _state_rank(next_event) - max(_state_rank(prev), best_rank)
    pass_delta = next_event.pass_rate - max(prev.pass_rate, best_pass_rate)
    rank_bonus = min(0.04, 0.015 * rank_delta) if rank_delta > 0 else 0.0
    pass_bonus = min(0.03, 0.03 * pass_delta) if pass_delta > 0 else 0.0
    if rank_bonus >= pass_bonus and rank_bonus > 0:
        return weight * rank_bonus, "state_rank_improved"
    if pass_bonus > 0:
        return weight * pass_bonus, "pass_rate_improved"
    return 0.0, None


def _alignment_bonus(
    prev: RewardEvent,
    next_event: RewardEvent,
    *,
    weight: float,
    progress_bonus: float,
    allow_weak_wa: bool,
    allow_positive_bonus: bool,
) -> tuple[float, str | None]:
    next_rank = _state_rank(next_event)
    prev_rank = _state_rank(prev)
    aligned = _repair_aligned(prev, next_event)
    if aligned and next_rank >= prev_rank and allow_positive_bonus:
        if prev.error_kind == "time_limit_exceeded" and progress_bonus <= 0:
            return 0.0, None
        value = 0.02 if prev.error_kind == "time_limit_exceeded" else 0.03
        return weight * value, "feedback_aligned_not_worse"
    if aligned and next_rank < prev_rank:
        return -weight * 0.02, "feedback_aligned_but_worse"
    if allow_positive_bonus and allow_weak_wa and prev.error_kind == "wrong_answer" and prev.code and next_event.code:
        if _logic_changed(prev.code, next_event.code) and next_rank >= prev_rank:
            return weight * WEAK_WA_LOGIC_CHANGED_BONUS, "wrong_answer_logic_changed_not_worse"
    return 0.0, None


def _transition_debug_score(
    prev: RewardEvent,
    next_event: RewardEvent,
    *,
    weight: float,
    allow_progress: bool,
    allow_alignment: bool,
    allow_weak_wa: bool,
    best_rank: int,
    best_pass_rate: float,
) -> tuple[float, dict[str, float]]:
    signals: dict[str, float] = {}
    if not prev.code or not next_event.code:
        return 0.0, signals

    prev_failed = prev.verdict != "accepted"
    same_code = bool(prev.semantic_hash and prev.semantic_hash == next_event.semantic_hash)
    changed = not same_code

    if prev_failed and same_code:
        value = -weight * 0.04
        _add_signal(signals, "same_code_after_failed_feedback", value)
        return value, signals

    if not changed:
        return 0.0, signals

    score = 0.0
    progress_value = 0.0
    positive_allowed = _state_rank(next_event) > best_rank or next_event.pass_rate > best_pass_rate
    if allow_progress:
        progress_value, progress_key = _progress_bonus(
            prev,
            next_event,
            weight,
            best_rank=best_rank,
            best_pass_rate=best_pass_rate,
        )
        if progress_key:
            score += progress_value
            _add_signal(signals, progress_key, progress_value)

    if allow_alignment:
        alignment_value, alignment_key = _alignment_bonus(
            prev,
            next_event,
            weight=weight,
            progress_bonus=progress_value,
            allow_weak_wa=allow_weak_wa,
            allow_positive_bonus=positive_allowed,
        )
        if alignment_key:
            score += alignment_value
            _add_signal(signals, alignment_key, alignment_value)

    if prev.code and next_event.code and _code_diff_ratio(prev.code, next_event.code) > 0.70 and _state_rank(next_event) < _state_rank(prev):
        value = -weight * 0.04
        score += value
        _add_signal(signals, "large_rewrite_worse", value)

    return score, signals


def _compute_debug_prm(events: list[RewardEvent]) -> tuple[float, dict[str, float]]:
    public_debug_score = 0.0
    submit_debug_score = 0.0
    signals: dict[str, float] = {}
    best_rank_by_tool = {"run_public_tests": -1, "submit_solution": -1}
    best_pass_rate_by_tool = {"run_public_tests": -1.0, "submit_solution": -1.0}

    def update_best(event: RewardEvent) -> None:
        best_rank_by_tool[event.tool] = max(best_rank_by_tool[event.tool], _state_rank(event))
        best_pass_rate_by_tool[event.tool] = max(best_pass_rate_by_tool[event.tool], event.pass_rate)

    for prev, next_event in zip(events, events[1:]):
        update_best(prev)
        pair = (prev.tool, next_event.tool)
        best_rank = best_rank_by_tool[next_event.tool]
        best_pass_rate = best_pass_rate_by_tool[next_event.tool]
        if pair == ("run_public_tests", "run_public_tests"):
            score, transition_signals = _transition_debug_score(
                prev,
                next_event,
                weight=1.0,
                allow_progress=True,
                allow_alignment=True,
                allow_weak_wa=True,
                best_rank=best_rank,
                best_pass_rate=best_pass_rate,
            )
            public_debug_score += score
        elif pair == ("submit_solution", "submit_solution"):
            score, transition_signals = _transition_debug_score(
                prev,
                next_event,
                weight=0.3,
                allow_progress=True,
                allow_alignment=True,
                allow_weak_wa=True,
                best_rank=best_rank,
                best_pass_rate=best_pass_rate,
            )
            submit_debug_score += score
        elif pair == ("submit_solution", "run_public_tests"):
            score, transition_signals = _transition_debug_score(
                prev,
                next_event,
                weight=0.3,
                allow_progress=False,
                allow_alignment=next_event.pass_rate > 0 or next_event.verdict == "accepted",
                allow_weak_wa=False,
                best_rank=best_rank,
                best_pass_rate=best_pass_rate,
            )
            submit_debug_score += score
        else:
            update_best(next_event)
            continue

        for key, value in transition_signals.items():
            _add_signal(signals, key, value)
        update_best(next_event)

    submit_debug_score = _clamp(submit_debug_score, -0.03, 0.03)
    return _clamp(public_debug_score + submit_debug_score, DEBUG_MIN, DEBUG_MAX), signals


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
    no_submit = not submit_events
    public_accepted = any(event.verdict == "accepted" for event in public_events)
    response_truncated = trace.get("terminal_reason") == "response_length_exceeded"

    if no_submit:
        add("no_submit", -0.30)
    if no_submit and public_accepted:
        add("public_acc_no_submit", -0.20)
    if response_truncated:
        add("response_truncated", -0.20)
    if response_truncated and no_submit:
        add("truncated_no_submit", -0.10)

    add("public_fail_same_code_submit", -min(0.20, 0.10 * _count_fail_same_code_submit(events, "run_public_tests")))
    add("submit_fail_same_code_submit", -min(0.24, 0.12 * _count_fail_same_code_submit(events, "submit_solution")))
    add("duplicate_public_code", -min(0.10, 0.03 * _count_duplicate_semantic_hashes(public_events)))
    add("duplicate_submit_code", -min(0.16, 0.05 * _count_duplicate_semantic_hashes(submit_events)))

    add("too_many_public_tests", -min(0.08, 0.02 * max(0, len(public_events) - 5)))
    add("too_many_submits", -min(0.12, 0.04 * max(0, len(submit_events) - 2)))

    if _accepted_then_tool_call(events) or trace.get("has_tool_call_after_submit_accepted"):
        add("accepted_then_tool_call", -0.20)
    if _accepted_then_later_failed_submit(events):
        add("accepted_then_later_failed_submit", -0.30)

    if any(event.verdict == "public_test_limit_exceeded" for event in public_events):
        add("public_test_limit_exceeded", -0.10)
    if any(event.verdict == "submission_limit_exceeded" for event in submit_events):
        add("submission_limit_exceeded", -0.15)

    add("malformed_tool_call", -min(0.15, 0.05 * parse_failures))

    accepted_text_chars = int(trace.get("assistant_chars_after_submit_accepted") or 0)
    if accepted_text_chars > 1500:
        add("accepted_then_long_text", -0.05)
    elif accepted_text_chars > 500:
        add("accepted_then_long_text", -0.02)

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
    debug_prm: float,
    bad_pattern: float,
    final: float,
    acc: float,
    outcome_verdict: str,
    outcome_pass_rate: float,
    submit_policy: str,
    debug_signals: dict[str, float],
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
            "debug_prm": _stable_float(debug_prm),
            "debug_signals": {key: _stable_float(value) for key, value in debug_signals.items()},
            "event_count": int(event_count),
            "final": _stable_float(final),
            "last_submit_pass_rate": _stable_float(float(submit_diagnostics["last_submit_pass_rate"])),
            "outcome_pass_rate": _stable_float(outcome_pass_rate),
            "outcome_reward": _stable_float(outcome_reward),
            "outcome_submit_policy": str(submit_diagnostics["outcome_submit_policy"]),
            "outcome_verdict": outcome_verdict,
            "parse_failures": int(parse_failures),
            "reward_formula": "outcome + debug_prm + bad_pattern",
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
    trace = _as_dict(extra.get(TRACE_KEY))
    parse_failures = _parse_failures(extra, trace)

    submit_events = [event for event in events if event.tool == "submit_solution"]
    last_submit = submit_events[-1] if submit_events else None
    acc = 1.0 if last_submit is not None and last_submit.verdict == "accepted" else 0.0

    submit_policy = str(extra.get("outcome_submit_policy") or OUTCOME_SUBMIT_POLICY)
    submit_diagnostics = _submit_diagnostics(submit_events, policy=submit_policy)
    outcome_reward, outcome_verdict, outcome_pass_rate = _compute_outcome_reward(
        submit_events,
        policy=submit_policy,
    )
    debug_prm, debug_signals = _compute_debug_prm(events)
    if not submit_events and debug_prm > 0:
        _add_signal(debug_signals, "no_submit_positive_debug_clamped", -debug_prm)
        debug_prm = 0.0
    bad_pattern, bad_patterns = _compute_bad_pattern_penalty(
        events,
        trace=trace,
        parse_failures=parse_failures,
    )

    accepted_for_outcome = outcome_verdict == "accepted"
    final = outcome_reward + debug_prm + bad_pattern
    if not accepted_for_outcome:
        final = min(final, NON_ACCEPTED_MAX)
    final = _stable_float(_clamp(final, FINAL_MIN, FINAL_MAX))

    breakdown = _breakdown_json(
        outcome_reward=outcome_reward,
        debug_prm=debug_prm,
        bad_pattern=bad_pattern,
        final=final,
        acc=acc,
        outcome_verdict=outcome_verdict,
        outcome_pass_rate=outcome_pass_rate,
        submit_policy=submit_policy,
        debug_signals=debug_signals,
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
        "debug_prm": _stable_float(debug_prm),
        "bad_pattern": _stable_float(bad_pattern),
    }
