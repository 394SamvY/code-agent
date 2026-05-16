import json
import math
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reward import compute_score, semantic_code_hash


def _event(tool: str, verdict: str, passed: int, total: int, code: str = "print(1)", **extra):
    event = {
        "tool": tool,
        "verdict": verdict,
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else 0.0,
        "code": code,
        "semantic_hash": semantic_code_hash(code),
        "observation": f"{tool}: {verdict}. {passed}/{total} tests passed.",
    }
    event.update(extra)
    return event


def _score(events, **extra_info):
    payload = {"code_agent_tool_events": events, **extra_info}
    return compute_score("codecontests", "ignored text", {"task_id": "dummy"}, extra_info=payload)


def _breakdown(result):
    return json.loads(result["reward_breakdown"])


def _assert_score(events, expected: float, *, acc: Optional[float] = None, **extra_info):
    result = _score(events, **extra_info)
    assert math.isclose(result["score"], expected, abs_tol=1e-9)
    assert math.isclose(result["reward"], expected, abs_tol=1e-9)
    if acc is not None:
        assert result["acc"] == acc
    assert math.isclose(_breakdown(result)["final"], expected, abs_tol=1e-9)
    return result


def test_compute_score_requires_structured_events():
    try:
        compute_score("codecontests", "submit_solution: accepted. 1/1 tests passed.", {}, extra_info={})
    except ValueError as exc:
        assert "code_agent_tool_events" in str(exc)
    else:
        raise AssertionError("compute_score should require structured tool events")


def test_outcome_reward_uses_last_submit_by_default():
    result = _assert_score(
        [_event("submit_solution", "wrong_answer", 8, 10)],
        -0.04,
        acc=0.0,
    )
    bd = _breakdown(result)
    assert bd["outcome_reward"] == -0.04
    assert bd["outcome_verdict"] == "wrong_answer"

    result = _assert_score(
        [
            _event("submit_solution", "accepted", 10, 10, code="print(1)"),
            _event("submit_solution", "wrong_answer", 4, 10, code="print(2)"),
        ],
        -0.5,
        acc=0.0,
    )
    bd = _breakdown(result)
    assert bd["acc_final"] == 0.0
    assert bd["acc_any"] == 1.0
    assert bd["best_submit_pass_rate"] == 1.0
    assert bd["last_submit_pass_rate"] == 0.4
    assert bd["outcome_submit_policy"] == "last_submit"
    assert result["acc_final"] == 0.0
    assert result["acc_any"] == 1.0
    assert result["best_submit_pass_rate"] == 1.0
    assert result["last_submit_pass_rate"] == 0.4


def test_any_ac_else_best_policy_is_available_but_acc_stays_last_submit():
    result = _assert_score(
        [
            _event("submit_solution", "accepted", 10, 10, code="print(1)"),
            _event("submit_solution", "wrong_answer", 5, 10, code="print(2)"),
        ],
        0.6,
        acc=0.0,
        outcome_submit_policy="any_ac_else_best",
    )
    bd = _breakdown(result)
    assert bd["outcome_reward"] == 1.0
    assert bd["bad_patterns"]["accepted_then_later_failed_submit"] == -0.25


def test_no_submit_bad_pattern_penalty():
    _assert_score([], -0.25, acc=0.0)
    _assert_score([_event("run_public_tests", "accepted", 3, 3)], -0.35, acc=0.0)


def test_response_truncated_bad_pattern_penalty():
    result = _assert_score(
        [_event("submit_solution", "accepted", 10, 10)],
        0.85,
        acc=1.0,
        code_agent_trace={"terminal_reason": "response_length_exceeded"},
    )
    bd = _breakdown(result)
    assert bd["bad_patterns"]["response_truncated"] == -0.15

    result = _assert_score(
        [],
        -0.45,
        acc=0.0,
        code_agent_trace={"terminal_reason": "response_length_exceeded"},
    )
    bd = _breakdown(result)
    assert bd["bad_pattern"] == -0.45
    assert bd["bad_patterns"]["no_submit"] == -0.25
    assert bd["bad_patterns"]["response_truncated"] == -0.15
    assert bd["bad_patterns"]["truncated_no_submit"] == -0.05


def test_no_submit_with_public_accepted_is_strongly_penalized():
    events = [
        _event("run_public_tests", "wrong_answer", 0, 2, code="print(0)"),
        _event("run_public_tests", "accepted", 2, 2, code="print(1)"),
    ]
    result = _assert_score(events, -0.35, acc=0.0)
    bd = _breakdown(result)
    assert bd["bad_patterns"]["no_submit"] == -0.25
    assert bd["bad_patterns"]["public_acc_no_submit"] == -0.1


def test_public_to_submit_does_not_get_debug_bonus():
    code = "print(0)"
    result = _assert_score(
        [
            _event("run_public_tests", "wrong_answer", 0, 2, code=code),
            _event("submit_solution", "wrong_answer", 0, 10, code=code),
        ],
        -0.26,
        acc=0.0,
    )
    bd = _breakdown(result)
    assert bd["bad_patterns"]["public_fail_same_code_submit"] == -0.06


def test_accepted_with_too_many_submits_is_penalized():
    events = [
        _event("submit_solution", "wrong_answer", 1, 10, code="print(0)"),
        _event("submit_solution", "wrong_answer", 8, 10, code="print(1)"),
        _event("submit_solution", "accepted", 10, 10, code="print(2)"),
    ]
    result = _assert_score(events, 0.97, acc=1.0)
    bd = _breakdown(result)
    assert bd["bad_patterns"]["too_many_submits"] == -0.03


def test_duplicate_and_tool_count_bad_patterns():
    code = "print(0)"
    events = [
        *[_event("run_public_tests", "wrong_answer", 0, 1, code=code) for _ in range(7)],
        *[_event("submit_solution", "wrong_answer", 0, 10, code=code) for _ in range(4)],
    ]
    result = _assert_score(events, -0.5, acc=0.0)
    bd = _breakdown(result)
    assert bd["bad_pattern"] == -0.47
    assert bd["bad_patterns"]["duplicate_public_code"] == -0.08
    assert bd["bad_patterns"]["duplicate_submit_code"] == -0.09
    assert bd["bad_patterns"]["too_many_public_tests"] == -0.02
    assert bd["bad_patterns"]["too_many_submits"] == -0.06


def test_accepted_then_tool_and_long_text_penalties():
    result = _assert_score(
        [
            _event("submit_solution", "accepted", 10, 10, code="print(1)"),
            _event("run_public_tests", "accepted", 3, 3, code="print(1)"),
        ],
        0.8,
        acc=1.0,
        code_agent_trace={"assistant_chars_after_submit_accepted": 1601},
    )
    bd = _breakdown(result)
    assert bd["bad_patterns"]["accepted_then_tool_call"] == -0.15
    assert bd["bad_patterns"]["accepted_then_long_text"] == -0.05


def test_repeated_empty_think_after_accepted_submit_is_penalized():
    result = _assert_score(
        [_event("submit_solution", "accepted", 10, 10)],
        1.0,
        acc=1.0,
        code_agent_trace={"empty_think_after_submit_accepted_count": 1},
    )
    assert "empty_think_after_submit_accepted" not in _breakdown(result)["bad_patterns"]

    result = _assert_score(
        [_event("submit_solution", "accepted", 10, 10)],
        0.96,
        acc=1.0,
        code_agent_trace={"empty_think_after_submit_accepted_count": 3},
    )
    bd = _breakdown(result)
    assert bd["bad_patterns"]["empty_think_after_submit_accepted"] == -0.04


def test_empty_think_after_accepted_submit_is_not_double_counted_with_tool_call():
    result = _assert_score(
        [
            _event("submit_solution", "accepted", 10, 10, code="print(1)"),
            _event("run_public_tests", "accepted", 3, 3, code="print(1)"),
        ],
        0.85,
        acc=1.0,
        code_agent_trace={
            "empty_think_after_submit_accepted_count": 3,
            "consecutive_empty_think_after_submit_accepted": True,
        },
    )
    bd = _breakdown(result)
    assert bd["bad_patterns"]["accepted_then_tool_call"] == -0.15
    assert "empty_think_after_submit_accepted" not in bd["bad_patterns"]


def test_parse_failure_penalty_comes_from_trace():
    result = _assert_score(
        [_event("submit_solution", "accepted", 10, 10)],
        0.7,
        acc=1.0,
        code_agent_trace={"parse_failures": 2},
    )
    assert _breakdown(result)["bad_patterns"]["malformed_tool_call"] == -0.3


def test_malformed_non_accepted_is_hard_failure_even_with_partial_submit():
    result = _assert_score(
        [
            _event("run_public_tests", "accepted", 2, 2, code="print(1)"),
            _event("submit_solution", "wrong_answer", 7, 10, code="print(1)"),
            _event(
                "malformed_tool_call",
                "tool_parse_error",
                0,
                0,
                code=None,
                error_kind="malformed_tool_call",
            ),
        ],
        -0.3,
        acc=0.0,
    )
    bd = _breakdown(result)
    assert bd["outcome_reward"] == -0.06
    assert bd["bad_patterns"]["malformed_tool_call"] == -0.15


def test_unknown_tool_is_recorded_and_penalized():
    result = _assert_score(
        [
            _event("run_public_tests", "wrong_answer", 0, 2, code="print(0)"),
            _event(
                "clear",
                "tool_execution_error",
                0,
                0,
                code=None,
                observation="Error when executing tool: 'clear'",
                error_kind="tool_execution_error",
            ),
        ],
        -0.45,
        acc=0.0,
    )
    bd = _breakdown(result)
    assert bd["event_count"] == 2
    assert bd["bad_patterns"]["unknown_tool"] == -0.2
    assert "tool_execution_error" not in bd["bad_patterns"]


if __name__ == "__main__":
    test_compute_score_requires_structured_events()
    test_outcome_reward_uses_last_submit_by_default()
    test_any_ac_else_best_policy_is_available_but_acc_stays_last_submit()
    test_no_submit_bad_pattern_penalty()
    test_response_truncated_bad_pattern_penalty()
    test_no_submit_with_public_accepted_is_strongly_penalized()
    test_public_to_submit_does_not_get_debug_bonus()
    test_accepted_with_too_many_submits_is_penalized()
    test_duplicate_and_tool_count_bad_patterns()
    test_accepted_then_tool_and_long_text_penalties()
    test_repeated_empty_think_after_accepted_submit_is_penalized()
    test_empty_think_after_accepted_submit_is_not_double_counted_with_tool_call()
    test_parse_failure_penalty_comes_from_trace()
    test_malformed_non_accepted_is_hard_failure_even_with_partial_submit()
    test_unknown_tool_is_recorded_and_penalized()
