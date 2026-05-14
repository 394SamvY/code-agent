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
        0.32,
        acc=0.0,
    )
    bd = _breakdown(result)
    assert bd["outcome_reward"] == 0.32
    assert bd["outcome_verdict"] == "wrong_answer"

    result = _assert_score(
        [
            _event("submit_solution", "accepted", 10, 10, code="print(1)"),
            _event("submit_solution", "wrong_answer", 4, 10, code="print(2)"),
        ],
        -0.24,
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
    assert bd["bad_patterns"]["accepted_then_later_failed_submit"] == -0.3


def test_no_submit_bad_pattern_penalty():
    _assert_score([], -0.3, acc=0.0)
    _assert_score([_event("run_public_tests", "accepted", 3, 3)], -0.4, acc=0.0)


def test_response_truncated_bad_pattern_penalty():
    result = _assert_score(
        [_event("submit_solution", "accepted", 10, 10)],
        0.8,
        acc=1.0,
        code_agent_trace={"terminal_reason": "response_length_exceeded"},
    )
    bd = _breakdown(result)
    assert bd["bad_patterns"]["response_truncated"] == -0.2

    result = _assert_score(
        [],
        -0.4,
        acc=0.0,
        code_agent_trace={"terminal_reason": "response_length_exceeded"},
    )
    bd = _breakdown(result)
    assert bd["bad_pattern"] == -0.4
    assert bd["bad_patterns"]["no_submit"] == -0.3
    assert bd["bad_patterns"]["response_truncated"] == -0.2
    assert bd["bad_patterns"]["truncated_no_submit"] == -0.1


def test_debug_prm_rewards_feedback_conditioned_improvement():
    old = "a=[]\nprint(a[0])"
    new = "a=[]\nif len(a) > 0:\n    print(a[0])\nelse:\n    print(0)"
    events = [
        _event(
            "run_public_tests",
            "runtime_error",
            0,
            2,
            code=old,
            first_failed={"stderr": "IndexError: list index out of range"},
            error_kind="index_error",
        ),
        _event("run_public_tests", "wrong_answer", 1, 2, code=new),
        _event("submit_solution", "wrong_answer", 0, 10, code=new),
    ]
    result = _assert_score(events, -0.055, acc=0.0)
    signals = _breakdown(result)["debug_signals"]
    assert signals["state_rank_improved"] == 0.015
    assert "pass_rate_improved" not in signals
    assert signals["feedback_aligned_not_worse"] == 0.03


def test_weak_wrong_answer_logic_change_bonus_is_small():
    events = [
        _event("run_public_tests", "wrong_answer", 0, 2, code="print(0)"),
        _event("run_public_tests", "wrong_answer", 1, 2, code="print(1)"),
        _event("submit_solution", "wrong_answer", 0, 10, code="print(1)"),
    ]
    result = _assert_score(events, -0.08, acc=0.0)
    bd = _breakdown(result)
    assert bd["debug_prm"] == 0.02
    assert bd["debug_signals"]["pass_rate_improved"] == 0.015
    assert bd["debug_signals"]["wrong_answer_logic_changed_not_worse"] == 0.005


def test_debug_prm_progress_only_counts_new_best_state():
    events = [
        _event("run_public_tests", "syntax_error", 0, 2, code="print("),
        _event("run_public_tests", "wrong_answer", 0, 2, code="print(0)"),
        _event("run_public_tests", "syntax_error", 0, 2, code="print("),
        _event("run_public_tests", "wrong_answer", 0, 2, code="print(1)"),
        _event("submit_solution", "wrong_answer", 0, 10, code="print(1)"),
    ]
    result = _score(events)
    bd = _breakdown(result)
    assert bd["debug_signals"]["state_rank_improved"] == 0.03
    assert "wrong_answer_logic_changed_not_worse" not in bd["debug_signals"]
    assert bd["debug_prm"] == 0.03


def test_no_submit_caps_positive_debug_prm_to_zero():
    events = [
        _event("run_public_tests", "wrong_answer", 0, 2, code="print(0)"),
        _event("run_public_tests", "accepted", 2, 2, code="print(1)"),
    ]
    result = _assert_score(events, -0.4, acc=0.0)
    bd = _breakdown(result)
    assert bd["debug_prm"] == 0.0
    assert bd["bad_patterns"]["no_submit"] == -0.3
    assert bd["bad_patterns"]["public_acc_no_submit"] == -0.2


def test_public_to_submit_does_not_get_debug_bonus():
    code = "print(0)"
    result = _assert_score(
        [
            _event("run_public_tests", "wrong_answer", 0, 2, code=code),
            _event("submit_solution", "wrong_answer", 0, 10, code=code),
        ],
        -0.1,
        acc=0.0,
    )
    bd = _breakdown(result)
    assert bd["debug_prm"] == 0.0
    assert bd["debug_signals"] == {}
    assert bd["bad_patterns"]["public_fail_same_code_submit"] == -0.1


def test_submit_debug_bonus_is_weak_and_capped():
    events = [
        _event("submit_solution", "wrong_answer", 1, 10, code="print(0)"),
        _event("submit_solution", "wrong_answer", 8, 10, code="print(1)"),
        _event("submit_solution", "accepted", 10, 10, code="print(2)"),
    ]
    result = _assert_score(events, 0.9738, acc=1.0)
    bd = _breakdown(result)
    assert bd["debug_prm"] <= 0.03
    assert bd["bad_patterns"]["too_many_submits"] == -0.04


def test_duplicate_and_tool_count_bad_patterns():
    code = "print(0)"
    events = [
        *[_event("run_public_tests", "wrong_answer", 0, 1, code=code) for _ in range(7)],
        *[_event("submit_solution", "wrong_answer", 0, 10, code=code) for _ in range(4)],
    ]
    result = _assert_score(events, -0.5, acc=0.0)
    bd = _breakdown(result)
    assert bd["bad_pattern"] == -0.4
    assert bd["bad_patterns"]["duplicate_public_code"] == -0.1
    assert bd["bad_patterns"]["duplicate_submit_code"] == -0.15
    assert bd["bad_patterns"]["too_many_public_tests"] == -0.04
    assert bd["bad_patterns"]["too_many_submits"] == -0.08


def test_accepted_then_tool_and_long_text_penalties():
    result = _assert_score(
        [
            _event("submit_solution", "accepted", 10, 10, code="print(1)"),
            _event("run_public_tests", "accepted", 3, 3, code="print(1)"),
        ],
        0.75,
        acc=1.0,
        code_agent_trace={"assistant_chars_after_submit_accepted": 1601},
    )
    bd = _breakdown(result)
    assert bd["bad_patterns"]["accepted_then_tool_call"] == -0.2
    assert bd["bad_patterns"]["accepted_then_long_text"] == -0.05


def test_parse_failure_penalty_comes_from_trace():
    result = _assert_score(
        [_event("submit_solution", "accepted", 10, 10)],
        0.9,
        acc=1.0,
        code_agent_trace={"parse_failures": 2},
    )
    assert _breakdown(result)["bad_patterns"]["malformed_tool_call"] == -0.1


if __name__ == "__main__":
    test_compute_score_requires_structured_events()
    test_outcome_reward_uses_last_submit_by_default()
    test_any_ac_else_best_policy_is_available_but_acc_stays_last_submit()
    test_no_submit_bad_pattern_penalty()
    test_response_truncated_bad_pattern_penalty()
    test_debug_prm_rewards_feedback_conditioned_improvement()
    test_weak_wrong_answer_logic_change_bonus_is_small()
    test_debug_prm_progress_only_counts_new_best_state()
    test_no_submit_caps_positive_debug_prm_to_zero()
    test_public_to_submit_does_not_get_debug_bonus()
    test_submit_debug_bonus_is_weak_and_capped()
    test_duplicate_and_tool_count_bad_patterns()
    test_accepted_then_tool_and_long_text_penalties()
    test_parse_failure_penalty_comes_from_trace()
