"""Focused tests for the code-agent verl AgentLoop wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.verl_agent_loop import CodeAgentToolAgentLoop


def _agent() -> CodeAgentToolAgentLoop:
    return object.__new__(CodeAgentToolAgentLoop)


def _agent_data() -> SimpleNamespace:
    return SimpleNamespace(
        extra_fields={},
        tools_kwargs={
            "run_public_tests": {"create_kwargs": {"max_public_test_calls": 2}},
            "submit_solution": {"create_kwargs": {"max_submissions": 3}},
        },
    )


def test_max_tool_calls_uses_oj_limits():
    agent = _agent()
    data = _agent_data()

    assert agent._max_tool_calls(data) == 7


def test_trace_initializes_stable_extra_field_keys():
    agent = _agent()
    data = _agent_data()

    agent._trace(data)

    assert data.extra_fields["code_agent_terminal_reason"] is None
    assert data.extra_fields["code_agent_parse_failures"] == 0
    assert data.extra_fields["code_agent_tool_tail_chars"] == 0
    assert data.extra_fields["code_agent_trace"]["num_tool_calls"] == 0


def test_public_limit_result_does_not_mark_terminal():
    agent = _agent()
    data = _agent_data()

    agent._record_tool_result(
        data,
        {
            "action": "run_public_tests",
            "verdict": "public_test_limit_exceeded",
            "public_test_call_count": 2,
        },
    )

    assert getattr(data, "code_agent_terminal", False) is False
    assert data.extra_fields["code_agent_terminal_reason"] is None
    assert data.extra_fields["code_agent_trace"]["terminal_reason"] is None
    assert data.extra_fields["code_agent_trace"]["last_verdict"] == "public_test_limit_exceeded"
    assert data.extra_fields["code_agent_trace"]["num_tool_calls"] == 1


def test_public_test_accepted_does_not_mark_terminal():
    agent = _agent()
    data = _agent_data()

    agent._record_tool_result(
        data,
        {
            "action": "run_public_tests",
            "verdict": "accepted",
            "public_test_call_count": 1,
        },
    )

    assert getattr(data, "code_agent_terminal", False) is False
    assert data.extra_fields["code_agent_terminal_reason"] is None
    assert data.extra_fields["code_agent_trace"]["terminal_reason"] is None
    assert data.extra_fields["code_agent_trace"]["last_verdict"] == "accepted"


def test_submit_accepted_does_not_mark_terminal():
    agent = _agent()
    data = _agent_data()

    agent._record_tool_result(
        data,
        {
            "action": "submit_solution",
            "verdict": "accepted",
            "submission_count": 1,
        },
    )

    assert getattr(data, "code_agent_terminal", False) is False
    assert data.extra_fields["code_agent_terminal_reason"] is None
    assert data.extra_fields["code_agent_trace"]["terminal_reason"] is None
    assert data.extra_fields["code_agent_trace"]["submission_count"] == 1


def test_no_tool_call_marks_normal_terminal_reason():
    agent = _agent()
    data = _agent_data()

    agent._record_no_tool_call_termination(data)

    assert data.code_agent_terminal is True
    assert data.code_agent_terminal_reason == "no_tool_call"
    assert data.extra_fields["code_agent_trace"]["terminal_reason"] == "no_tool_call"


def test_hard_cap_marks_terminal():
    agent = _agent()
    data = _agent_data()

    for _ in range(7):
        agent._record_tool_result(data, {"action": "run_public_tests", "verdict": "wrong_answer"})

    assert agent._should_terminate(data) is True
    assert data.code_agent_terminal_reason == "tool_call_limit_exhausted"


if __name__ == "__main__":
    test_max_tool_calls_uses_oj_limits()
    test_trace_initializes_stable_extra_field_keys()
    test_public_limit_result_does_not_mark_terminal()
    test_public_test_accepted_does_not_mark_terminal()
    test_submit_accepted_does_not_mark_terminal()
    test_no_tool_call_marks_normal_terminal_reason()
    test_hard_cap_marks_terminal()
    print("\nAll tests passed!")
