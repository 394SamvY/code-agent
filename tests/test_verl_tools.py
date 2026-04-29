"""
OJ-like env/tool protocol local tests.

These tests avoid importing verl. They validate the shared Env + Tools logic
that the verl BaseTool adapters call.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import CodeProblem, OJTestCase
from src.env.code_env import CodeEnvironment
from src.env.tools import (
    VERDICT_ACCEPTED,
    VERDICT_NO_TESTS,
    VERDICT_PUBLIC_TEST_LIMIT_EXCEEDED,
    VERDICT_RUNTIME_ERROR,
    VERDICT_SUBMISSION_LIMIT_EXCEEDED,
    VERDICT_SYNTAX_ERROR,
    VERDICT_TIME_LIMIT_EXCEEDED,
    VERDICT_WRONG_ANSWER,
    reward_for_result,
)


def _problem(
    public_tests: list[OJTestCase] | None = None,
    private_tests: list[OJTestCase] | None = None,
    timeout: float = 2.0,
) -> CodeProblem:
    return CodeProblem(
        task_id="codecontests/local",
        dataset="codecontests",
        problem_statement="Read an integer and print it plus one.",
        public_tests=public_tests if public_tests is not None else [OJTestCase("1\n", "2\n")],
        private_tests=private_tests if private_tests is not None else [OJTestCase("41\n", "42\n")],
        time_limit_seconds=timeout,
    )


def test_public_tests_accept_correct_stdio_program():
    env = CodeEnvironment(_problem())
    code = "import sys\nx = int(sys.stdin.read())\nprint(x + 1)"

    observation = env.execute_tool("run_public_tests", code=code)

    assert "run_public_tests: accepted. 1/1 tests passed." in observation
    assert env.max_submissions == 5
    assert env.public_results_history[-1]["verdict"] == VERDICT_ACCEPTED
    assert env.current_code == code
    assert env.tool_history == ["run_public_tests"]

    print("[PASS] test_public_tests_accept_correct_stdio_program")


def test_wrong_answer_includes_case_detail():
    env = CodeEnvironment(_problem())
    code = "import sys\nx = int(sys.stdin.read())\nprint(x)"

    observation = env.execute_tool("run_public_tests", code=code)
    result = env.public_results_history[-1]

    assert result["verdict"] == VERDICT_WRONG_ANSWER
    assert "Case 1 FAILED (wrong_answer)" in observation
    assert "Input:\n1" in observation
    assert "Expected:\n2" in observation
    assert "Stdout:\n1" in observation

    print("[PASS] test_wrong_answer_includes_case_detail")


def test_public_tests_stop_at_first_failed_case():
    env = CodeEnvironment(
        _problem(
            public_tests=[
                OJTestCase("1\n", "2\n"),
                OJTestCase("2\n", "3\n"),
            ]
        )
    )

    observation = env.execute_tool("run_public_tests", code="print(0)")
    result = env.public_results_history[-1]

    assert result["verdict"] == VERDICT_WRONG_ANSWER
    assert result["passed"] == 0
    assert result["total"] == 2
    assert len(result["tests"]) == 1
    assert result["stopped_early"] is True
    assert "Input:\n1" in observation
    assert "Input:\n2" not in observation

    print("[PASS] test_public_tests_stop_at_first_failed_case")


def test_runtime_error_verdict():
    env = CodeEnvironment(_problem())

    env.execute_tool("run_public_tests", code="raise RuntimeError('boom')")
    result = env.public_results_history[-1]

    assert result["verdict"] == VERDICT_RUNTIME_ERROR
    assert "RuntimeError: boom" in result["first_failed"]["stderr"]

    print("[PASS] test_runtime_error_verdict")


def test_time_limit_verdict():
    env = CodeEnvironment(_problem(timeout=0.2))

    env.execute_tool("run_public_tests", code="while True:\n    pass")
    result = env.public_results_history[-1]

    assert result["verdict"] == VERDICT_TIME_LIMIT_EXCEEDED
    assert result["first_failed"]["timed_out"] is True

    print("[PASS] test_time_limit_verdict")


def test_syntax_error_verdict():
    env = CodeEnvironment(_problem())

    observation = env.execute_tool("run_public_tests", code="def broken(:\n    pass")
    result = env.public_results_history[-1]

    assert result["verdict"] == VERDICT_SYNTAX_ERROR
    assert "Syntax error" in observation

    print("[PASS] test_syntax_error_verdict")


def test_no_tests_verdict():
    env = CodeEnvironment(_problem(public_tests=[], private_tests=[]))

    observation = env.execute_tool("run_public_tests", code="print('ok')")
    result = env.public_results_history[-1]

    assert result["verdict"] == VERDICT_NO_TESTS
    assert "No tests are available" in observation

    print("[PASS] test_no_tests_verdict")


def test_public_tests_can_run_multiple_times():
    env = CodeEnvironment(_problem())

    env.execute_tool("run_public_tests", code="print(0)")
    env.execute_tool("run_public_tests", code="print(2)")

    assert len(env.public_results_history) == 2
    assert env.public_results_history[0]["verdict"] == VERDICT_WRONG_ANSWER
    assert env.public_results_history[1]["verdict"] == VERDICT_ACCEPTED
    assert env.last_public_pass_ratio == 1.0

    print("[PASS] test_public_tests_can_run_multiple_times")


def test_public_test_call_limit_guides_submit_without_consuming_submission():
    env = CodeEnvironment(_problem(), max_public_test_calls=2)

    env.execute_tool("run_public_tests", code="print(0)")
    env.execute_tool("run_public_tests", code="print(0)")
    observation = env.execute_tool("run_public_tests", code="print(0)")
    result = env.public_results_history[-1]

    assert result["verdict"] == VERDICT_PUBLIC_TEST_LIMIT_EXCEEDED
    assert result["public_test_call_count"] == 2
    assert env._state["submission_count"] == 0
    assert "Public test call limit reached" in observation
    assert "submit_solution" in observation

    print("[PASS] test_public_test_call_limit_guides_submit_without_consuming_submission")


def test_submit_solution_uses_private_tests_and_accepts():
    env = CodeEnvironment(_problem())
    code = "import sys\nx = int(sys.stdin.read())\nprint(x + 1)"

    result = env.submit_solution(code)

    assert result["verdict"] == VERDICT_ACCEPTED
    assert env.is_accepted is True
    assert env.submission_history[-1]["passed"] == 1

    print("[PASS] test_submit_solution_uses_private_tests_and_accepts")


def test_submit_solution_limit():
    env = CodeEnvironment(_problem(), max_submissions=1)

    env.execute_tool("submit_solution", code="print(0)")
    observation = env.execute_tool("submit_solution", code="print(0)")

    assert env.submission_history[-1]["verdict"] == VERDICT_SUBMISSION_LIMIT_EXCEEDED
    assert "submission_limit_exceeded" in observation

    print("[PASS] test_submit_solution_limit")


def test_submit_solution_stops_at_first_failed_private_case():
    env = CodeEnvironment(
        _problem(
            private_tests=[
                OJTestCase("10\n", "11\n"),
                OJTestCase("20\n", "21\n"),
            ]
        )
    )

    observation = env.execute_tool("submit_solution", code="print(0)")
    result = env.submission_history[-1]

    assert result["verdict"] == VERDICT_WRONG_ANSWER
    assert len(result["tests"]) == 1
    assert result["stopped_early"] is True
    assert result["first_failed"]["input"] == "10\n"
    assert "Input:\n10" in observation
    assert "Input:\n20" not in observation

    print("[PASS] test_submit_solution_stops_at_first_failed_private_case")


def test_reward_policy():
    env = CodeEnvironment(_problem())

    env.execute_tool("run_public_tests", code="print(2)")
    public_reward = reward_for_result(env.public_results_history[-1])

    env.execute_tool("submit_solution", code="print(42)")
    submit_reward = reward_for_result(env.submission_history[-1])

    assert public_reward == 0.0
    assert submit_reward == 1.0

    print("[PASS] test_reward_policy")


def test_unknown_tool():
    env = CodeEnvironment(_problem())

    result = env.execute_tool("execute_code", code="print(1)")

    assert "Error: Unknown tool" in result

    print("[PASS] test_unknown_tool")


if __name__ == "__main__":
    test_public_tests_accept_correct_stdio_program()
    test_wrong_answer_includes_case_detail()
    test_public_tests_stop_at_first_failed_case()
    test_runtime_error_verdict()
    test_time_limit_verdict()
    test_syntax_error_verdict()
    test_no_tests_verdict()
    test_public_tests_can_run_multiple_times()
    test_public_test_call_limit_guides_submit_without_consuming_submission()
    test_submit_solution_uses_private_tests_and_accepts()
    test_submit_solution_limit()
    test_submit_solution_stops_at_first_failed_private_case()
    test_reward_policy()
    test_unknown_tool()
    print("\nAll tests passed!")
