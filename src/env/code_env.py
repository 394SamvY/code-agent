"""
OJ-like code environment.

CodeEnvironment manages one CodeProblem episode with two actions:
public test debugging and private/full judge submission.
"""

from __future__ import annotations

from typing import Any

from src.data.dataset import CodeProblem

from .tools import (
    TOOL_EXECUTORS,
    VERDICT_ACCEPTED,
    format_judge_observation,
    run_oj_judge,
)


class CodeEnvironment:
    """Interactive environment for one OJ-like programming problem."""

    def __init__(
        self,
        problem: CodeProblem,
        timeout: int | float | None = None,
        max_submissions: int = 5,
    ):
        self.problem = problem
        self.timeout = timeout if timeout is not None else problem.time_limit_seconds or 5
        self.max_submissions = max_submissions

        self._state: dict[str, Any] = {
            "current_code": "",
            "public_tests": problem.public_tests,
            "private_tests": problem.private_tests,
            "timeout": self.timeout,
            "max_submissions": max_submissions,
            "submission_count": 0,
            "tool_history": [],
            "public_results_history": [],
            "submission_history": [],
            "last_result": None,
        }

    def run_public_tests(self, code: str) -> dict[str, Any]:
        """Run code against public tests and return the structured result."""
        self._state["current_code"] = code
        result = run_oj_judge(
            code=code,
            tests=self.problem.public_tests,
            action="run_public_tests",
            timeout=self.timeout,
        )
        self._state["public_results_history"].append(result)
        self._state["last_result"] = result
        return result

    def submit_solution(self, code: str) -> dict[str, Any]:
        """Run code against private tests, respecting the submission limit."""
        observation = self.execute_tool("submit_solution", code=code)
        result = self._state["last_result"]
        if result is None:
            raise RuntimeError(f"submit_solution failed without result: {observation}")
        return result

    def execute_tool(self, tool_name: str, **kwargs: Any) -> str:
        """Execute one OJ tool and return observation text."""
        executor = TOOL_EXECUTORS.get(tool_name)
        if executor is None:
            return f"Error: Unknown tool '{tool_name}'. Available tools: {list(TOOL_EXECUTORS.keys())}"

        try:
            observation = executor(self._state, **kwargs)
        except TypeError as e:
            return f"Error calling {tool_name}: {e}"

        self._state["tool_history"].append(tool_name)
        return observation

    def format_result(self, result: dict[str, Any], include_all_failures: bool = False) -> str:
        return format_judge_observation(result, include_all_failures=include_all_failures)

    @property
    def current_code(self) -> str:
        return self._state["current_code"]

    @property
    def tool_history(self) -> list[str]:
        return self._state["tool_history"]

    @property
    def public_results_history(self) -> list[dict[str, Any]]:
        return self._state["public_results_history"]

    @property
    def submission_history(self) -> list[dict[str, Any]]:
        return self._state["submission_history"]

    @property
    def last_result(self) -> dict[str, Any] | None:
        return self._state["last_result"]

    @property
    def is_accepted(self) -> bool:
        if not self.submission_history:
            return False
        return self.submission_history[-1]["verdict"] == VERDICT_ACCEPTED

    @property
    def last_public_pass_ratio(self) -> float:
        if not self.public_results_history:
            return 0.0
        last = self.public_results_history[-1]
        return last["passed"] / last["total"] if last["total"] else 0.0

    @property
    def last_submission_verdict(self) -> str | None:
        if not self.submission_history:
            return None
        return self.submission_history[-1]["verdict"]

    def reset(self) -> None:
        """Reset mutable episode state for the same problem."""
        self._state["current_code"] = ""
        self._state["submission_count"] = 0
        self._state["tool_history"] = []
        self._state["public_results_history"] = []
        self._state["submission_history"] = []
        self._state["last_result"] = None
