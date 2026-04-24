"""
OJ-like agent tools
===================

v1 exposes two actions:

- run_public_tests: debug against public stdin/stdout cases.
- submit_solution: run the private/full judge.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from src.data.dataset import OJTestCase

from .sandbox import execute_stdio


VERDICT_ACCEPTED = "accepted"
VERDICT_WRONG_ANSWER = "wrong_answer"
VERDICT_RUNTIME_ERROR = "runtime_error"
VERDICT_TIME_LIMIT_EXCEEDED = "time_limit_exceeded"
VERDICT_SYNTAX_ERROR = "syntax_error"
VERDICT_NO_TESTS = "no_tests"
VERDICT_SUBMISSION_LIMIT_EXCEEDED = "submission_limit_exceeded"

TERMINAL_VERDICTS = {
    VERDICT_ACCEPTED,
    VERDICT_SUBMISSION_LIMIT_EXCEEDED,
}


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_public_tests",
            "description": (
                "Run the complete Python stdin/stdout program on the public tests. "
                "Use this to debug before submitting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The complete Python program that reads stdin and writes stdout.",
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
            "description": (
                "Submit the complete Python stdin/stdout program to the full judge. "
                "This runs private tests and consumes one submission attempt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The complete Python program that reads stdin and writes stdout.",
                    }
                },
                "required": ["code"],
            },
        },
    },
]


def normalize_output(text: str) -> str:
    """Normalize stdout/expected text before exact comparison."""
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip()


def parse_oj_tests(raw: Any) -> list[OJTestCase]:
    """Parse tests from CodeProblem objects, JSON strings, or plain dicts."""
    if raw in (None, "", []):
        return []

    if isinstance(raw, str):
        raw = json.loads(raw)

    tests: list[OJTestCase] = []
    for item in raw:
        if isinstance(item, OJTestCase):
            tests.append(item)
        elif isinstance(item, dict):
            tests.append(
                OJTestCase(
                    input=str(item.get("input", "")),
                    output=str(item.get("output", "")),
                )
            )
        else:
            raise TypeError(f"Unsupported OJ test case item: {type(item)!r}")
    return tests


def serialize_oj_tests(tests: list[OJTestCase]) -> list[dict[str, str]]:
    return [{"input": test.input, "output": test.output} for test in tests]


def _syntax_error_result(action: str, tests: list[OJTestCase], error: SyntaxError) -> dict[str, Any]:
    first_input = tests[0].input if tests else ""
    first_expected = tests[0].output if tests else ""
    first_failed = {
        "index": 1,
        "passed": False,
        "verdict": VERDICT_SYNTAX_ERROR,
        "input": first_input,
        "expected": first_expected,
        "stdout": "",
        "stderr": f"Syntax error at line {error.lineno}: {error.msg}",
        "returncode": -1,
        "timed_out": False,
        "runtime_seconds": 0.0,
    }
    return {
        "action": action,
        "verdict": VERDICT_SYNTAX_ERROR,
        "passed": 0,
        "total": len(tests),
        "first_failed": first_failed,
        "tests": [first_failed],
    }


def _no_tests_result(action: str) -> dict[str, Any]:
    return {
        "action": action,
        "verdict": VERDICT_NO_TESTS,
        "passed": 0,
        "total": 0,
        "first_failed": None,
        "tests": [],
    }


def run_oj_judge(
    code: str,
    tests: list[OJTestCase],
    action: str,
    timeout: int | float = 5,
) -> dict[str, Any]:
    """Run code on stdin/stdout tests and return the structured judge result."""
    try:
        compile(code, "<solution>", "exec")
    except SyntaxError as e:
        return _syntax_error_result(action, tests, e)

    if not tests:
        return _no_tests_result(action)

    case_results: list[dict[str, Any]] = []
    passed = 0

    for index, test in enumerate(tests, start=1):
        result = execute_stdio(code, stdin=test.input, timeout=timeout)
        if result.timed_out:
            verdict = VERDICT_TIME_LIMIT_EXCEEDED
            is_passed = False
        elif result.returncode != 0:
            verdict = VERDICT_RUNTIME_ERROR
            is_passed = False
        elif normalize_output(result.stdout) == normalize_output(test.output):
            verdict = VERDICT_ACCEPTED
            is_passed = True
        else:
            verdict = VERDICT_WRONG_ANSWER
            is_passed = False

        if is_passed:
            passed += 1

        case_results.append(
            {
                "index": index,
                "passed": is_passed,
                "verdict": verdict,
                "input": test.input,
                "expected": test.output,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "runtime_seconds": result.runtime_seconds,
            }
        )

    first_failed = next((case for case in case_results if not case["passed"]), None)
    verdict = VERDICT_ACCEPTED if first_failed is None else first_failed["verdict"]

    return {
        "action": action,
        "verdict": verdict,
        "passed": passed,
        "total": len(tests),
        "first_failed": first_failed,
        "tests": case_results,
    }


def _clip(text: Any, limit: int = 1000) -> str:
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]"


def _format_case(case: dict[str, Any]) -> str:
    return (
        f"Case {case['index']} FAILED ({case['verdict']}):\n"
        f"Input:\n{_clip(case.get('input'))}\n"
        f"Expected:\n{_clip(case.get('expected'))}\n"
        f"Stdout:\n{_clip(case.get('stdout'))}\n"
        f"Stderr:\n{_clip(case.get('stderr'))}"
    )


def format_judge_observation(
    result: dict[str, Any],
    *,
    include_all_failures: bool,
) -> str:
    """Format a structured judge result into the tool observation text."""
    action = result["action"]
    verdict = result["verdict"]
    passed = result["passed"]
    total = result["total"]

    lines = [f"{action}: {verdict}. {passed}/{total} tests passed."]
    if verdict == VERDICT_ACCEPTED:
        lines.append("All tests passed.")
        return "\n".join(lines)
    if verdict == VERDICT_NO_TESTS:
        lines.append("No tests are available for this action.")
        return "\n".join(lines)

    failures = [case for case in result["tests"] if not case["passed"]]
    if not include_all_failures and result.get("first_failed") is not None:
        failures = [result["first_failed"]]

    if failures:
        lines.append("")
        lines.append("\n---\n".join(_format_case(case) for case in failures))
    return "\n".join(lines)


def reward_for_result(result: dict[str, Any]) -> float:
    """Submit-dominant reward policy.

    Public tests are diagnostic only; they do not contribute reward.
    """
    total = result["total"]
    pass_rate = result["passed"] / total if total else 0.0
    action = result["action"]
    verdict = result["verdict"]

    if action == "run_public_tests":
        return 0.0
    if action == "submit_solution":
        if verdict == VERDICT_ACCEPTED:
            return 1.0
        if verdict in {VERDICT_WRONG_ANSWER, VERDICT_RUNTIME_ERROR, VERDICT_TIME_LIMIT_EXCEEDED}:
            return 0.2 * pass_rate
    return 0.0


def tool_run_public_tests(env_state: dict[str, Any], code: str) -> str:
    env_state["current_code"] = code
    result = run_oj_judge(
        code=code,
        tests=env_state.get("public_tests", []),
        action="run_public_tests",
        timeout=env_state.get("timeout", 5),
    )
    env_state["public_results_history"].append(result)
    env_state["last_result"] = result
    return format_judge_observation(result, include_all_failures=True)


def tool_submit_solution(env_state: dict[str, Any], code: str) -> str:
    if env_state["submission_count"] >= env_state["max_submissions"]:
        result = {
            "action": "submit_solution",
            "verdict": VERDICT_SUBMISSION_LIMIT_EXCEEDED,
            "passed": 0,
            "total": len(env_state.get("private_tests", [])),
            "first_failed": None,
            "tests": [],
        }
        env_state["submission_history"].append(result)
        env_state["last_result"] = result
        return format_judge_observation(result, include_all_failures=False)

    env_state["submission_count"] += 1
    env_state["current_code"] = code
    result = run_oj_judge(
        code=code,
        tests=env_state.get("private_tests", []),
        action="submit_solution",
        timeout=env_state.get("timeout", 5),
    )
    result["submission_count"] = env_state["submission_count"]
    result["max_submissions"] = env_state["max_submissions"]
    env_state["submission_history"].append(result)
    env_state["last_result"] = result
    return format_judge_observation(result, include_all_failures=False)


TOOL_EXECUTORS: dict[str, Callable[..., str]] = {
    "run_public_tests": tool_run_public_tests,
    "submit_solution": tool_submit_solution,
}


_TOOL_NAME_TO_VERL_CLASS: dict[str, str] = {
    "run_public_tests": "src.verl_tools.oj_tools.RunPublicTestsTool",
    "submit_solution": "src.verl_tools.oj_tools.SubmitSolutionTool",
}


def generate_tool_config_yaml() -> str:
    """Generate verl tool_config.yaml from the tool schemas."""
    tools = []
    for schema in TOOLS_SCHEMA:
        name = schema["function"]["name"]
        tools.append(
            {
                "class_name": _TOOL_NAME_TO_VERL_CLASS[name],
                "config": {"type": "native"},
                "tool_schema": schema,
            }
        )

    try:
        import yaml
    except ModuleNotFoundError:
        return _generate_tool_config_yaml_fallback(tools)
    return yaml.dump({"tools": tools}, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _generate_tool_config_yaml_fallback(tools: list[dict[str, Any]]) -> str:
    lines = ["tools:"]
    for tool in tools:
        function = tool["tool_schema"]["function"]
        code_desc = function["parameters"]["properties"]["code"]["description"]
        lines.extend(
            [
                f"- class_name: {tool['class_name']}",
                "  config:",
                f"    type: {tool['config']['type']}",
                "  tool_schema:",
                f"    type: {tool['tool_schema']['type']}",
                "    function:",
                f"      name: {function['name']}",
                f"      description: {function['description']}",
                "      parameters:",
                "        type: object",
                "        properties:",
                "          code:",
                "            type: string",
                f"            description: {code_desc}",
                "        required:",
                "        - code",
            ]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    from pathlib import Path

    output = Path(__file__).resolve().parents[2] / "configs" / "verl" / "tool_config.yaml"
    content = generate_tool_config_yaml()
    output.write_text(content)
    print(f"Generated {output}")
