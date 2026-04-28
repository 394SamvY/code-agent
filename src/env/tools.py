"""
OJ-like agent tools
===================

本文件定义 OJ-like v1 的工具协议和 judge 逻辑。

它处在环境的中间层：

- `CodeEnvironment` 负责管理一道题的一次 episode 状态；
- 本文件负责工具语义、判题、observation 格式和 reward；
- `sandbox.py` 负责真正启动 Python 子进程执行代码。

v1 只暴露两个 actions：

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

# 环境语义上的终止 verdict。只有 accepted 或提交次数耗尽会结束一轮正式交互。
TERMINAL_VERDICTS = {
    VERDICT_ACCEPTED,
    VERDICT_SUBMISSION_LIMIT_EXCEEDED,
}


# 这是提供给模型/verl 的公开工具 schema。字段名和描述应与真实工具行为保持一致，
# 否则模型看到的接口和本地执行语义会漂移。
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
    """Normalize stdout/expected text before exact comparison.

    v1 规则很简单：

    - 统一换行符
    - 去掉末尾空白
    - 然后做精确字符串比较

    这里故意不做 special judge、数值容忍或更激进的空白忽略。
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip()


def parse_oj_tests(raw: Any) -> list[OJTestCase]:
    """Parse tests from CodeProblem objects, JSON strings, or plain dicts.

    这个 helper 主要服务 verl / tool adapter 场景：有时 tests 已经是
    `list[OJTestCase]`，有时来自 parquet/JSON，需要恢复成统一结构。
    """
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
    """Serialize `OJTestCase` into JSON-friendly dicts for parquet / tool kwargs."""
    return [{"input": test.input, "output": test.output} for test in tests]


def _syntax_error_result(action: str, tests: list[OJTestCase], error: SyntaxError) -> dict[str, Any]:
    """Build the canonical judge result for syntax errors.

    语法错误在真正执行前就能确定，因此不会进入 `sandbox.execute_stdio()`。
    为了让 observation 和日志结构保持一致，这里仍返回标准的 judge result schema。
    """
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
    """Build the canonical judge result for actions without available tests."""
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
    """Run code on stdin/stdout tests and return the structured judge result.

    这是本环境真正的“判题核心”：

    1. 先用 `compile()` 做语法检查
    2. 再逐个测试调用 `sandbox.execute_stdio()`
    3. 根据 timeout / returncode / stdout 比较得到 per-case verdict
    4. 汇总出 judge-level verdict、passed 数、first_failed 和完整 tests 列表

    注意：它只负责“给定代码 + 给定 tests 怎么判”，不负责 submission limit、
    reward、tool history 或 observation 文本，这些都在更外层处理。
    """
    try:
        compile(code, "<solution>", "exec")
    except SyntaxError as e:
        return _syntax_error_result(action, tests, e)

    if not tests:
        return _no_tests_result(action)

    case_results: list[dict[str, Any]] = []
    passed = 0

    for index, test in enumerate(tests, start=1):
        # 真正的代码执行在 sandbox 层。这里拿回 stdout/stderr/returncode 后，
        # 再用环境协议映射成 OJ verdict。
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


def _clip(text: Any, limit: int = 2000) -> str:
    """Clip long stdout/stderr blocks before showing them to the model."""
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]"


def _format_case(case: dict[str, Any]) -> str:
    """Render one failed case into human-readable observation text."""
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
    """Format a structured judge result into the tool observation text.

    结构化 result 是主接口；这个函数只负责把它压成模型可读文本。

    `run_public_tests` 会展开所有失败 public case，便于调试；
    `submit_solution` 默认只展开第一条失败 private case，避免反馈过量。
    """
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

    TODO: reward 调优方向（可做的文章很多）：
      - public tests 全部通过时给少量 shaping reward（如 0.05），引导模型先对齐公开用例
      - submit 失败时根据失败比例阶梯给分：0~0.2 太粗，可以更细粒度
      - 区分 verdict 类型：runtime_error 比 wrong_answer 更严重，可以不同惩罚
      - 解空间探索奖励：调用工具次数、代码修改幅度、尝试不同算法
      - 最终 reward 聚合策略：目前是 max(tool_rewards)，可以尝试 avg、最后一次、递减折扣
      - 代码风格/简洁度：字符数、运行时间作为辅助奖励
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
    """Execute the public-test tool against the current environment state.

    这个函数既被 `CodeEnvironment.execute_tool()` 调用，也被 verl tool adapter 复用。
    它负责更新共享状态，并返回 observation text 给上层 agent。
    """
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
    """Execute the formal submission tool against the current environment state.

    与 `tool_run_public_tests` 的关键区别：

    - 会检查并消耗 `submission_count`
    - 使用 `private_tests`
    - observation 默认只展开第一条失败 private case
    """
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

    # 提交次数只有在真正进入判题前才增加；public tests 不会触碰这个计数器。
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


# 工具名到 verl BaseTool 类的映射。这样 `tool_config.yaml` 可以直接从本文件生成，
# 避免 schema 和适配类路径手写漂移。
_TOOL_NAME_TO_VERL_CLASS: dict[str, str] = {
    "run_public_tests": "src.verl_tools.oj_tools.RunPublicTestsTool",
    "submit_solution": "src.verl_tools.oj_tools.SubmitSolutionTool",
}


def generate_tool_config_yaml() -> str:
    """Generate verl `tool_config.yaml` from the in-code tool schemas."""
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

    import yaml
    return yaml.dump({"tools": tools}, default_flow_style=False, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    from pathlib import Path

    output = Path(__file__).resolve().parents[2] / "configs" / "verl" / "tool_config.yaml"
    content = generate_tool_config_yaml()
    output.write_text(content)
    print(f"Generated {output}")
