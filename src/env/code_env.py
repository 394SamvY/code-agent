"""
OJ-like code environment.

`CodeEnvironment` 管理一道 `CodeProblem` 的一次交互 episode。
它不是底层 sandbox，也不是 verl tool 本身，而是本地评测 / 单元测试 / eval 复用的
轻量环境封装：

- `src.env.tools` 负责真实 tool 语义、submission limit、observation 格式和 reward。
- `src.env.sandbox` 负责启动 Python 子进程、写 stdin、捕获 stdout/stderr。
- 本文件负责把一道题的 tests、timeout、提交次数和历史记录组织成一个 episode state。
"""

from __future__ import annotations

from typing import Any

from src.data.dataset import CodeProblem

from .tools import (
    TOOL_EXECUTORS,
    VERDICT_ACCEPTED,
    format_judge_observation,
)


class CodeEnvironment:
    """Interactive environment for one OJ-like programming problem.

    一个实例只对应一道题。agent 可以多次调用 `run_public_tests` 调试，也可以调用
    `submit_solution` 正式提交。`submit_solution` 受 `max_submissions` 限制，
    `run_public_tests` 不消耗提交次数。
    """

    def __init__(
        self,
        problem: CodeProblem,
        timeout: int | float | None = None,
        max_submissions: int = 5,
        max_public_test_calls: int = 15,
    ):
        self.problem = problem
        # timeout 优先级：显式传入 > 题目 time_limit_seconds > 默认 5 秒。
        # 这里的 timeout 会传给每个 test case 的子进程执行。
        self.timeout = timeout if timeout is not None else problem.time_limit_seconds or 5
        self.max_submissions = max_submissions
        self.max_public_test_calls = max_public_test_calls

        # `_state` 是工具执行器的共享状态。`src.env.tools.TOOL_EXECUTORS` 接收的就是
        # 这个 dict，因此本地环境、eval 和 verl tool adapter 能复用同一套工具语义。
        #
        # 重要字段：
        # - public_tests/private_tests: 分别供 run_public_tests / submit_solution 使用
        # - submission_count: 只由 submit_solution 增加
        # - public_results_history/submission_history: 结构化 judge 结果历史
        # - last_result: 最近一次工具调用的结构化结果
        self._state: dict[str, Any] = {
            "current_code": "",
            "public_tests": problem.public_tests,
            "private_tests": problem.private_tests,
            "timeout": self.timeout,
            "max_submissions": max_submissions,
            "max_public_test_calls": max_public_test_calls,
            "public_test_call_count": 0,
            "submission_count": 0,
            "tool_history": [],
            "public_results_history": [],
            "submission_history": [],
            "last_result": None,
        }

    def run_public_tests(self, code: str) -> dict[str, Any]:
        """Run code against public tests and return the structured result.

        这是给本地 Python 调用方用的便捷接口，内部仍走真实 tool executor，
        因此会遵守 `max_public_test_calls`。如果需要 observation text，应直接使用
        `execute_tool("run_public_tests", code=...)`。
        """
        observation = self.execute_tool("run_public_tests", code=code)
        result = self._state["last_result"]
        if result is None:
            raise RuntimeError(f"run_public_tests failed without result: {observation}")
        return result

    def submit_solution(self, code: str) -> dict[str, Any]:
        """Run code against private tests, respecting the submission limit.

        这里故意通过 `execute_tool("submit_solution", ...)` 走工具执行器，而不是直接
        调 `run_oj_judge`，因为 submission limit、submission history 和 observation
        策略都在 `tool_submit_solution` 里维护。
        """
        observation = self.execute_tool("submit_solution", code=code)
        result = self._state["last_result"]
        if result is None:
            raise RuntimeError(f"submit_solution failed without result: {observation}")
        return result

    def execute_tool(self, tool_name: str, **kwargs: Any) -> str:
        """Execute one OJ tool and return observation text.

        这是 multi-turn agent 调用的兼容入口。调用方传入 tool name 和 arguments，
        本方法查 `TOOL_EXECUTORS`，让工具修改 `_state`，并返回给模型看的 observation。
        结构化结果保存在 `last_result` 和对应 history 中。
        """
        executor = TOOL_EXECUTORS.get(tool_name)
        if executor is None:
            return f"Error: Unknown tool '{tool_name}'. Available tools: {list(TOOL_EXECUTORS.keys())}"

        try:
            observation = executor(self._state, **kwargs)
        except TypeError as e:
            # 常见原因是工具参数缺失或名字错误。这里返回文本 observation，让 eval/agent loop
            # 可以继续处理，而不是让整个 rollout 因 Python 异常中断。
            return f"Error calling {tool_name}: {e}"

        self._state["tool_history"].append(tool_name)
        return observation

    def format_result(self, result: dict[str, Any], include_all_failures: bool = False) -> str:
        """把结构化 judge result 转成和 tool observation 一致的文本。"""
        return format_judge_observation(result, include_all_failures=include_all_failures)

    @property
    def current_code(self) -> str:
        """最近一次提交给环境的完整 Python 程序。"""
        return self._state["current_code"]

    @property
    def tool_history(self) -> list[str]:
        """已经成功分发到 TOOL_EXECUTORS 的工具调用名序列。"""
        return self._state["tool_history"]

    @property
    def public_results_history(self) -> list[dict[str, Any]]:
        """所有 public test 调用的结构化 judge 结果。"""
        return self._state["public_results_history"]

    @property
    def submission_history(self) -> list[dict[str, Any]]:
        """所有 submit_solution 调用的结构化 judge 结果。"""
        return self._state["submission_history"]

    @property
    def last_result(self) -> dict[str, Any] | None:
        """最近一次工具调用的结构化结果；没有调用过工具时为 None。"""
        return self._state["last_result"]

    @property
    def is_accepted(self) -> bool:
        """最近一次正式提交是否 accepted。public tests 通过不算 accepted。"""
        if not self.submission_history:
            return False
        return self.submission_history[-1]["verdict"] == VERDICT_ACCEPTED

    @property
    def last_public_pass_ratio(self) -> float:
        """最近一次 public tests 的通过率；没有 public 调用时返回 0。"""
        if not self.public_results_history:
            return 0.0
        last = self.public_results_history[-1]
        return last["passed"] / last["total"] if last["total"] else 0.0

    @property
    def last_submission_verdict(self) -> str | None:
        """最近一次正式提交的 verdict；还没提交过时返回 None。"""
        if not self.submission_history:
            return None
        return self.submission_history[-1]["verdict"]

    def reset(self) -> None:
        """Reset mutable episode state for the same problem.

        题目、tests、timeout、max_submissions 不变，只清空当前代码、提交计数和历史。
        """
        self._state["current_code"] = ""
        self._state["public_test_call_count"] = 0
        self._state["submission_count"] = 0
        self._state["tool_history"] = []
        self._state["public_results_history"] = []
        self._state["submission_history"] = []
        self._state["last_result"] = None
