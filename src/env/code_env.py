"""
代码执行环境
============

CodeEnvironment 管理单个编程题的完整生命周期：
初始化 → agent 多轮 tool 调用 → 获取最终 reward。
"""

from __future__ import annotations

from typing import Any

from .tools import TOOL_EXECUTORS


class CodeEnvironment:
    """一道编程题的交互环境.

    维护 env_state 字典，每次 tool 调用都读写该字典。
    同时追踪工具调用历史和测试结果历史，供 reward 计算使用。

    Attributes:
        problem_description: 题目描述文本
        test_list: 测试用例列表（assert 语句）
        timeout: 单次代码执行的超时秒数
    """

    def __init__(
        self,
        problem_description: str,
        test_list: list[str],
        entry_point: str | None = None,
        timeout: int = 5,
    ):
        self.problem_description = problem_description
        self.entry_point = entry_point

        self._state: dict[str, Any] = {
            "current_code": "",
            "test_list": test_list,
            "last_traceback": "",
            "timeout": timeout,
            "tool_history": [],
            "test_results_history": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_tool(self, tool_name: str, **kwargs: Any) -> str:
        """执行一个 tool 并返回 observation.

        自动将 tool_name 追加到 tool_history。

        Args:
            tool_name: tool 名称，必须在 TOOL_EXECUTORS 中
            **kwargs: 传给 tool 函数的参数

        Returns:
            observation 字符串
        """
        executor = TOOL_EXECUTORS.get(tool_name)
        if executor is None:
            return f"Error: Unknown tool '{tool_name}'. Available tools: {list(TOOL_EXECUTORS.keys())}"
        try:
            observation = executor(self._state, **kwargs)
            self._state["tool_history"].append(tool_name)
            return observation
        except TypeError as e:
            return f"Error calling {tool_name}: {e}"

    @property
    def current_code(self) -> str:
        return self._state["current_code"]

    @property
    def tool_history(self) -> list[str]:
        return self._state["tool_history"]

    @property
    def test_results_history(self) -> list[dict]:
        return self._state["test_results_history"]

    @property
    def is_all_passed(self) -> bool:
        """最近一次测试是否全部通过。"""
        if not self._state["test_results_history"]:
            return False
        last = self._state["test_results_history"][-1]
        return last["passed"] == last["total"] and last["total"] > 0

    @property
    def last_pass_ratio(self) -> float:
        """最近一次测试的通过率。"""
        if not self._state["test_results_history"]:
            return 0.0
        last = self._state["test_results_history"][-1]
        return last["passed"] / last["total"] if last["total"] > 0 else 0.0

    def reset(self) -> None:
        """重置环境状态，复用同一道题."""
        self._state["current_code"] = ""
        self._state["last_traceback"] = ""
        self._state["tool_history"] = []
        self._state["test_results_history"] = []
