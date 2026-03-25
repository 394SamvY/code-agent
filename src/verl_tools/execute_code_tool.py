"""
ExecuteCodeTool — verl BaseTool 实现
=====================================

单工具设计：接收完整代码 → 语法检查 → 跑全部测试 → 返回结果。
每次 create → execute → release 独立，无状态。

参考 ReTool (verl-recipe/retool) 和 GSM8K 官方示例的设计模式。
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from src.env.sandbox import execute_with_tests


class ExecuteCodeTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict] = {}

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {})
        test_list = create_kwargs.get("test_list", [])
        if isinstance(test_list, str):
            import json
            test_list = json.loads(test_list)

        self._instances[instance_id] = {
            "test_list": test_list,
            "entry_point": create_kwargs.get("entry_point", ""),
            "timeout": create_kwargs.get("timeout", 5),
            "attempts": [],  # 每次 execute 的 (passed, total)
        }

        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        inst = self._instances.get(instance_id)
        if inst is None:
            return ToolResponse(text="Error: instance not found."), 0.0, {}

        code = parameters.get("code", "")
        if not isinstance(code, str):
            code = str(code)

        # 1. 语法检查
        try:
            compile(code, "<solution>", "exec")
        except SyntaxError as e:
            inst["attempts"].append((0, max(len(inst["test_list"]), 1)))
            return (
                ToolResponse(text=f"Syntax error at line {e.lineno}: {e.msg}. Please fix it."),
                0.0,
                {},
            )

        # 2. 逐条跑测试
        test_list = inst["test_list"]
        if not test_list:
            return ToolResponse(text="No test cases available."), 0.0, {}

        timeout = inst["timeout"]
        passed, failed = 0, 0
        failure_details: list[str] = []

        for i, test in enumerate(test_list):
            result = execute_with_tests(code, test, timeout=timeout)
            if result.success:
                passed += 1
            else:
                failed += 1
                tb = result.stderr.strip() if result.stderr else "Unknown error"
                tb_lines = tb.split("\n")
                if len(tb_lines) > 10:
                    tb = "\n".join(tb_lines[:3] + ["  ..."] + tb_lines[-5:])
                failure_details.append(f"Test {i + 1}: {test.strip()}\n{tb}")

        total = len(test_list)
        inst["attempts"].append((passed, total))

        # 3. 构造返回文本
        summary = f"{passed}/{total} tests passed."
        if failure_details:
            summary += "\n\nFailure details:\n" + "\n---\n".join(failure_details)
        else:
            summary += " All tests passed!"

        # step_reward: 通过率 * 0.1，鼓励模型写出更好的代码
        step_reward = (passed / total) * 0.1 if total > 0 else 0.0

        return ToolResponse(text=summary), step_reward, {"passed": passed, "total": total}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """计算整条轨迹的最终 reward。

        组成:
        1. exec_reward: 最后一次尝试的 passed / total (0.0 ~ 1.0)
        2. fix_reward: 首次失败但最终修复通过 -> +0.2
        """
        inst = self._instances.get(instance_id)
        if inst is None or not inst["attempts"]:
            return 0.0

        attempts = inst["attempts"]
        last_passed, last_total = attempts[-1]

        # exec_reward: 最终通过率
        exec_reward = last_passed / last_total if last_total > 0 else 0.0

        # fix_reward: 首次没全部通过，但最终全部通过
        fix_reward = 0.0
        if len(attempts) >= 2:
            first_passed, first_total = attempts[0]
            if first_passed < first_total and last_passed == last_total:
                fix_reward = 0.2

        return exec_reward + fix_reward

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
