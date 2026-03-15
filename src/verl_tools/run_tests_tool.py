"""
RunTestsTool — verl BaseTool 实现
==================================

运行全部测试用例，委托给 CodeEnvironment.execute_tool()。
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .state_manager import CodeEnvStateManager


class RunTestsTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        if not CodeEnvStateManager.exists(instance_id):
            create_kwargs = kwargs.get("create_kwargs", {})
            CodeEnvStateManager.create(
                instance_id,
                test_list=create_kwargs.get("test_list", []),
                entry_point=create_kwargs.get("entry_point", ""),
                timeout=create_kwargs.get("timeout", 5),
            )

        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        env = CodeEnvStateManager.get(instance_id)

        if not env.current_code.strip():
            return (
                ToolResponse(text="Error: No code written yet. Use write_code first."),
                0.0,
                {},
            )

        observation = env.execute_tool("run_tests")

        last_result = env.test_results_history[-1] if env.test_results_history else {}
        passed = last_result.get("passed", 0)
        total = last_result.get("total", 0)

        step_reward = 0.05 if env.current_code.strip() else 0.0
        return ToolResponse(text=observation), step_reward, {"passed": passed, "total": total}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        pass
