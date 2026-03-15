"""
WriteCodeTool — verl BaseTool 实现
===================================

写入/替换当前代码，委托给 CodeEnvironment.execute_tool()。
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .state_manager import CodeEnvStateManager


class WriteCodeTool(BaseTool):

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
        code = parameters.get("code", "")
        if not isinstance(code, str):
            code = str(code)

        observation = env.execute_tool("write_code", code=code)
        step_reward = 0.05 if "Syntax OK" in observation else 0.0

        return ToolResponse(text=observation), step_reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        pass
