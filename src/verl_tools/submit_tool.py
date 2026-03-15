"""
SubmitTool — verl BaseTool 实现
================================

提交最终答案，委托给 CodeEnvironment.execute_tool()。
calc_reward() 返回整条轨迹的最终 reward，包含：
  - 部分执行奖励 (passed/total)
  - 正确顺序奖励 (write_code 先于 run_tests)
  - 修复奖励 (先失败后修好)
  - 提交奖励 (全部通过后 submit)
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .state_manager import CodeEnvStateManager


class SubmitTool(BaseTool):

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
            observation = env.execute_tool("submit")
            return ToolResponse(text=observation), 0.0, {}

        observation = env.execute_tool("submit")

        total = len(env._state["test_list"]) or 1
        passed = round(env.submit_pass_ratio * total)
        return ToolResponse(text=observation), 0.0, {"passed": passed, "total": total}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """计算整条轨迹的最终 reward。

        组成:
        1. 部分执行: passed / total (0.0 ~ 1.0)
        2. 正确顺序: write_code 出现在首个 run_tests 之前 -> +0.1
        3. 修复奖励: 先失败后通过 -> +0.2
        4. 提交奖励: 全部通过且调了 submit -> +0.1
        """
        if not CodeEnvStateManager.exists(instance_id):
            return 0.0

        env = CodeEnvStateManager.get(instance_id)
        history = env.tool_history
        test_history = env.test_results_history

        exec_reward = env.submit_pass_ratio

        order_reward = 0.0
        if history:
            wc_indices = [i for i, t in enumerate(history) if t == "write_code"]
            rt_indices = [i for i, t in enumerate(history) if t == "run_tests"]
            if wc_indices and rt_indices and wc_indices[0] < rt_indices[0]:
                order_reward = 0.1

        fix_reward = 0.0
        if len(test_history) >= 2:
            first = test_history[0]
            last = test_history[-1]
            if first["passed"] < first["total"] and last["passed"] == last["total"]:
                fix_reward = 0.2

        submit_reward = 0.0
        if env.submit_passed and "submit" in history:
            submit_reward = 0.1

        return exec_reward + order_reward + fix_reward + submit_reward

    async def release(self, instance_id: str, **kwargs) -> None:
        CodeEnvStateManager.release(instance_id)
