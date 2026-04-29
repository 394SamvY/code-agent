"""
verl BaseTool adapters for the OJ-like two-action protocol.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from src.env.tools import (
    DEFAULT_MAX_PUBLIC_TEST_CALLS,
    VERDICT_PUBLIC_TEST_LIMIT_EXCEEDED,
    VERDICT_SUBMISSION_LIMIT_EXCEEDED,
    format_judge_observation,
    parse_oj_tests,
    reward_for_result,
    run_oj_judge,
)


def _load_tests(value: Any):
    if isinstance(value, str):
        value = json.loads(value)
    return parse_oj_tests(value)


def _resolve_create_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Accept common verl create payload shapes."""
    raw_create_kwargs = kwargs.get("create_kwargs") or {}
    if isinstance(raw_create_kwargs, str):
        raw_create_kwargs = json.loads(raw_create_kwargs)
    create_kwargs = dict(raw_create_kwargs)
    extra_info = kwargs.get("extra_info") or {}
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)
    if isinstance(extra_info, dict):
        create_kwargs.update(extra_info.get("create_kwargs") or {})
    for key in (
        "public_tests",
        "private_tests",
        "time_limit_seconds",
        "timeout",
        "max_submissions",
        "max_public_test_calls",
    ):
        if key in kwargs and key not in create_kwargs:
            create_kwargs[key] = kwargs[key]
    return create_kwargs


class _OJBaseTool(BaseTool):
    action_name = ""
    tests_key = ""
    include_all_failures = False

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict[str, Any]] = {}

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = _resolve_create_kwargs(kwargs)
        timeout = create_kwargs.get("time_limit_seconds") or create_kwargs.get("timeout") or 5
        max_submissions = create_kwargs.get("max_submissions", 5)
        max_public_test_calls = create_kwargs.get(
            "max_public_test_calls",
            DEFAULT_MAX_PUBLIC_TEST_CALLS,
        )
        self._instances[instance_id] = {
            "public_tests": _load_tests(create_kwargs.get("public_tests", [])),
            "private_tests": _load_tests(create_kwargs.get("private_tests", [])),
            "timeout": float(timeout),
            "max_submissions": int(max_submissions),
            "max_public_test_calls": int(max_public_test_calls),
            "public_test_call_count": 0,
            "submission_count": 0,
            "results": [],
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

        result = self._run(inst, code)
        inst["results"].append(result)

        observation = format_judge_observation(
            result,
            include_all_failures=self.include_all_failures,
        )
        step_reward = reward_for_result(result)
        return ToolResponse(text=observation), step_reward, result

    def _run(self, inst: dict[str, Any], code: str) -> dict[str, Any]:
        return run_oj_judge(
            code=code,
            tests=inst[self.tests_key],
            action=self.action_name,
            timeout=inst["timeout"],
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        inst = self._instances.get(instance_id)
        if inst is None or not inst["results"]:
            return 0.0
        return reward_for_result(inst["results"][-1])

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


class RunPublicTestsTool(_OJBaseTool):
    action_name = "run_public_tests"
    tests_key = "public_tests"
    include_all_failures = False

    def _run(self, inst: dict[str, Any], code: str) -> dict[str, Any]:
        if inst["public_test_call_count"] >= inst["max_public_test_calls"]:
            return {
                "action": self.action_name,
                "verdict": VERDICT_PUBLIC_TEST_LIMIT_EXCEEDED,
                "passed": 0,
                "total": len(inst[self.tests_key]),
                "first_failed": None,
                "tests": [],
                "public_test_call_count": inst["public_test_call_count"],
                "max_public_test_calls": inst["max_public_test_calls"],
            }

        inst["public_test_call_count"] += 1
        result = super()._run(inst, code)
        result["public_test_call_count"] = inst["public_test_call_count"]
        result["max_public_test_calls"] = inst["max_public_test_calls"]
        return result


class SubmitSolutionTool(_OJBaseTool):
    action_name = "submit_solution"
    tests_key = "private_tests"
    include_all_failures = False

    def _run(self, inst: dict[str, Any], code: str) -> dict[str, Any]:
        if inst["submission_count"] >= inst["max_submissions"]:
            return {
                "action": self.action_name,
                "verdict": VERDICT_SUBMISSION_LIMIT_EXCEEDED,
                "passed": 0,
                "total": len(inst[self.tests_key]),
                "first_failed": None,
                "tests": [],
            }

        inst["submission_count"] += 1
        result = super()._run(inst, code)
        result["submission_count"] = inst["submission_count"]
        result["max_submissions"] = inst["max_submissions"]
        return result
