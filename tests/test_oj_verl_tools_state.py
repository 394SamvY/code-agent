"""Focused tests for verl OJ BaseTool trajectory state."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.verl_tools.oj_tools import RunPublicTestsTool, SubmitSolutionTool
from verl.tools.schemas import OpenAIFunctionToolSchema


def _schema(name: str) -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    })


async def _execute_once(tool, create_kwargs, code: str, agent_data):
    instance_id, _ = await tool.create(create_kwargs=create_kwargs)
    try:
        return await tool.execute(instance_id, {"code": code}, agent_data=agent_data)
    finally:
        await tool.release(instance_id)


async def _test_public_test_count_persists_across_instances():
    tool = RunPublicTestsTool({}, _schema("run_public_tests"))
    agent_data = SimpleNamespace(extra_fields={})
    create_kwargs = {
        "public_tests": [{"input": "", "output": "1\n"}],
        "time_limit_seconds": 1,
        "max_public_test_calls": 2,
    }

    await _execute_once(tool, create_kwargs, "print(1)", agent_data)
    await _execute_once(tool, create_kwargs, "print(1)", agent_data)
    _, _, result = await _execute_once(tool, create_kwargs, "print(1)", agent_data)

    assert result["verdict"] == "public_test_limit_exceeded"
    assert result["public_test_call_count"] == 2
    assert agent_data.code_agent_oj_tool_state["public_test_call_count"] == 2
    assert "code_agent_oj_tool_state" not in agent_data.extra_fields

    print("[PASS] test_public_test_count_persists_across_instances")


async def _test_submit_acceptance_marks_trajectory_terminal():
    tool = SubmitSolutionTool({}, _schema("submit_solution"))
    agent_data = SimpleNamespace(extra_fields={})
    create_kwargs = {
        "private_tests": [{"input": "", "output": "1\n"}],
        "time_limit_seconds": 1,
        "max_submissions": 5,
    }

    _, reward, result = await _execute_once(tool, create_kwargs, "print(1)", agent_data)

    assert reward == 1.0
    assert result["verdict"] == "accepted"
    assert result["terminal"] is True
    assert result["terminal_reason"] == "accepted"
    assert agent_data.code_agent_terminal is True
    assert agent_data.code_agent_terminal_reason == "accepted"
    assert "code_agent_terminal" not in agent_data.extra_fields

    print("[PASS] test_submit_acceptance_marks_trajectory_terminal")


async def _test_submit_count_persists_across_instances():
    tool = SubmitSolutionTool({}, _schema("submit_solution"))
    agent_data = SimpleNamespace(extra_fields={})
    create_kwargs = {
        "private_tests": [{"input": "", "output": "1\n"}],
        "time_limit_seconds": 1,
        "max_submissions": 1,
    }

    _, _, first = await _execute_once(tool, create_kwargs, "print(0)", agent_data)
    _, _, second = await _execute_once(tool, create_kwargs, "print(0)", agent_data)

    assert first["verdict"] == "wrong_answer"
    assert first["terminal"] is True
    assert first["terminal_reason"] == "submission_limit_exhausted"
    assert second["verdict"] == "submission_limit_exceeded"
    assert agent_data.code_agent_oj_tool_state["submission_count"] == 1
    assert "code_agent_oj_tool_state" not in agent_data.extra_fields

    print("[PASS] test_submit_count_persists_across_instances")


def test_public_test_count_persists_across_instances():
    asyncio.run(_test_public_test_count_persists_across_instances())


def test_submit_acceptance_marks_trajectory_terminal():
    asyncio.run(_test_submit_acceptance_marks_trajectory_terminal())


def test_submit_count_persists_across_instances():
    asyncio.run(_test_submit_count_persists_across_instances())


if __name__ == "__main__":
    test_public_test_count_persists_across_instances()
    test_submit_acceptance_marks_trajectory_terminal()
    test_submit_count_persists_across_instances()
    print("\nAll tests passed!")
