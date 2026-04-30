"""Tests for the custom verl AgentLoop used by code-agent eval."""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.verl_agent_loop import CodeAgentToolAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop


@contextmanager
def _env(**values: str):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value == "":
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _loop(response_length: int = 8192) -> CodeAgentToolAgentLoop:
    loop = object.__new__(CodeAgentToolAgentLoop)
    loop.response_length = response_length
    return loop


async def _capture_generating_params(agent_data):
    captured = []
    original = ToolAgentLoop._handle_generating_state

    async def fake_base(self, inner_agent_data, sampling_params, ignore_termination=False):
        captured.append(dict(sampling_params))
        return AgentState.TERMINATED

    ToolAgentLoop._handle_generating_state = fake_base
    try:
        state = await _loop()._handle_generating_state(
            agent_data,
            {"temperature": 0.6, "max_tokens": 9999},
        )
    finally:
        ToolAgentLoop._handle_generating_state = original
    return state, captured


def test_first_and_followup_turn_budgets():
    async def run():
        with _env(
            CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET="3072",
            CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET="2048",
        ):
            first = SimpleNamespace(
                assistant_turns=0,
                response_mask=[],
                metrics={},
            )
            _, first_params = await _capture_generating_params(first)
            assert first_params[0]["max_new_tokens"] == 3072
            assert "max_tokens" not in first_params[0]

            followup = SimpleNamespace(
                assistant_turns=1,
                response_mask=[1] * 100,
                metrics={},
            )
            _, followup_params = await _capture_generating_params(followup)
            assert followup_params[0]["max_new_tokens"] == 2048

    asyncio.run(run())
    print("[PASS] test_first_and_followup_turn_budgets")


def test_remaining_budget_caps_turn_budget():
    async def run():
        with _env(
            CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET="3072",
            CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET="2048",
        ):
            agent_data = SimpleNamespace(
                assistant_turns=1,
                response_mask=[1] * 8000,
                metrics={},
            )
            _, params = await _capture_generating_params(agent_data)
            assert params[0]["max_new_tokens"] == 192

    asyncio.run(run())
    print("[PASS] test_remaining_budget_caps_turn_budget")


def test_terminal_tool_state_stops_agent_loop():
    async def run():
        original = ToolAgentLoop._handle_processing_tools_state

        async def fake_base(self, agent_data):
            return AgentState.GENERATING

        ToolAgentLoop._handle_processing_tools_state = fake_base
        try:
            agent_data = SimpleNamespace(
                code_agent_terminal=True,
                code_agent_terminal_reason="accepted",
                metrics={},
            )
            state = await _loop()._handle_processing_tools_state(agent_data)
            assert state == AgentState.TERMINATED
        finally:
            ToolAgentLoop._handle_processing_tools_state = original

    asyncio.run(run())
    print("[PASS] test_terminal_tool_state_stops_agent_loop")


if __name__ == "__main__":
    test_first_and_followup_turn_budgets()
    test_remaining_budget_caps_turn_budget()
    test_terminal_tool_state_stops_agent_loop()
    print("\nAll tests passed!")
