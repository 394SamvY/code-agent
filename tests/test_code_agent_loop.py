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
from verl.experimental.agent_loop.tool_parser import FunctionCall
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
                extra_fields={},
            )
            state = await _loop()._handle_processing_tools_state(agent_data)
            assert state == AgentState.TERMINATED
        finally:
            ToolAgentLoop._handle_processing_tools_state = original

    asyncio.run(run())
    print("[PASS] test_terminal_tool_state_stops_agent_loop")


def test_thinking_budget_inserts_early_stop_and_continues_to_tool_call():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(1000, 1000 + len(text)))

        def decode(self, ids, skip_special_tokens=False):
            if ids == [1, 2]:
                return "<think>partial reasoning"
            if ids == [3]:
                return '<tool_call>{"name":"submit_solution","arguments":{"code":"print(1)"}}</tool_call>'
            return ""

        def convert_tokens_to_ids(self, token):
            return 151645 if token == "<|im_end|>" else 151668

    class FakeServerManager:
        def __init__(self):
            self.calls = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs["sampling_params"])
            if len(self.calls) == 1:
                return SimpleNamespace(
                    token_ids=[1, 2],
                    log_probs=None,
                    num_preempted=None,
                    extra_fields={},
                    routed_experts=None,
                    stop_reason="length",
                )
            return SimpleNamespace(
                token_ids=[3],
                log_probs=None,
                num_preempted=None,
                extra_fields={},
                routed_experts=None,
                stop_reason="stop",
            )

    class FakeToolParser:
        async def extract_tool_calls(self, response_ids, tools):
            return "", [FunctionCall(name="submit_solution", arguments='{"code":"print(1)"}')]

    async def run():
        with _env(
            CODE_AGENT_ENABLE_THINKING_EARLY_STOP="1",
            CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET="8",
            CODE_AGENT_THINKING_TOKEN_BUDGET="2",
        ):
            loop = _loop(response_length=512)
            loop.server_manager = FakeServerManager()
            loop.tokenizer = FakeTokenizer()
            loop.loop = asyncio.get_running_loop()
            loop.tool_parser = FakeToolParser()
            loop.tools = {}
            loop.max_assistant_turns = None
            loop.max_user_turns = None
            loop.max_parallel_calls = 1
            agent_data = SimpleNamespace(
                assistant_turns=0,
                user_turns=0,
                response_mask=[],
                response_logprobs=[],
                prompt_ids=[],
                response_ids=[],
                metrics={},
                extra_fields={},
                image_data=None,
                video_data=None,
                request_id="rid",
                routed_experts=None,
                messages=[{"role": "user", "content": "solve"}],
                tool_calls=[],
            )

            state = await loop._handle_generating_state(agent_data, {"temperature": 0.6})

            assert state == AgentState.PROCESSING_TOOLS
            assert loop.server_manager.calls[0]["max_new_tokens"] == 2
            assert "stop" not in loop.server_manager.calls[0]
            assert loop.server_manager.calls[1]["max_new_tokens"] == 6
            assert agent_data.extra_fields["code_agent_thinking_budget_reached"] is True
            trace = agent_data.extra_fields["code_agent_trace"]
            assert trace["assistant_turns"][0]["thinking_closed"] is True
            assert "I have to give the solution" in trace["assistant_turns"][0]["text"]
            assert trace["assistant_turns"][0]["tool_call_count"] == 1

    asyncio.run(run())
    print("[PASS] test_thinking_budget_inserts_early_stop_and_continues_to_tool_call")


def test_im_end_after_tool_call_still_executes_tool():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(1000, 1000 + len(text)))

        def decode(self, ids, skip_special_tokens=False):
            if ids == [4, 151645]:
                return '<tool_call>{"name":"submit_solution","arguments":{"code":"print(1)"}}</tool_call><|im_end|>'
            return ""

        def convert_tokens_to_ids(self, token):
            if token == "<|im_end|>":
                return 151645
            if token == "</think>":
                return 151668
            return None

    class FakeServerManager:
        def __init__(self):
            self.calls = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs["sampling_params"])
            return SimpleNamespace(
                token_ids=[4, 151645],
                log_probs=None,
                num_preempted=None,
                extra_fields={},
                routed_experts=None,
                stop_reason="stop",
            )

    class FakeToolParser:
        async def extract_tool_calls(self, response_ids, tools):
            return "", [FunctionCall(name="submit_solution", arguments='{"code":"print(1)"}')]

    async def run():
        with _env(
            CODE_AGENT_ENABLE_THINKING_EARLY_STOP="1",
            CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET="8",
            CODE_AGENT_THINKING_TOKEN_BUDGET="8",
        ):
            loop = _loop(response_length=512)
            loop.server_manager = FakeServerManager()
            loop.tokenizer = FakeTokenizer()
            loop.loop = asyncio.get_running_loop()
            loop.tool_parser = FakeToolParser()
            loop.tools = {}
            loop.max_assistant_turns = None
            loop.max_user_turns = None
            loop.max_parallel_calls = 1
            agent_data = SimpleNamespace(
                assistant_turns=0,
                user_turns=0,
                response_mask=[],
                response_logprobs=[],
                prompt_ids=[],
                response_ids=[],
                metrics={},
                extra_fields={},
                image_data=None,
                video_data=None,
                request_id="rid",
                routed_experts=None,
                messages=[{"role": "user", "content": "solve"}],
                tool_calls=[],
            )

            state = await loop._handle_generating_state(agent_data, {"temperature": 0.6})

            assert state == AgentState.PROCESSING_TOOLS
            assert len(loop.server_manager.calls) == 1
            assert agent_data.tool_calls[0].name == "submit_solution"
            trace = agent_data.extra_fields["code_agent_trace"]
            assert trace["terminal_reason"] is None
            assert trace["assistant_turns"][0]["model_im_end"] is True
            assert trace["assistant_turns"][0]["tool_call_count"] == 1
            assert agent_data.tool_calls[0].name == "submit_solution"

    asyncio.run(run())
    print("[PASS] test_im_end_after_tool_call_still_executes_tool")


if __name__ == "__main__":
    test_first_and_followup_turn_budgets()
    test_remaining_budget_caps_turn_budget()
    test_terminal_tool_state_stops_agent_loop()
    test_thinking_budget_inserts_early_stop_and_continues_to_tool_call()
    test_im_end_after_tool_call_still_executes_tool()
    print("\nAll tests passed!")
