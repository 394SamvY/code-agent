"""Code-agent specific verl agent loop.

The upstream ``tool_agent`` loop knows how to inject tool schemas and execute
tools, but it does not understand OJ terminal semantics.  This wrapper keeps
the upstream implementation and only adds the stop conditions required by the
two-tool OJ protocol.
"""

from __future__ import annotations

import json
import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.tools.schemas import ToolResponse

from src.env.tools import DEFAULT_MAX_PUBLIC_TEST_CALLS


_TRACE_KEY = "code_agent_trace"
_TERMINAL_KEY = "code_agent_terminal"
_TERMINAL_REASON_KEY = "code_agent_terminal_reason"
_PARSE_FAILURES_KEY = "code_agent_parse_failures"
_TOOL_TAIL_CHARS_KEY = "code_agent_tool_tail_chars"


@register("code_agent_tool_agent")
class CodeAgentToolAgentLoop(ToolAgentLoop):
    """ToolAgentLoop with OJ-like terminal handling and trace fields."""

    def _ensure_extra_fields(self, agent_data: AgentData) -> None:
        """Initialize optional dump fields so verl can concat parallel outputs."""
        agent_data.extra_fields.setdefault(_TERMINAL_REASON_KEY, None)
        agent_data.extra_fields.setdefault(_PARSE_FAILURES_KEY, 0)
        agent_data.extra_fields.setdefault(_TOOL_TAIL_CHARS_KEY, 0)

    def _trace(self, agent_data: AgentData) -> dict[str, Any]:
        self._ensure_extra_fields(agent_data)
        trace = agent_data.extra_fields.get(_TRACE_KEY)
        if not isinstance(trace, dict):
            trace = {
                "num_tool_calls": 0,
                "parse_failures": 0,
                "terminal_reason": None,
                "last_action": None,
                "last_verdict": None,
                "public_test_call_count": 0,
                "submission_count": 0,
                "max_tool_calls": self._max_tool_calls(agent_data),
            }
            agent_data.extra_fields[_TRACE_KEY] = trace
        return trace

    def _create_kwargs(self, agent_data: AgentData, tool_name: str) -> dict[str, Any]:
        tool_kwargs = agent_data.tools_kwargs.get(tool_name, {})
        create_kwargs = tool_kwargs.get("create_kwargs", {})
        if isinstance(create_kwargs, str):
            try:
                create_kwargs = json.loads(create_kwargs)
            except json.JSONDecodeError:
                create_kwargs = {}
        return create_kwargs if isinstance(create_kwargs, dict) else {}

    def _max_tool_calls(self, agent_data: AgentData) -> int:
        env_value = os.getenv("CODE_AGENT_MAX_TOOL_CALLS")
        if env_value and env_value.isdigit():
            return int(env_value)

        public_kwargs = self._create_kwargs(agent_data, "run_public_tests")
        submit_kwargs = self._create_kwargs(agent_data, "submit_solution")
        max_public = int(public_kwargs.get("max_public_test_calls", DEFAULT_MAX_PUBLIC_TEST_CALLS))
        max_submissions = int(
            submit_kwargs.get(
                "max_submissions",
                public_kwargs.get("max_submissions", 5),
            )
        )
        # Keep one extra observation for each limit-exceeded feedback.  The
        # hard cap is only a rollout guard; it should not consume the OJ
        # budgets before the model can observe that a limit was reached.
        return max_public + max_submissions + 2

    def _mark_terminal(self, agent_data: AgentData, reason: str) -> None:
        setattr(agent_data, _TERMINAL_KEY, True)
        setattr(agent_data, _TERMINAL_REASON_KEY, reason)
        trace = self._trace(agent_data)
        trace["terminal_reason"] = reason
        agent_data.extra_fields[_TERMINAL_REASON_KEY] = reason

    def _terminal_reason(self, agent_data: AgentData) -> str | None:
        trace = agent_data.extra_fields.get(_TRACE_KEY)
        if isinstance(trace, dict) and trace.get("terminal_reason"):
            return str(trace["terminal_reason"])
        reason = agent_data.extra_fields.get(_TERMINAL_REASON_KEY)
        if reason:
            return str(reason)
        return None

    def _should_terminate(self, agent_data: AgentData) -> bool:
        trace = self._trace(agent_data)
        if int(trace.get("num_tool_calls", 0)) >= int(trace.get("max_tool_calls", self._max_tool_calls(agent_data))):
            self._mark_terminal(agent_data, "tool_call_limit_exhausted")
            return True
        return False

    def _record_no_tool_call_termination(self, agent_data: AgentData) -> None:
        if self._terminal_reason(agent_data):
            return
        self._mark_terminal(agent_data, "no_tool_call")

    def _record_parse_failure_if_needed(self, agent_data: AgentData) -> None:
        if agent_data.tool_calls:
            return
        if not agent_data.response_ids:
            return
        text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False)
        if "<tool_call>" not in text:
            return
        trace = self._trace(agent_data)
        trace["parse_failures"] = int(trace.get("parse_failures", 0)) + 1
        trace["tool_tail_chars"] = len(text[-512:])
        agent_data.extra_fields[_PARSE_FAILURES_KEY] = trace["parse_failures"]
        agent_data.extra_fields[_TOOL_TAIL_CHARS_KEY] = trace["tool_tail_chars"]

    def _record_tool_result(self, agent_data: AgentData, result: dict[str, Any]) -> None:
        trace = self._trace(agent_data)
        trace["num_tool_calls"] = int(trace.get("num_tool_calls", 0)) + 1

        action = result.get("action") if isinstance(result, dict) else None
        verdict = result.get("verdict") if isinstance(result, dict) else None
        if action:
            trace["last_action"] = action
        if verdict:
            trace["last_verdict"] = verdict
        if "public_test_call_count" in result:
            trace["public_test_call_count"] = int(result["public_test_call_count"])
        if "submission_count" in result:
            trace["submission_count"] = int(result["submission_count"])

        if int(trace.get("num_tool_calls", 0)) >= int(trace.get("max_tool_calls", self._max_tool_calls(agent_data))):
            self._mark_terminal(agent_data, self._terminal_reason(agent_data) or "tool_call_limit_exhausted")

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        if state == AgentState.TERMINATED and agent_data.response_ids:
            tools = [tool.tool_schema for tool in self.tools.values()]
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
        self._record_parse_failure_if_needed(agent_data)
        if state == AgentState.TERMINATED and not agent_data.tool_calls:
            self._record_no_tool_call_termination(agent_data)
        if self._should_terminate(agent_data):
            return AgentState.TERMINATED
        return state

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        if self._should_terminate(agent_data):
            return AgentState.TERMINATED
        state = await super()._handle_processing_tools_state(agent_data)
        if self._should_terminate(agent_data):
            return AgentState.TERMINATED
        return state

    async def _call_tool(
        self, tool_call, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        tool_response, tool_reward, result = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        self._record_tool_result(agent_data, result if isinstance(result, dict) else {})
        return tool_response, tool_reward, result
