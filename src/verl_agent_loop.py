"""Code-agent verl AgentLoop implementations.

This module is loaded inside verl AgentLoopWorker Ray actors via
``actor_rollout_ref.rollout.agent.agent_loop_config_path``. Logic that must run
in the actual per-sample agent loop process belongs here, not in
``src.verl_runtime_patch`` installed by the TaskRunner actor.
"""

from __future__ import annotations

import os
from typing import Any

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop


def _positive_int_from_env(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        print(f"[code-agent] Ignoring invalid {name}={raw!r}; expected a positive integer")
        return None
    return value if value > 0 else None


def _budget_for_turn(agent_data: Any) -> int | None:
    """Return the assistant generation budget for the current turn.

    ``agent_data.assistant_turns`` is incremented by verl after a generation
    completes, so value 0 means "about to generate the first assistant turn".
    """
    if int(getattr(agent_data, "assistant_turns", 0) or 0) <= 0:
        return _positive_int_from_env("CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET")
    return _positive_int_from_env("CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET")


def _sampling_params_with_turn_budget(
    *,
    sampling_params: dict[str, Any],
    turn_budget: int,
    remaining_budget: int,
) -> dict[str, Any]:
    """Apply a per-assistant-turn cap while respecting trajectory remaining budget."""
    max_new_tokens = min(turn_budget, remaining_budget)
    capped_sampling_params = dict(sampling_params)
    for key in ("max_new_tokens", "max_tokens"):
        if key not in capped_sampling_params:
            continue
        try:
            max_new_tokens = min(max_new_tokens, int(capped_sampling_params[key]))
        except (TypeError, ValueError):
            pass
        capped_sampling_params.pop(key, None)
    capped_sampling_params["max_new_tokens"] = max_new_tokens
    return capped_sampling_params


class CodeAgentToolAgentLoop(ToolAgentLoop):
    """OJ-like tool agent loop with eval-time generation budget controls."""

    async def _handle_generating_state(
        self,
        agent_data,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        turn_budget = _budget_for_turn(agent_data)
        if turn_budget is not None:
            remaining_budget = max(
                0,
                int(self.response_length) - len(agent_data.response_mask),
            )
            if remaining_budget <= 0:
                return AgentState.TERMINATED
            sampling_params = _sampling_params_with_turn_budget(
                sampling_params=sampling_params,
                turn_budget=turn_budget,
                remaining_budget=remaining_budget,
            )
            agent_data.metrics["code_agent_assistant_turn_token_budget"] = turn_budget
            agent_data.metrics["code_agent_assistant_turn_index"] = int(
                getattr(agent_data, "assistant_turns", 0) or 0
            )

        return await super()._handle_generating_state(
            agent_data,
            sampling_params,
            ignore_termination=ignore_termination,
        )

    async def _handle_processing_tools_state(self, agent_data) -> AgentState:
        state = await super()._handle_processing_tools_state(agent_data)
        if getattr(agent_data, "code_agent_terminal", False):
            agent_data.metrics["code_agent_terminal"] = 1
            return AgentState.TERMINATED
        return state
