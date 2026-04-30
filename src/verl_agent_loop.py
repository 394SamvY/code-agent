"""Code-agent verl AgentLoop implementations.

This module is loaded inside verl AgentLoopWorker Ray actors via
``actor_rollout_ref.rollout.agent.agent_loop_config_path``. Logic that must run
in the actual per-sample agent loop process belongs here, not in
``src.verl_runtime_patch`` installed by the TaskRunner actor.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer

from src.env.tools import TERMINAL_VERDICTS


_THINK_END = "</think>"
_EARLY_STOPPING_TEXT = (
    "\n\nConsidering the limited time by the user, "
    "I have to give the solution based on the thinking directly now.\n"
    "</think>\n\n"
)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


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


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _budget_for_turn(agent_data: Any) -> int | None:
    """Return the assistant generation budget for the current turn.

    ``agent_data.assistant_turns`` is incremented by verl after a generation
    completes, so value 0 means "about to generate the first assistant turn".
    """
    if int(getattr(agent_data, "assistant_turns", 0) or 0) <= 0:
        return _positive_int_from_env("CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET")
    return _positive_int_from_env("CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET")


def _thinking_budget_for_turn(agent_data: Any, turn_budget: int | None) -> int | None:
    budget = _positive_int_from_env("CODE_AGENT_THINKING_TOKEN_BUDGET")
    if budget is None and turn_budget is not None:
        budget = min(1024, turn_budget)
    if budget is None:
        budget = 1024
    if budget is None:
        return None
    return min(budget, turn_budget) if turn_budget is not None else budget


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


def _token_id(tokenizer: Any, token: str, fallback: int) -> int:
    try:
        value = tokenizer.convert_tokens_to_ids(token)
        return int(value) if value is not None else fallback
    except Exception:
        return fallback


def _contains_im_end(tokenizer: Any, token_ids: list[int]) -> bool:
    return _token_id(tokenizer, "<|im_end|>", 151645) in token_ids


def _contains_think_end(tokenizer: Any, token_ids: list[int]) -> bool:
    return _token_id(tokenizer, _THINK_END, 151668) in token_ids


def _assistant_tail_after_tool_call(text: str) -> str:
    matches = list(_TOOL_CALL_RE.finditer(text or ""))
    if not matches:
        return ""
    return text[matches[-1].end() :].strip()


def _count_malformed_tool_calls(text: str, parsed_tool_call_count: int) -> int:
    complete_tags = list(_TOOL_CALL_RE.finditer(text or ""))
    malformed = 0
    valid = 0
    for match in complete_tags:
        try:
            payload = json.loads(match.group(1).strip())
            if payload.get("name") and isinstance(payload.get("arguments"), dict):
                valid += 1
            else:
                malformed += 1
        except Exception:
            malformed += 1
    return malformed + max(0, valid - parsed_tool_call_count)


def _ensure_trace(agent_data: Any) -> dict[str, Any]:
    agent_data.extra_fields.setdefault(
        "code_agent_messages",
        list(getattr(agent_data, "messages", []) or []),
    )
    agent_data.extra_fields.setdefault("code_agent_terminal_reason", "")
    agent_data.extra_fields.setdefault("code_agent_parse_failures", 0)
    agent_data.extra_fields.setdefault("code_agent_tool_tail_chars", 0)
    agent_data.extra_fields.setdefault("code_agent_thinking_budget_reached", False)
    agent_data.extra_fields.setdefault("code_agent_thinking_unclosed", False)

    trace = agent_data.extra_fields.get("code_agent_trace")
    if isinstance(trace, dict):
        return trace
    trace = {
        "assistant_turns": [],
        "tool_calls": [],
        "parse_failures": 0,
        "tail_text_chars": 0,
        "terminal_reason": None,
        "thinking_unclosed": False,
    }
    agent_data.extra_fields["code_agent_trace"] = trace
    return trace


def _record_terminal(agent_data: Any, reason: str) -> None:
    trace = _ensure_trace(agent_data)
    trace["terminal_reason"] = reason
    agent_data.extra_fields["code_agent_terminal_reason"] = reason
    agent_data.metrics["code_agent_terminal"] = 1
    agent_data.metrics["code_agent_terminal_reason"] = reason


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

        if _truthy_env("CODE_AGENT_ENABLE_THINKING_EARLY_STOP"):
            return await self._handle_generating_state_with_thinking_stop(
                agent_data,
                sampling_params,
                ignore_termination=ignore_termination,
            )

        return await super()._handle_generating_state(
            agent_data,
            sampling_params,
            ignore_termination=ignore_termination,
        )

    async def _generate_once(self, agent_data: Any, sampling_params: dict[str, Any]):
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )

        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        return output

    def _append_generated_tokens(
        self,
        agent_data: Any,
        token_ids: list[int],
        log_probs: list[float] | None,
        *,
        mask_value: int = 1,
    ) -> None:
        agent_data.response_ids = token_ids
        agent_data.prompt_ids += token_ids
        agent_data.response_mask += [mask_value] * len(token_ids)
        if log_probs:
            agent_data.response_logprobs += log_probs
        elif agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(token_ids)

    def _append_control_text(self, agent_data: Any, text: str) -> list[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        self._append_generated_tokens(
            agent_data,
            list(token_ids),
            None,
            mask_value=0,
        )
        return list(token_ids)

    async def _handle_generating_state_with_thinking_stop(
        self,
        agent_data: Any,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        trace = _ensure_trace(agent_data)
        remaining_budget = max(0, int(self.response_length) - len(agent_data.response_mask))
        if remaining_budget <= 0:
            return AgentState.TERMINATED

        turn_budget = _budget_for_turn(agent_data) or remaining_budget
        turn_budget = min(turn_budget, remaining_budget)
        thinking_budget = _thinking_budget_for_turn(agent_data, turn_budget) or turn_budget
        thinking_budget = min(thinking_budget, turn_budget, remaining_budget)

        thinking_params = _sampling_params_with_turn_budget(
            sampling_params=sampling_params,
            turn_budget=thinking_budget,
            remaining_budget=remaining_budget,
        )
        thinking_output = await self._generate_once(agent_data, thinking_params)
        thinking_ids = list(thinking_output.token_ids)
        thinking_logprobs = list(thinking_output.log_probs) if thinking_output.log_probs else None
        thinking_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(thinking_ids, skip_special_tokens=False),
        )
        overall_finished = _contains_im_end(self.tokenizer, thinking_ids)
        thinking_closed = _contains_think_end(self.tokenizer, thinking_ids) or _THINK_END in thinking_text

        if overall_finished:
            self._append_generated_tokens(agent_data, thinking_ids, thinking_logprobs)
            agent_data.assistant_turns += 1

            tools = [tool.tool_schema for tool in self.tools.values()]
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
            parse_failures = _count_malformed_tool_calls(thinking_text, len(agent_data.tool_calls))
            tail = _assistant_tail_after_tool_call(thinking_text)
            trace["parse_failures"] += parse_failures
            trace["tail_text_chars"] += len(tail)
            agent_data.extra_fields["code_agent_parse_failures"] = trace["parse_failures"]
            agent_data.extra_fields["code_agent_tool_tail_chars"] = trace["tail_text_chars"]

            tool_calls_for_message = [
                {
                    "id": f"call_{len(trace['tool_calls']) + index}",
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for index, call in enumerate(agent_data.tool_calls[: self.max_parallel_calls])
            ]
            trace["assistant_turns"].append(
                {
                    "turn_index": int(agent_data.assistant_turns) - 1,
                    "text": thinking_text,
                    "thinking_closed": thinking_closed,
                    "tool_call_count": len(agent_data.tool_calls),
                    "parse_failures": parse_failures,
                    "tail_text_chars": len(tail),
                    "model_im_end": True,
                }
            )
            messages = agent_data.extra_fields.setdefault(
                "code_agent_messages",
                list(getattr(agent_data, "messages", []) or []),
            )
            assistant_message = {"role": "assistant", "content": thinking_text}
            if tool_calls_for_message:
                assistant_message["tool_calls"] = tool_calls_for_message
            messages.append(assistant_message)

            # In chat-completion style tool calling, <|im_end|> only ends the
            # assistant message. A complete tool call before it must still be
            # executed; otherwise formal submissions are silently dropped.
            if agent_data.tool_calls:
                return AgentState.PROCESSING_TOOLS

            trace["terminal_reason"] = "model_im_end"
            return AgentState.TERMINATED

        self._append_generated_tokens(agent_data, thinking_ids, thinking_logprobs)

        if not thinking_closed:
            trace["thinking_budget_reached"] = True
            agent_data.extra_fields["code_agent_thinking_budget_reached"] = True
            agent_data.metrics["code_agent_thinking_budget_reached"] = 1
            self._append_control_text(agent_data, _EARLY_STOPPING_TEXT)
            thinking_text += _EARLY_STOPPING_TEXT
            thinking_closed = True

        remaining_budget = max(0, int(self.response_length) - len(agent_data.response_mask))
        action_budget = max(0, min(turn_budget - len(thinking_ids), remaining_budget))
        action_text = ""
        if action_budget > 0:
            action_params = _sampling_params_with_turn_budget(
                sampling_params=sampling_params,
                turn_budget=action_budget,
                remaining_budget=remaining_budget,
            )
            action_params.pop("stop", None)
            action_output = await self._generate_once(agent_data, action_params)
            action_ids = list(action_output.token_ids)
            action_logprobs = list(action_output.log_probs) if action_output.log_probs else None
            action_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(action_ids, skip_special_tokens=False),
            )
            self._append_generated_tokens(agent_data, action_ids, action_logprobs)

        assistant_text = thinking_text + action_text
        agent_data.assistant_turns += 1

        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        tools = [tool.tool_schema for tool in self.tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
        parse_failures = _count_malformed_tool_calls(assistant_text, len(agent_data.tool_calls))
        tail = _assistant_tail_after_tool_call(assistant_text)
        trace["parse_failures"] += parse_failures
        trace["tail_text_chars"] += len(tail)
        agent_data.extra_fields["code_agent_parse_failures"] = trace["parse_failures"]
        agent_data.extra_fields["code_agent_tool_tail_chars"] = trace["tail_text_chars"]

        tool_calls_for_message = [
            {
                "id": f"call_{len(trace['tool_calls']) + index}",
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments,
                },
            }
            for index, call in enumerate(agent_data.tool_calls[: self.max_parallel_calls])
        ]
        messages = agent_data.extra_fields.setdefault(
            "code_agent_messages",
            list(getattr(agent_data, "messages", []) or []),
        )
        assistant_message = {"role": "assistant", "content": assistant_text}
        if tool_calls_for_message:
            assistant_message["tool_calls"] = tool_calls_for_message
        messages.append(assistant_message)

        trace["assistant_turns"].append(
            {
                "turn_index": int(agent_data.assistant_turns) - 1,
                "text": assistant_text,
                "thinking_closed": True,
                "tool_call_count": len(agent_data.tool_calls),
                "parse_failures": parse_failures,
                "tail_text_chars": len(tail),
            }
        )

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data) -> AgentState:
        state = await super()._handle_processing_tools_state(agent_data)
        if getattr(agent_data, "code_agent_terminal", False):
            reason = getattr(agent_data, "code_agent_terminal_reason", "terminal")
            _record_terminal(agent_data, reason)
            return AgentState.TERMINATED
        return state

    async def _call_tool(
        self,
        tool_call,
        tools_kwargs: dict[str, Any],
        agent_data: Any,
    ) -> tuple[ToolResponse, float | None, dict]:
        trace = _ensure_trace(agent_data)
        tool_name = getattr(tool_call, "name", "")
        raw_arguments = getattr(tool_call, "arguments", "")
        event: dict[str, Any] = {
            "name": tool_name,
            "arguments": None,
            "raw_arguments": raw_arguments,
            "executed": False,
            "parse_error": None,
            "observation": "",
            "reward": 0.0,
            "result": {},
        }

        tool = None
        instance_id = None
        tool_reward: float | None = None
        result: dict[str, Any] = {}
        tool_execution_response = ToolResponse(text="")
        try:
            try:
                tool_args = json.loads(raw_arguments)
            except Exception as exc:
                event["parse_error"] = f"malformed_arguments_json: {exc}"
                trace["parse_failures"] += 1
                tool_execution_response = ToolResponse(text=f"Error when parsing tool call arguments: {exc}")
                tool_args = None

            if event["parse_error"] is None and (
                not isinstance(tool_args, dict)
                or "code" not in tool_args
                or not isinstance(tool_args.get("code"), str)
            ):
                event["arguments"] = tool_args if isinstance(tool_args, dict) else None
                event["parse_error"] = "missing_or_invalid_arguments.code"
                trace["parse_failures"] += 1
                tool_execution_response = ToolResponse(
                    text="Error when parsing tool call: missing required string argument 'code'."
                )

            if event["parse_error"] is None and tool_name not in self.tools:
                event["arguments"] = tool_args
                event["parse_error"] = f"unknown_tool: {tool_name}"
                trace["parse_failures"] += 1
                tool_execution_response = ToolResponse(text=f"Error when executing tool: unknown tool '{tool_name}'.")

            if event["parse_error"] is None:
                event["arguments"] = tool_args
                tool = self.tools[tool_name]
                kwargs = tools_kwargs.get(tool_name, {})
                instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
                tool_execution_response, tool_reward, result = await tool.execute(
                    instance_id,
                    tool_args,
                    agent_data=agent_data,
                )
                event["executed"] = True
                event["reward"] = float(tool_reward or 0.0)
                event["result"] = result or {}
        except Exception as exc:
            event["parse_error"] = f"execution_error: {exc}"
            trace["parse_failures"] += 1
            tool_execution_response = ToolResponse(text=f"Error when executing tool: {exc}")
            tool_reward = None
            result = {}
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        event["observation"] = tool_response_text or ""
        event["terminal"] = bool((result or {}).get("terminal"))
        event["terminal_reason"] = (result or {}).get("terminal_reason")
        trace["tool_calls"].append(event)
        agent_data.extra_fields["code_agent_trace"] = trace
        agent_data.extra_fields["code_agent_parse_failures"] = trace["parse_failures"]

        messages = agent_data.extra_fields.setdefault(
            "code_agent_messages",
            list(getattr(agent_data, "messages", []) or []),
        )
        messages.append(
            {
                "role": "tool",
                "name": tool_name,
                "content": tool_response_text or "",
            }
        )

        verdict = (result or {}).get("verdict")
        if event["executed"] and (event.get("terminal") or verdict in TERMINAL_VERDICTS):
            reason = event.get("terminal_reason") or verdict or "terminal"
            _record_terminal(agent_data, reason)

        return ToolResponse(text=tool_response_text), tool_reward, result
