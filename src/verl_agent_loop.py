"""Code-agent verl AgentLoop implementations.（自定义 agent loop 实现）

加载位置：verl AgentLoopWorker Ray actor 内部（CPU 进程，每个样本一个实例）。
加载方式：configs/verl/code_agent_loop.yaml → hydra.utils.instantiate() → 此类。

继承关系：ToolAgentLoop → AgentLoopBase → ABC
覆写的三个方法：_handle_generating_state, _handle_processing_tools_state, _call_tool
新增的核心方法：_handle_generating_state_with_thinking_stop（两阶段生成）

注意：此模块运行在 AgentLoopWorker 进程内，不是 TaskRunner 进程。
      TaskRunner 侧的 patch（dump generation 等）在 src/verl_runtime_patch.py。
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


# ── 常量 ──────────────────────────────────────────────
_THINK_END = "</think>"  # Qwen3 thinking 块的关闭 token
# 当 thinking budget 耗尽但模型还没闭合 </think> 时，注入这段文本强制闭合
_EARLY_STOPPING_TEXT = (
    "\n\nConsidering the limited time by the user, "
    "I have to give the solution based on the thinking directly now.\n"
    "</think>\n\n"
)
# 从生成文本中提取完整 <tool_call> JSON 的正则
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


# ── 配置读取 ──────────────────────────────────────────
def _positive_int_from_env(name: str) -> int | None:
    """从环境变量读取正整数。eval 脚本 export 的参数（如 CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET=3072）"""
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
    """布尔开关——CODE_AGENT_ENABLE_THINKING_EARLY_STOP=1 → True"""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ── Budget 计算 ───────────────────────────────────────
def _budget_for_turn(agent_data: Any) -> int | None:
    """当前 assistant turn 可用总 token 数。

    首轮从 CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET 取（默认 3072），
    后续轮从 CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET 取（默认 2048）。
    assistant_turns=0 表示还没生成过，即将进行第一轮。
    """
    if int(getattr(agent_data, "assistant_turns", 0) or 0) <= 0:
        return _positive_int_from_env("CODE_AGENT_FIRST_ASSISTANT_TURN_TOKEN_BUDGET")
    return _positive_int_from_env("CODE_AGENT_FOLLOWUP_ASSISTANT_TURN_TOKEN_BUDGET")


def _thinking_budget_for_turn(agent_data: Any, turn_budget: int | None) -> int:
    """当前 assistant turn 中 thinking 阶段的 token 上限（默认 1024）。

    优先级：环境变量 > min(1024, turn_budget) > 1024 > turn_budget

    最终不大于 turn_budget——thinking 不能超过整轮预算。
    """
    budget = _positive_int_from_env("CODE_AGENT_THINKING_TOKEN_BUDGET")  # 优先读环境变量
    if budget is None:
        budget = min(1024, turn_budget or 1024)  # 未设置时用默认值
    return min(budget, turn_budget) if turn_budget else budget


def _sampling_params_with_turn_budget(
    *,
    sampling_params: dict[str, Any],
    turn_budget: int,
    remaining_budget: int,
) -> dict[str, Any]:
    """在 sampling_params 中注入 max_new_tokens，取 turn_budget 和 remaining_budget 的较小值。
    确保单轮生成不超过 turn budget，也不超过整条 trajectory 剩余 budget。
    """
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


# ── Token 分析工具 ─────────────────────────────────────
def _token_id(tokenizer: Any, token: str, fallback: int) -> int:
    """查 tokenizer 中某个特殊 token 的 ID，失败则返回 fallback 值。
    例如 Qwen3 tokenizer 中 </think> 的 ID 为 151668。
    """
    try:
        value = tokenizer.convert_tokens_to_ids(token)
        return int(value) if value is not None else fallback
    except Exception:
        return fallback


def _contains_im_end(tokenizer: Any, token_ids: list[int]) -> bool:
    """检查 token 序列中是否含有 Qwen3 的 <|im_end|>（ID 151645）。
    模型生成 <|im_end|> 表示本轮 assistant turn 结束。
    """
    return _token_id(tokenizer, "<|im_end|>", 151645) in token_ids


def _contains_think_end(tokenizer: Any, token_ids: list[int]) -> bool:
    """检查 token 序列中是否含有 Qwen3 的 </think>（ID 151668）。
    模型生成 </think> 表示 thinking 块闭合。
    """
    return _token_id(tokenizer, _THINK_END, 151668) in token_ids


def _assistant_tail_after_tool_call(text: str) -> str:
    """提取最后一个 </tool_call> 之后的多余文本（模型在工具调用后多余的絮叨）。"""
    matches = list(_TOOL_CALL_RE.finditer(text or ""))
    if not matches:
        return ""
    return text[matches[-1].end() :].strip()


def _count_malformed_tool_calls(text: str, parsed_tool_call_count: int) -> int:
    """统计模型生成的 tool call 文本中有多少个格式错误。
    把完整匹配（name + arguments 都合法）计入 valid，
    然后 malformed = (总匹配 - valid) + (valid - 解析出来的)
    """
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


# ── Trace 管理 ────────────────────────────────────────
def _ensure_trace(agent_data: Any) -> dict[str, Any]:
    """确保 agent_data.extra_fields 中有 code_agent_trace（首次写入时初始化）。
    所有工具函数默认初始化 extra_fields 的扁平字段，
    然后创建内含 assistant_turns / tool_calls 的 trace 字典。
    """
    agent_data.extra_fields.setdefault("code_agent_terminal_reason", "")
    agent_data.extra_fields.setdefault("code_agent_parse_failures", 0)
    agent_data.extra_fields.setdefault("code_agent_tool_tail_chars", 0)
    agent_data.extra_fields.setdefault("code_agent_thinking_budget_reached", False)
    agent_data.extra_fields.setdefault("code_agent_thinking_unclosed", False)

    trace = agent_data.extra_fields.get("code_agent_trace")
    if isinstance(trace, dict):
        return trace
    trace = {
        "assistant_turns": [],   # 每轮 assistant 的生成记录
        "tool_calls": [],        # 每个 tool call 的执行记录
        "parse_failures": 0,     # tool call JSON 解析失败次数
        "tail_text_chars": 0,    # </tool_call> 后的多余字符总数
        "terminal_reason": None, # 终止原因（None/accepted/submission_limit_exhausted）
        "thinking_unclosed": False, # 是否有未闭合的 <think>
    }
    agent_data.extra_fields["code_agent_trace"] = trace
    return trace


def _record_terminal(agent_data: Any, reason: str) -> None:
    """标记当前 agent loop 的终止原因并写入 trace。"""
    trace = _ensure_trace(agent_data)
    trace["terminal_reason"] = reason
    agent_data.extra_fields["code_agent_terminal_reason"] = reason
    agent_data.metrics["code_agent_terminal"] = 1
    agent_data.metrics["code_agent_terminal_reason"] = reason


class CodeAgentToolAgentLoop(ToolAgentLoop):
    """OJ-like tool agent loop with eval-time generation budget controls.

    继承 ToolAgentLoop，覆写三个方法 + 新增一个核心方法。
    通过 configs/verl/code_agent_loop.yaml 注册为 code_agent_tool_agent。
    """

    # ── 状态机 handler ─────────────────────────────────

    async def _handle_generating_state(
        self,
        agent_data,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        """GENERATING 状态 handler（覆写父类）。

        1. 计算当前 turn 的 per-turn budget（首轮 3072 / 后续 2048）
        2. 检查整条 trajectory 的 remaining budget
        3. 如果 CODE_AGENT_ENABLE_THINKING_EARLY_STOP=1 → 走两阶段生成
        4. 否则 → 走父类原生逻辑（单次 generate，不做 thinking 介入）
        """
        # ── 注入 per-turn budget ───────────────────
        turn_budget = _budget_for_turn(agent_data)
        if turn_budget is not None:
            # 检查整条 trajectory 还剩多少 token（response_length - 已生成的）
            remaining_budget = max(
                0,
                int(self.response_length) - len(agent_data.response_mask),
            )
            if remaining_budget <= 0:
                return AgentState.TERMINATED
            # 把 turn_budget 注入 sampling_params → max_new_tokens
            sampling_params = _sampling_params_with_turn_budget(
                sampling_params=sampling_params,
                turn_budget=turn_budget,
                remaining_budget=remaining_budget,
            )
            agent_data.metrics["code_agent_assistant_turn_token_budget"] = turn_budget
            agent_data.metrics["code_agent_assistant_turn_index"] = int(
                getattr(agent_data, "assistant_turns", 0) or 0
            )

        # ── 路由：thinking stop 开关 ────────────────
        if _truthy_env("CODE_AGENT_ENABLE_THINKING_EARLY_STOP"):
            return await self._handle_generating_state_with_thinking_stop(
                agent_data,
                sampling_params,
                ignore_termination=ignore_termination,
            )

        # 开关关闭时走父类原生逻辑（单次 generate，不拆分 thinking/action）
        return await super()._handle_generating_state(
            agent_data,
            sampling_params,
            ignore_termination=ignore_termination,
        )

    # ── 底层 generation 封装 ─────────────────────────

    async def _generate_once(self, agent_data: Any, sampling_params: dict[str, Any]):
        """调 SGLang server 生成一次。覆写父类只为了额外记录 num_preempted metrics。

        这是跟 SGLang 的直接通信点——prompt_ids + sampling_params → TokenOutput。
        两阶段生成通过这个方法调两次 SGLang（thinking 一次，action 一次）。
        """
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,     # 当前累积的 prompt（含之前轮次的 tool response）
                sampling_params=sampling_params,       # 含 max_new_tokens 的生成参数
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )

        # preemption: SGLang 调度器抢占比当前请求更高优先级的请求时发生
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
        """把生成的新 token 追加到 agent_data 的累积序列中。

        mask_value=1: 模型生成的 token（在 response_mask 中标记为 1）
        mask_value=0: 控制 token（如注入的 _EARLY_STOPPING_TEXT），不计入模型生成
        """
        agent_data.response_ids = token_ids
        agent_data.prompt_ids += token_ids
        agent_data.response_mask += [mask_value] * len(token_ids)
        if log_probs:
            agent_data.response_logprobs += log_probs
        elif agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(token_ids)

    def _append_control_text(self, agent_data: Any, text: str) -> list[int]:
        """注入控制文本（_EARLY_STOPPING_TEXT）到 token 序列。
        这些 token 的 mask_value=0，不计入模型生成的 response 部分。
        """
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
        """两阶段生成：thinking phase + action phase（★ 核心方法）。

        父类是单次 generate()，没法在 thinking 过长时介入。
        这里拆成两次 generate：
          Phase 1 (thinking): max_new_tokens = thinking_budget (1024)
            只生成 <think>...</think> 部分。
            如果 thinking_budget 耗尽但 </think> 还没出 → 注入 _EARLY_STOPPING_TEXT
          Phase 2 (action): max_new_tokens = turn_budget - thinking 已用 token
            在 </think> 之后继续生成 <tool_call>...</tool_call>

        两个特殊路径：
          a. 模型在 thinking 阶段就生成了 <|im_end|> → 直接结束，提取 tool call
          b. thinking 未闭合 → 注入 _EARLY_STOPPING_TEXT，然后生成 action

        返回值：PROCESSING_TOOLS（有 tool call）、TERMINATED（无 tool call 或预算耗尽）
        """
        trace = _ensure_trace(agent_data)
        remaining_budget = max(0, int(self.response_length) - len(agent_data.response_mask))
        if remaining_budget <= 0:
            return AgentState.TERMINATED

        # ── 计算预算 ────────────────────────────────
        turn_budget = _budget_for_turn(agent_data) or remaining_budget
        turn_budget = min(turn_budget, remaining_budget)
        thinking_budget = _thinking_budget_for_turn(agent_data, turn_budget) or turn_budget
        thinking_budget = min(thinking_budget, turn_budget, remaining_budget)

        # ── Phase 1: Thinking ────────────────────────
        # 只给 1024 token 让模型写 <think> 块
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
        # 判断 thinking 阶段：模型是否自己结束了（<|im_end|>）？think 是否闭合了（</think>）？
        overall_finished = _contains_im_end(self.tokenizer, thinking_ids)
        thinking_closed = _contains_think_end(self.tokenizer, thinking_ids) or _THINK_END in thinking_text

        # ── 路径 A：模型在 thinking 阶段就结束了 ──────
        # 即模型在 1024 token 内输出了 <think>...</think><tool_call>...<|im_end|>
        if overall_finished:
            self._append_generated_tokens(agent_data, thinking_ids, thinking_logprobs)
            agent_data.assistant_turns += 1

            # 从 response token 中提取 tool call
            tools = [tool.tool_schema for tool in self.tools.values()]
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
            parse_failures = _count_malformed_tool_calls(thinking_text, len(agent_data.tool_calls))
            tail = _assistant_tail_after_tool_call(thinking_text)
            trace["parse_failures"] += parse_failures
            trace["tail_text_chars"] += len(tail)
            agent_data.extra_fields["code_agent_parse_failures"] = trace["parse_failures"]
            agent_data.extra_fields["code_agent_tool_tail_chars"] = trace["tail_text_chars"]

            # dump 用的 messages 构建
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

            # <|im_end|> 只是结束 assistant message，
            # 如果前面有完整 tool call 必须返回 PROCESSING_TOOLS 去执行
            if agent_data.tool_calls:
                return AgentState.PROCESSING_TOOLS

            trace["terminal_reason"] = "model_im_end"
            return AgentState.TERMINATED

        # ── 路径 B：thinking 未结束，继续生成 action ──
        # 先把 thinking 阶段的 token 写入 agent_data
        self._append_generated_tokens(agent_data, thinking_ids, thinking_logprobs)

        # thinking 没闭合 → 注入 _EARLY_STOPPING_TEXT 强制闭合
        if not thinking_closed:
            trace["thinking_budget_reached"] = True
            agent_data.extra_fields["code_agent_thinking_budget_reached"] = True
            agent_data.metrics["code_agent_thinking_budget_reached"] = 1
            # 注入"鉴于时间有限，我必须基于目前的思考给出答案\n</think>"
            self._append_control_text(agent_data, _EARLY_STOPPING_TEXT)
            thinking_text += _EARLY_STOPPING_TEXT
            thinking_closed = True

        # ── Phase 2: Action ──────────────────────────
        # 用 turn_budget - thinking 已用 token 作为 action 预算
        remaining_budget = max(0, int(self.response_length) - len(agent_data.response_mask))
        action_budget = max(0, min(turn_budget - len(thinking_ids), remaining_budget))
        action_text = ""
        if action_budget > 0:
            action_params = _sampling_params_with_turn_budget(
                sampling_params=sampling_params,
                turn_budget=action_budget,
                remaining_budget=remaining_budget,
            )
            action_params.pop("stop", None)  # 去掉 stop tokens，让模型自由生成
            action_output = await self._generate_once(agent_data, action_params)
            action_ids = list(action_output.token_ids)
            action_logprobs = list(action_output.log_probs) if action_output.log_probs else None
            action_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(action_ids, skip_special_tokens=False),
            )
            self._append_generated_tokens(agent_data, action_ids, action_logprobs)

        # ── 合并结果 ─────────────────────────────────
        assistant_text = thinking_text + action_text
        agent_data.assistant_turns += 1

        # 终止条件检查
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # 提取 tool call
        tools = [tool.tool_schema for tool in self.tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
        parse_failures = _count_malformed_tool_calls(assistant_text, len(agent_data.tool_calls))
        tail = _assistant_tail_after_tool_call(assistant_text)
        trace["parse_failures"] += parse_failures
        trace["tail_text_chars"] += len(tail)
        agent_data.extra_fields["code_agent_parse_failures"] = trace["parse_failures"]
        agent_data.extra_fields["code_agent_tool_tail_chars"] = trace["tail_text_chars"]

        trace["assistant_turns"].append(
            {
                "turn_index": int(agent_data.assistant_turns) - 1,
                "text": assistant_text,
                "thinking_closed": True,   # 路径 B 保证了一定是闭合的
                "tool_call_count": len(agent_data.tool_calls),
                "parse_failures": parse_failures,
                "tail_text_chars": len(tail),
            }
        )

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        return AgentState.TERMINATED

    # ── 工具处理覆写 ─────────────────────────────────

    async def _handle_processing_tools_state(self, agent_data) -> AgentState:
        """覆写父类：在 tool 执行完成后检查 terminal 标记。

        父类正常执行 tool → 组装 tool response → tokenize 追加到 prompt。
        然后检查 OjTool 是否在 agent_data 上设置了 code_agent_terminal=True
        （accepted / submission_limit_exhausted），是则 TERMINATED。
        """
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
        """覆写父类：执行单个 tool call，加完整 trace 记录 + parse error 处理。

        父类只做 create → execute → release 并返回 ToolResponse。
        我们增加了：
          - 三层 parse error 检测（JSON 格式错误 / 缺少 code 参数 / 未知 tool）
          - 执行结果写入 code_agent_trace.tool_calls
          - verdict 检测：accepted / submission_limit_exceeded → _record_terminal

        tool 的生命周期： create → execute → release（每次调用都建新实例）
        tool state 持久化通过 agent_data.code_agent_oj_tool_state（普通属性）实现。
        """
        trace = _ensure_trace(agent_data)
        tool_name = getattr(tool_call, "name", "")
        raw_arguments = getattr(tool_call, "arguments", "")
        # event: 存入 trace.tool_calls 的完整记录
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
            # ── 第一层：JSON 解析错误 ───────────────
            try:
                tool_args = json.loads(raw_arguments)
            except Exception as exc:
                event["parse_error"] = f"malformed_arguments_json: {exc}"
                trace["parse_failures"] += 1
                tool_execution_response = ToolResponse(text=f"Error when parsing tool call arguments: {exc}")
                tool_args = None

            # ── 第二层：缺少 required 参数 ────────────
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

            # ── 第三层：未知 tool ─────────────────────
            if event["parse_error"] is None and tool_name not in self.tools:
                event["arguments"] = tool_args
                event["parse_error"] = f"unknown_tool: {tool_name}"
                trace["parse_failures"] += 1
                tool_execution_response = ToolResponse(text=f"Error when executing tool: unknown tool '{tool_name}'.")

            # ── 正常执行 ─────────────────────────────
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

        # ── 截断过长 tool response ──────────────────
        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # ── 写入 trace ───────────────────────────────
        event["observation"] = tool_response_text or ""
        event["terminal"] = bool((result or {}).get("terminal"))
        event["terminal_reason"] = (result or {}).get("terminal_reason")
        trace["tool_calls"].append(event)
        agent_data.extra_fields["code_agent_trace"] = trace
        agent_data.extra_fields["code_agent_parse_failures"] = trace["parse_failures"]

        # ── terminal 检测 ────────────────────────────
        # 只有正式提交的 accepted / submission_limit 才是 OJ terminal；
        # run_public_tests 可能也返回 accepted，但只代表 public tests 通过。
        verdict = (result or {}).get("verdict")
        action = (result or {}).get("action")
        is_submit_terminal = action == "submit_solution" and verdict in TERMINAL_VERDICTS
        if event["executed"] and (event.get("terminal") or is_submit_terminal):
            reason = event.get("terminal_reason") or verdict or "terminal"
            _record_terminal(agent_data, reason)

        return ToolResponse(text=tool_response_text), tool_reward, result
