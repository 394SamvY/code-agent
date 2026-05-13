"""Code-agent 专用的 verl agent loop。

verl 上游的 ``ToolAgentLoop`` 已经负责大部分通用能力：

- 把 prompt 交给 rollout server 生成 assistant response
- 从 response 中解析 tool call
- 调用 verl BaseTool，并把 tool observation 拼回 messages
- 在多轮状态机中反复切换 GENERATING / PROCESSING_TOOLS

本项目只在它外面补 OJ-like 协议需要的语义：

- 记录 run_public_tests / submit_solution 的调用轨迹
- 把最后一次正式 submit 的 verdict 写成结构化 trace
- 识别 no_tool_call、malformed_tool_call、response_length_exceeded 等停止原因
- 给整条 trajectory 加一个总 tool call guard，避免模型无限工具循环

注意：这里不直接计算最终 reward；但会记录 ``code_agent_tool_events``。
最终训练 / 评测 reward 在 ``src/reward.py`` 里只消费这些真实工具执行事件。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
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
_TOOL_EVENTS_KEY = "code_agent_tool_events"


@register("code_agent_tool_agent")
class CodeAgentToolAgentLoop(ToolAgentLoop):
    """带 OJ-like 终止语义和 trace 字段的 ``ToolAgentLoop``。

    verl 通过 register 名称在配置里找到这个类：

    ``actor_rollout_ref.rollout.agent.default_agent_loop=code_agent_tool_agent``

    如果没有显式指定这个 loop，verl 会走默认 single-turn / tool agent 行为，
    就不会有本文件里的 OJ 终止原因、工具计数和结构化 dump 字段。
    """

    def _ensure_extra_fields(self, agent_data: AgentData) -> None:
        """初始化可选 dump 字段，保证并行 worker 的输出能被 verl concat。

        verl 的 ``DataProto.concat`` 要求不同 worker 返回的 non-tensor 字段集合
        基本一致。某些字段只有异常路径才会出现，因此这里提前放默认值，避免
        某个 batch 里一部分样本有字段、一部分样本没有字段导致拼接失败。
        """
        agent_data.extra_fields.setdefault(_TERMINAL_REASON_KEY, None)
        agent_data.extra_fields.setdefault(_PARSE_FAILURES_KEY, 0)
        agent_data.extra_fields.setdefault(_TOOL_TAIL_CHARS_KEY, 0)
        agent_data.extra_fields.setdefault(_TOOL_EVENTS_KEY, [])

    def _trace(self, agent_data: AgentData) -> dict[str, Any]:
        """返回当前 trajectory 的结构化 trace，必要时创建默认结构。

        trace 挂在 ``agent_data.extra_fields`` 上。agent loop 结束后，这些字段会
        跟随 verl 的 ``DataProto`` 一起回到 validation / training 侧，用于：

        - 写入 ``generations/*.jsonl``，方便复盘单条轨迹
        - 统计 terminal_reason、tool calls、judge runtime 等效率指标
        - 后续设计 reward 时区分“没提交”“最后一次提交失败”“AC 后继续工具”等行为

        这里记录的是观测和诊断信息；``code_agent_tool_events`` 是最终 reward
        的权威输入，reward 口径仍由 ``src/reward.py`` 决定。
        """
        self._ensure_extra_fields(agent_data)
        trace = agent_data.extra_fields.get(_TRACE_KEY)
        if not isinstance(trace, dict):
            trace = {
                # 所有 tool call 的总次数，run_public_tests 和 submit_solution 都计入。
                "num_tool_calls": 0,
                # 模型输出里疑似包含 <tool_call>，但 parser 没能解析成合法 tool call 的次数。
                "parse_failures": 0,
                # 本 loop 归一化后的停止原因，写入 generation schema 的 stop_reason。
                "terminal_reason": None,
                # 最近一次工具动作和最近一次 judge verdict，便于快速看最后状态。
                "last_action": None,
                "last_verdict": None,
                # 两类工具各自的业务计数；它们来自 tool result，而不是纯文本解析。
                "public_test_call_count": 0,
                "submission_count": 0,
                # 最后一次 submit_solution 的 judge 结果。没有 submit 时保持 no_submission。
                "final_submit_verdict": "no_submission",
                "final_submit_reward": 0.0,
                "final_submit_passed": None,
                "final_submit_total": None,
                # 是否曾经提交 AC，以及 AC 后是否又调用任何工具。
                "submit_accepted_seen": False,
                "has_tool_call_after_submit_accepted": False,
                "assistant_chars_after_submit_accepted": 0,
                "assistant_tokens_after_submit_accepted": 0,
                # rollout 级别的总 tool call guard，不等同于 submit 次数上限。
                "max_tool_calls": self._max_tool_calls(agent_data),
                # 工具调用 wall time 和 judge runtime，用来判断慢在模型还是环境。
                "tool_wall_seconds": 0.0,
                "max_tool_wall_seconds": 0.0,
                "judge_runtime_seconds": 0.0,
                "max_judge_runtime_seconds": 0.0,
                # 动作和 verdict 分布，保留细粒度行为统计。
                "action_counts": {},
                "verdict_counts": {},
            }
            agent_data.extra_fields[_TRACE_KEY] = trace
        return trace

    def _create_kwargs(self, agent_data: AgentData, tool_name: str) -> dict[str, Any]:
        """读取某个工具的 create_kwargs。

        verl dataset 会把每道题的 public/private tests、timeout、submit 上限等
        放在 ``extra_info.tools_kwargs.<tool>.create_kwargs`` 中。到 agent loop
        这里时，这些信息已经被展开到 ``agent_data.tools_kwargs``。

        兼容字符串形式是为了适配 parquet / JSON 序列化后的 payload。
        """
        tool_kwargs = agent_data.tools_kwargs.get(tool_name, {})
        create_kwargs = tool_kwargs.get("create_kwargs", {})
        if isinstance(create_kwargs, str):
            try:
                create_kwargs = json.loads(create_kwargs)
            except json.JSONDecodeError:
                create_kwargs = {}
        return create_kwargs if isinstance(create_kwargs, dict) else {}

    def _max_tool_calls(self, agent_data: AgentData) -> int:
        """计算整条 trajectory 允许的总 tool call 上限。

        这不是 submit 上限。submit 上限由 ``SubmitSolutionTool`` 内部用
        ``submission_count`` 和 ``max_submissions`` 控制；public test 上限同理由
        ``RunPublicTestsTool`` 控制。

        这里的 ``max_tool_calls`` 是更外层的 rollout guard，防止模型在工具状态机里
        长时间循环。默认值为：

        ``max_public_test_calls + max_submissions + 2``

        多出来的 2 次用于让模型有机会看到 limit-exceeded observation，而不是在业务
        工具刚触顶时被 agent loop 直接截断。调试时可以用环境变量
        ``CODE_AGENT_MAX_TOOL_CALLS`` 强行覆盖。
        """
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
        # 为 public-test limit 和 submission limit 各预留一次反馈空间。
        # 这个 hard cap 只是 rollout 保护，不应该抢在业务工具返回
        # limit-exceeded observation 之前把 trajectory 截断。
        return max_public + max_submissions + 2

    def _mark_terminal(self, agent_data: AgentData, reason: str) -> None:
        """把当前 trajectory 标记为终止，并同步写入 trace / extra_fields。

        ``_TERMINAL_KEY`` 是给当前 Python 对象内的状态判断用的；
        ``_TERMINAL_REASON_KEY`` 和 trace 则会随 ``DataProto`` 返回上层，用于
        generation dump 和指标统计。
        """
        setattr(agent_data, _TERMINAL_KEY, True)
        setattr(agent_data, _TERMINAL_REASON_KEY, reason)
        trace = self._trace(agent_data)
        trace["terminal_reason"] = reason
        agent_data.extra_fields[_TERMINAL_REASON_KEY] = reason

    def _terminal_reason(self, agent_data: AgentData) -> str | None:
        """读取当前已记录的终止原因。

        优先读 trace，是因为 trace 是最终 dump 的权威结构；如果 trace 尚未初始化，
        再回退到 extra_fields 里的轻量字段。
        """
        trace = agent_data.extra_fields.get(_TRACE_KEY)
        if isinstance(trace, dict) and trace.get("terminal_reason"):
            return str(trace["terminal_reason"])
        reason = agent_data.extra_fields.get(_TERMINAL_REASON_KEY)
        if reason:
            return str(reason)
        return None

    def _should_terminate(self, agent_data: AgentData) -> bool:
        """判断是否触发 agent loop 外层终止条件。

        当前只检查总 tool call guard。submit 次数耗尽、public test 次数耗尽等业务
        终止由各自工具写入 ``agent_data``，然后在 processing-tools 状态返回后生效。
        """
        trace = self._trace(agent_data)
        if int(trace.get("num_tool_calls", 0)) >= int(trace.get("max_tool_calls", self._max_tool_calls(agent_data))):
            self._mark_terminal(agent_data, "tool_call_limit_exhausted")
            return True
        return False

    def _record_no_tool_call_termination(self, agent_data: AgentData) -> None:
        """记录模型自然停止但没有产生 tool call 的情况。

        对 OJ agent 来说，纯文本停止通常意味着模型没有继续调用工具，也没有显式提交。
        如果此前已经有 accepted submit，这可能是合理收尾；如果从未 submit，则通常是
        无效长思考或未完成解题。
        """
        if self._terminal_reason(agent_data):
            return
        self._mark_terminal(agent_data, "no_tool_call")

    def _record_parse_failure_if_needed(self, agent_data: AgentData) -> bool:
        """检测“像 tool call 但 parser 没解析出来”的失败。

        上游 parser 只有解析成功时才会填充 ``agent_data.tool_calls``。这里额外检查
        原始 token 文本里是否出现 ``<tool_call>`` 标记：如果有标记但没有合法 tool
        call，说明模型大概率输出了畸形工具调用。记录 parse failure 后，上层会把
        terminal_reason 标成 ``malformed_tool_call``。
        """
        if agent_data.tool_calls:
            return False
        if not agent_data.response_ids:
            return False
        text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False)
        if "<tool_call>" not in text:
            return False
        trace = self._trace(agent_data)
        trace["parse_failures"] = int(trace.get("parse_failures", 0)) + 1
        trace["tool_tail_chars"] = len(text[-512:])
        agent_data.extra_fields[_PARSE_FAILURES_KEY] = trace["parse_failures"]
        agent_data.extra_fields[_TOOL_TAIL_CHARS_KEY] = trace["tool_tail_chars"]
        return True

    def _tool_call_arguments(self, tool_call) -> dict[str, Any]:
        """Decode the model-generated tool arguments for structured reward events."""
        try:
            arguments = json.loads(getattr(tool_call, "arguments", "{}") or "{}")
        except Exception:
            return {}
        return arguments if isinstance(arguments, dict) else {}

    def _code_hash(self, code: Any) -> str | None:
        if not isinstance(code, str):
            return None
        normalized = code.replace("\r\n", "\n").replace("\r", "\n")
        normalized = "\n".join(line.rstrip() for line in normalized.split("\n")).strip()
        if not normalized:
            return None
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    def _event_error_kind(self, result: dict[str, Any], observation: str) -> str | None:
        verdict = str(result.get("verdict") or "")
        first_failed = result.get("first_failed") if isinstance(result.get("first_failed"), dict) else {}
        stderr = str(first_failed.get("stderr") or observation or "")
        if "IndexError" in stderr or "index out of range" in stderr:
            return "index_error"
        if "KeyError" in stderr:
            return "key_error"
        if "RecursionError" in stderr or "maximum recursion depth exceeded" in stderr:
            return "recursion_error"
        if verdict == "time_limit_exceeded" or "Time Limit Exceeded" in stderr or "TLE" in stderr:
            return "time_limit_exceeded"
        if verdict in {"syntax_error", "runtime_error", "wrong_answer"}:
            return verdict
        return verdict or None

    def _record_tool_event(
        self,
        agent_data: AgentData,
        tool_call,
        tool_response: ToolResponse,
        result: dict[str, Any],
        tool_reward: float | None,
    ) -> None:
        """Append one authoritative tool execution event for reward computation."""
        self._ensure_extra_fields(agent_data)
        arguments = self._tool_call_arguments(tool_call)
        code = arguments.get("code")
        if code is not None and not isinstance(code, str):
            code = str(code)
        observation = tool_response.text or ""
        passed = int(result.get("passed") or 0)
        total = int(result.get("total") or 0)
        event = {
            "index": len(agent_data.extra_fields[_TOOL_EVENTS_KEY]),
            "tool": str(result.get("action") or getattr(tool_call, "name", "")),
            "verdict": str(result.get("verdict") or "tool_execution_error"),
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total else 0.0,
            "code": code,
            "code_hash": self._code_hash(code),
            "observation": observation,
            "first_failed": result.get("first_failed") if isinstance(result.get("first_failed"), dict) else None,
            "tool_reward": float(tool_reward or 0.0),
            "error_kind": self._event_error_kind(result, observation),
        }
        agent_data.extra_fields[_TOOL_EVENTS_KEY].append(event)

    def _record_tool_result(
        self,
        agent_data: AgentData,
        result: dict[str, Any],
        tool_reward: float | None = None,
    ) -> None:
        """把一次工具执行结果合并进 trace。

        ``result`` 是 OJ tool 返回的结构化 judge result，包含 action、verdict、
        passed/total、tests、submission_count 等。这里不解析 observation 文本，而是
        直接使用结构化 result，避免统计口径和展示文本耦合。

        ``tool_reward`` 是单次 tool 的即时 reward，主要用于记录最后一次 submit 的
        reward；最终训练 reward 不在这里聚合。
        """
        trace = self._trace(agent_data)
        trace["num_tool_calls"] = int(trace.get("num_tool_calls", 0)) + 1

        action = result.get("action") if isinstance(result, dict) else None
        verdict = result.get("verdict") if isinstance(result, dict) else None
        # 一旦曾经 submit accepted，后续任何工具调用都标记为 AC 后继续调用。
        # 这不一定改变当前 reward，但对后续 reward shaping 很重要。
        if action and trace.get("submit_accepted_seen"):
            trace["has_tool_call_after_submit_accepted"] = True
        if action:
            trace["last_action"] = action
            action_counts = trace.setdefault("action_counts", {})
            if isinstance(action_counts, dict):
                action_counts[action] = int(action_counts.get(action, 0)) + 1
        if verdict:
            trace["last_verdict"] = verdict
            verdict_counts = trace.setdefault("verdict_counts", {})
            if isinstance(verdict_counts, dict):
                verdict_counts[verdict] = int(verdict_counts.get(verdict, 0)) + 1
        if "public_test_call_count" in result:
            trace["public_test_call_count"] = int(result["public_test_call_count"])
        if "submission_count" in result:
            trace["submission_count"] = int(result["submission_count"])
        if action == "submit_solution":
            # final_submit_* 始终覆盖成最后一次正式提交的结果。
            # 如果 AC 后又 submit 失败，这里会记录最后一次失败 submit。
            trace["final_submit_verdict"] = str(verdict or "no_submission")
            trace["final_submit_reward"] = float(tool_reward or 0.0)
            trace["final_submit_passed"] = (
                int(result["passed"]) if result.get("passed") is not None else None
            )
            trace["final_submit_total"] = (
                int(result["total"]) if result.get("total") is not None else None
            )
            if verdict == "accepted":
                trace["submit_accepted_seen"] = True
        judge_runtime = 0.0
        # result["tests"] 里只有真正执行过的 case。OJ judge 遇到首个失败 case 会停，
        # 因此这里统计的是实际消耗的 judge runtime，不是全量 private tests 估计值。
        for case in result.get("tests", []) if isinstance(result, dict) else []:
            try:
                judge_runtime += float(case.get("runtime_seconds", 0.0))
            except Exception:
                continue
        if judge_runtime:
            trace["judge_runtime_seconds"] = float(trace.get("judge_runtime_seconds", 0.0)) + judge_runtime
            trace["max_judge_runtime_seconds"] = max(
                float(trace.get("max_judge_runtime_seconds", 0.0)),
                judge_runtime,
            )

        # 工具执行后立即再检查一次总 tool call guard，保证当前工具结果已经被记录，
        # 但不会继续进入下一轮生成。
        if int(trace.get("num_tool_calls", 0)) >= int(trace.get("max_tool_calls", self._max_tool_calls(agent_data))):
            self._mark_terminal(agent_data, self._terminal_reason(agent_data) or "tool_call_limit_exhausted")

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """处理 GENERATING 状态，并补充 OJ-specific 停止原因。

        上游 ``ToolAgentLoop`` 在这里完成一次模型生成，并尝试解析 tool call。
        我们额外处理三类情况：

        - response 打满 ``response_length``：``response_length_exceeded``
        - 文本里像 tool call 但 parser 失败：``malformed_tool_call``
        - 模型自然结束且没有 tool call：``no_tool_call``

        另外，上游在某些 TERMINATED 路径不会保留解析出的 ``tool_calls``，因此这里在
        TERMINATED 且有 response_ids 时再尝试解析一次，避免把合法工具调用误判为
        no_tool_call。
        """
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        trace = self._trace(agent_data)
        if trace.get("submit_accepted_seen") and agent_data.response_ids:
            text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False)
            trace["assistant_chars_after_submit_accepted"] = int(
                trace.get("assistant_chars_after_submit_accepted", 0)
            ) + len(text)
            trace["assistant_tokens_after_submit_accepted"] = int(
                trace.get("assistant_tokens_after_submit_accepted", 0)
            ) + len(agent_data.response_ids)
        if state == AgentState.TERMINATED and agent_data.response_ids:
            tools = [tool.tool_schema for tool in self.tools.values()]
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
        parse_failed = self._record_parse_failure_if_needed(agent_data)
        if state == AgentState.TERMINATED and not agent_data.tool_calls:
            if len(agent_data.response_mask) >= self.response_length:
                self._mark_terminal(agent_data, "response_length_exceeded")
            elif parse_failed:
                self._mark_terminal(agent_data, "malformed_tool_call")
            else:
                self._record_no_tool_call_termination(agent_data)
        if self._should_terminate(agent_data):
            return AgentState.TERMINATED
        return state

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """处理 PROCESSING_TOOLS 状态，并在工具返回后检查 OJ 终止条件。

        上游逻辑会执行所有待处理 tool call，把 observation 追加回 messages，然后通常
        回到 GENERATING。这里在进入和退出上游逻辑时各检查一次：

        - 进入前：如果上一轮已经触发 hard guard，就不要再执行新工具。
        - 退出后：如果工具返回导致 submission/public-test limit 或总 tool call guard
          触发，就直接终止。

        如果上游因为 response budget 不足而终止但没有写 terminal_reason，这里统一标为
        ``response_length_exceeded``。
        """
        if self._should_terminate(agent_data):
            return AgentState.TERMINATED
        state = await super()._handle_processing_tools_state(agent_data)
        if state == AgentState.TERMINATED and not self._terminal_reason(agent_data):
            self._mark_terminal(agent_data, "response_length_exceeded")
        if self._should_terminate(agent_data):
            return AgentState.TERMINATED
        return state

    async def _call_tool(
        self, tool_call, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """调用上游 tool，并记录 wall time 与结构化 result。

        真正的 tool 创建、参数校验、执行和 observation 构造都交给上游
        ``ToolAgentLoop`` 以及 ``src.verl_tools.oj_tools``。本方法只包一层计时和 trace
        记录，保持行为和上游工具协议解耦。
        """
        started_at = time.perf_counter()
        tool_response, tool_reward, result = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        elapsed = time.perf_counter() - started_at
        trace = self._trace(agent_data)
        trace["tool_wall_seconds"] = float(trace.get("tool_wall_seconds", 0.0)) + elapsed
        trace["max_tool_wall_seconds"] = max(
            float(trace.get("max_tool_wall_seconds", 0.0)),
            elapsed,
        )
        structured_result = result if isinstance(result, dict) else {}
        self._record_tool_event(agent_data, tool_call, tool_response, structured_result, tool_reward)
        self._record_tool_result(agent_data, structured_result, tool_reward)
        return tool_response, tool_reward, result
