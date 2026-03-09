"""
多轮 Rollout
=============

实现 agent 与 CodeEnvironment 之间的多轮交互循环。

核心流程（每轮）：
    model.generate() → 解析 tool_call → env.execute_tool() → 拼接 observation

Token 管理策略：
    维护一个持续增长的 all_token_ids 列表，同时用 env_mask 标记每个 token 的来源：
    - env_mask[i] = 1 → 该 token 由 model 生成（agent token，需要计算 loss）
    - env_mask[i] = 0 → 该 token 来自 prompt 或 env 返回（不计算 loss）

    每轮交互后，用 apply_chat_template 重新编码完整 messages，
    然后与上一轮的 all_token_ids 做 diff，得出新增的 observation token。
    这样确保包含 <|im_start|>tool 等 Qwen 特殊 token，避免手动 encode 遗漏。

Qwen 原生 tool calling 格式：
    - 工具描述通过 apply_chat_template(tools=TOOLS_SCHEMA) 自动注入到 system prompt
    - model 输出 <tool_call>{"name":"...", "arguments":{...}}</tool_call>
    - observation 使用 role="tool"（Qwen 会编码为 <|im_start|>tool\n...）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.agent.parser import parse_first_tool_call
from src.agent.prompts import build_agentic_messages
from src.env.code_env import CodeEnvironment
from src.env.tools import TOOLS_SCHEMA


@dataclass
class RolloutResult:
    """一次多轮 rollout 的完整结果.

    Attributes:
        messages:      完整的对话历史 [system, user, assistant, tool, assistant, ...]
        reward:        最终 reward（由 env.final_reward 给出，全部测试通过=1.0）
        num_turns:     实际交互轮数（每次 model.generate 算一轮）
        submitted:     agent 是否执行了 submit
        all_token_ids: 完整轨迹的 token ids（prompt + 所有轮的 completion + observation）
        prompt_len:    初始 prompt 的 token 长度（用于在训练时切分 prompt/completion）
        env_mask:      与 all_token_ids 等长，标记每个 token 是 agent(1) 还是 env(0)
    """

    messages: list[dict[str, str]]
    reward: float
    num_turns: int
    submitted: bool

    all_token_ids: list[int] = field(default_factory=list)
    prompt_len: int = 0
    env_mask: list[int] = field(default_factory=list)


def _encode_messages(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    add_generation_prompt: bool = True,
) -> list[int]:
    """用 Qwen chat template + tools 将 messages 编码为 token ids.

    关键点：传入 tools=TOOLS_SCHEMA，让 Qwen tokenizer 自动在 system prompt 中
    注入工具描述，并正确处理 role="tool" 的特殊 token。

    Args:
        tokenizer:              Qwen tokenizer
        messages:               对话历史（role/content dict 列表）
        add_generation_prompt:  是否在末尾追加 <|im_start|>assistant\n
                                （True = 准备让 model 继续生成；
                                 False = 对话已结束，如 submit 后）

    Returns:
        token ids 列表（1D）
    """
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS_SCHEMA,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
    )
    # apply_chat_template 的返回类型不固定，统一转为 list[int]
    if isinstance(encoded, torch.Tensor):
        return encoded.squeeze(0).tolist()
    if hasattr(encoded, "input_ids"):
        ids = encoded.input_ids
        return ids[0] if isinstance(ids[0], list) else list(ids)
    if isinstance(encoded, list):
        return encoded
    return list(encoded)


def multi_turn_rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    problem_description: str,
    env: CodeEnvironment,
    max_turns: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> RolloutResult:
    """执行一次完整的多轮 rollout.

    整体流程：

        messages = [system, user]           ← 初始 prompt
        prompt_ids = encode(messages)       ← 编码为 token ids
        all_token_ids = prompt_ids          ← 开始构建完整轨迹
        env_mask = [0] * len(prompt_ids)    ← prompt 部分标记为 0

        for each turn:
            # 1. 模型生成
            new_ids = model.generate(all_token_ids)
            all_token_ids += new_ids        ← 拼接 agent 输出
            env_mask += [1] * len(new_ids)  ← agent token 标记为 1

            # 2. 解析 tool call
            tool_call = parse(decode(new_ids))
            if no tool_call: break          ← 模型没有调用工具，结束

            # 3. 执行工具，获取 observation
            observation = env.execute_tool(tool_call)

            # 4. 将 observation 追加到 messages，重新编码
            messages.append({role: "tool", content: observation})
            full_ids = encode(messages)

            # 5. 计算 observation 新增的 token（diff）
            obs_ids = full_ids[len(all_token_ids):]
            all_token_ids = full_ids        ← 更新为重新编码的完整序列
            env_mask += [0] * len(obs_ids)  ← env token 标记为 0

    这样得到的 (all_token_ids, env_mask) 可以直接送入 TRL GRPO：
    - all_token_ids[:prompt_len] → prompt_ids
    - all_token_ids[prompt_len:] → completion_ids
    - env_mask[prompt_len:]      → 传入 extra_fields["env_mask"]
                                    TRL 0.28 自动转为 tool_mask 并在 loss 中应用

    Args:
        model:               Qwen 模型（eval mode）
        tokenizer:           对应的 tokenizer
        problem_description: 题目描述文本
        env:                 CodeEnvironment 实例（已初始化 test_list）
        max_turns:           最大交互轮数
        max_new_tokens:      每轮最多生成的 token 数
        temperature:         采样温度（0 = greedy）

    Returns:
        RolloutResult 包含完整轨迹、reward、token ids 和 env_mask
    """
    # ── Step 0: 构建初始 messages 并编码 ──
    messages = build_agentic_messages(problem_description)
    # messages = [
    #   {"role": "system", "content": "You are an expert..."},
    #   {"role": "user",   "content": "Solve the following..."}
    # ]

    prompt_ids = _encode_messages(tokenizer, messages)
    prompt_len = len(prompt_ids)

    # all_token_ids: 持续增长的完整轨迹 token 序列
    all_token_ids: list[int] = list(prompt_ids)
    # env_mask: 与 all_token_ids 等长，prompt 部分全 0
    env_mask: list[int] = [0] * prompt_len

    num_turns = 0

    # ── Step 1~N: 多轮交互循环 ──
    for _turn in range(max_turns):

        # ---- 1. 将当前完整序列作为输入，让模型生成 ----
        input_ids = torch.tensor([all_token_ids], device=model.device)

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.95

            outputs = model.generate(input_ids, **gen_kwargs)
            # outputs.shape = [1, input_len + generated_len]

        # 截取本轮新生成的 token（去掉 input 部分）
        new_ids = outputs[0][len(all_token_ids):].tolist()
        if not new_ids:
            break  # 模型没有生成任何新 token（可能直接输出了 eos）

        # ---- 2. 将 agent 生成的 token 加入轨迹 ----
        all_token_ids.extend(new_ids)
        env_mask.extend([1] * len(new_ids))  # agent token → 1（计算 loss）
        num_turns += 1

        # 解码为文本，追加到 messages
        response_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response_text})

        # ---- 3. 解析 tool call ----
        tool_call = parse_first_tool_call(response_text)
        # tool_call 格式: ToolCall(name="write_code", arguments={"code": "..."})

        if tool_call is None:
            # 模型没有输出 <tool_call>，可能直接给了文本答案，结束 rollout
            break

        # ---- 4. 在环境中执行工具 ----
        observation = env.execute_tool(tool_call.name, **tool_call.arguments)
        # observation 示例:
        #   write_code → "Code saved successfully. Syntax OK."
        #   run_tests  → "2 passed, 1 failed out of 3 tests.\n\nFailure details:..."
        #   submit     → "Accepted! All tests passed."

        # ---- 5. submit 特殊处理：执行完直接结束 ----
        if tool_call.name == "submit":
            messages.append({"role": "tool", "content": observation})
            # 重新编码获取 observation 对应的 token ids
            # add_generation_prompt=False 因为不需要模型继续生成
            full_ids = _encode_messages(tokenizer, messages, add_generation_prompt=False)
            obs_ids_len = len(full_ids) - len(all_token_ids)
            if obs_ids_len > 0:
                all_token_ids = full_ids
                env_mask.extend([0] * obs_ids_len)  # observation → 0
            break

        # ---- 6. 非 submit 工具：追加 observation 并继续下一轮 ----
        messages.append({"role": "tool", "content": observation})

        # 重新编码完整 messages → 得到包含 <|im_start|>tool 特殊 token 的精确序列
        # 这比手动 tokenizer.encode(observation) 更准确，因为 Qwen 在 role=tool 前后
        # 有特殊的 control token，直接 encode 纯文本会漏掉
        full_ids = _encode_messages(tokenizer, messages)

        # 计算 observation 部分新增了多少 token（diff）
        obs_ids_len = len(full_ids) - len(all_token_ids)
        if obs_ids_len > 0:
            new_obs_ids = full_ids[len(all_token_ids):]
            all_token_ids.extend(new_obs_ids)
            env_mask.extend([0] * len(new_obs_ids))  # env token → 0（不计算 loss）

        # 同步：确保 all_token_ids 与重新编码的结果一致
        # （可能存在 tokenizer 对中间内容重新分词的微小差异）
        all_token_ids = full_ids

        # 安全对齐：如果重新编码导致 all_token_ids 变长了，补齐 env_mask
        if len(env_mask) < len(all_token_ids):
            env_mask.extend([0] * (len(all_token_ids) - len(env_mask)))

        # 检查环境是否已结束（submit 已在上面处理，这里主要防御性检查）
        if env.is_done:
            break

    # ── 收尾：如果 agent 耗尽轮数仍未 submit，强制提交 ──
    if not env.is_done:
        env.execute_tool("submit")

    return RolloutResult(
        messages=messages,
        reward=env.final_reward,       # 全部测试通过=1.0，否则=0.0
        num_turns=num_turns,
        submitted=env.is_done,
        all_token_ids=all_token_ids,
        prompt_len=prompt_len,
        env_mask=env_mask,
    )


def batch_rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    problems: list[dict],
    max_turns: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    timeout: int = 5,
) -> list[RolloutResult]:
    """对一批 problem 逐个串行执行 rollout.

    串行而非并行的原因：每道题的 rollout 长度不同（取决于 agent 行为），
    难以统一 batch 内的 padding。训练时由 MultiTurnGRPOTrainer 管理。

    Args:
        problems: dict 列表，每个 dict 包含 "prompt", "test_list", "entry_point"
        其余参数同 multi_turn_rollout

    Returns:
        与 problems 等长的 RolloutResult 列表
    """
    results: list[RolloutResult] = []

    for prob in problems:
        env = CodeEnvironment(
            problem_description=prob["prompt"],
            test_list=prob["test_list"],
            entry_point=prob.get("entry_point"),
            timeout=timeout,
        )
        result = multi_turn_rollout(
            model=model,
            tokenizer=tokenizer,
            problem_description=prob["prompt"],
            env=env,
            max_turns=max_turns,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        results.append(result)

    return results
