"""
Reward 函数
===========

Phase 1: code_execution_reward — 二元 execution reward
Phase 2: multi_turn_reward — 三组分（execution + format + behavioral）
"""

from __future__ import annotations

from src.agent.parser import (
    extract_code_from_completion,
    extract_last_written_code,
    parse_tool_calls,
)
from src.env.sandbox import execute_with_tests


def _extract_text(completion) -> str:
    """从 TRL 传来的 completion 中提取纯文本.

    """
    if isinstance(completion, list):
        texts = [m.get("content", "") for m in completion if isinstance(m, dict)]
        return "\n".join(texts)
    return str(completion)


# ---------------------------------------------------------------------------
# Phase 1: One-shot Reward
# ---------------------------------------------------------------------------


def code_execution_reward(
    prompts: list | None = None,
    completions: list | None = None,
    completion_ids: list | None = None,
    test_list: list[list[str]] | None = None,
    timeout: int = 5,
    **kwargs,
) -> list[float]:
    """One-shot reward: 执行代码 + 测试 → 1.0 / 0.0.

    TRL 0.28 的 reward_funcs 签名：
        reward_func(prompts, completions, completion_ids, **column_kwargs)
    """
    if completions is None:
        return []

    rewards: list[float] = []

    for i, completion in enumerate(completions):
        text = _extract_text(completion)
        code = extract_code_from_completion(text)

        if prompts and i < len(prompts):
            p = _extract_text(prompts[i]) if isinstance(prompts[i], list) else str(prompts[i])
            if "def " in p and not code.startswith("def "):
                code = p + code

        tests = test_list[i] if test_list else []
        if not tests:
            rewards.append(0.0)
            continue

        all_passed = all(
            execute_with_tests(code, test, timeout=timeout).success
            for test in tests
        )
        rewards.append(1.0 if all_passed else 0.0)

    return rewards


# ---------------------------------------------------------------------------
# Phase 2: Multi-turn Reward (三组分)
# ---------------------------------------------------------------------------


def make_multi_turn_reward(
    format_weight: float = 0.1,
    behavioral_weight: float = 0.05,
):
    """创建可配置权重的 multi-turn reward 函数.

    Args:
        format_weight: format_reward 的权重，默认 0.1
        behavioral_weight: behavioral_reward 每项的权重，默认 0.05。
            设为 0.0 可关闭 behavioral_reward（用于 ablation 实验）。

    Returns:
        reward 函数（签名兼容 TRL GRPOTrainer）
    """
    def reward_func(
        prompts: list | None = None,
        completions: list | None = None,
        completion_ids: list | None = None,
        test_list: list[list[str]] | None = None,
        **kwargs,
    ) -> list[float]:
        if completions is None:
            return []

        rewards: list[float] = []

        for i, completion in enumerate(completions):
            text = _extract_text(completion)
            tests = test_list[i] if test_list else []

            # ---- 1. Execution Reward (主信号) ----
            code = extract_last_written_code(text)
            if not code:
                code = extract_code_from_completion(text)

            if code and tests:
                all_passed = all(
                    execute_with_tests(code, t).success for t in tests
                )
                exec_r = 1.0 if all_passed else 0.0
            else:
                exec_r = 0.0

            # ---- 2. Format Reward (辅助信号) ----
            tool_calls = parse_tool_calls(text)
            format_r = format_weight if len(tool_calls) > 0 else 0.0

            # ---- 3. Behavioral Reward (微弱信号，可关闭) ----
            behavioral_r = 0.0
            if behavioral_weight > 0:
                tool_names = [tc.name for tc in tool_calls]

                if "write_code" in tool_names and "run_tests" in tool_names:
                    behavioral_r += behavioral_weight

                if exec_r > 0 and "submit" in tool_names:
                    behavioral_r += behavioral_weight

            rewards.append(exec_r + format_r + behavioral_r)

        return rewards

    return reward_func


# 默认配置的 multi_turn_reward（向后兼容）
multi_turn_reward = make_multi_turn_reward()
