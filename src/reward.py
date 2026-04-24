"""Custom compute_score for OJ-like two-action GRPO training."""

from __future__ import annotations


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    if extra_info is None:
        extra_info = {}

    tool_rewards = extra_info.get("tool_rewards", [])

    if not tool_rewards:
        return {
            "score": 0.0,
            "tool_reward": 0.0,
            "num_tool_calls": 0,
        }

    # Tools already encode the OJ reward policy:
    # public tests = 0, failed submits <= 0.2, accepted submit = 1.0.
    total = max(float(reward) for reward in tool_rewards)

    return {
        "score": total,
        "tool_reward": total,
        "num_tool_calls": len(tool_rewards),
    }
