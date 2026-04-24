"""Custom compute_score for OJ-like two-action GRPO training."""

from __future__ import annotations


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Compute the OJ-like scalar reward and eval-friendly metrics.

    `score` 继续沿用训练奖励口径：

    - public tests: 0
    - failed submit: <= 0.2
    - accepted submit: 1.0

    同时额外暴露 `acc`，让 verl validation 可以直接按二值正确率聚合，
    而不是只看 shaped reward。
    """
    if extra_info is None:
        extra_info = {}

    tool_rewards = extra_info.get("tool_rewards", [])

    if not tool_rewards:
        return {
            "score": 0.0,
            "tool_reward": 0.0,
            "num_tool_calls": 0,
            "acc": 0.0,
        }

    # Tools already encode the OJ reward policy:
    # public tests = 0, failed submits <= 0.2, accepted submit = 1.0.
    total = max(float(reward) for reward in tool_rewards)
    accepted = 1.0 if total >= 1.0 else 0.0

    return {
        "score": total,
        "tool_reward": total,
        "num_tool_calls": len(tool_rewards),
        "acc": accepted,
    }
