"""
Custom compute_score for verl GRPO training.

verl's reward manager calls this function after each rollout trajectory.
The tool_rewards from ExecuteCodeTool.execute() are available in extra_info.
"""

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
            "exec_reward": 0.0,
            "fix_reward": 0.0,
            "num_tool_calls": 0,
        }

    last_step_reward = tool_rewards[-1]
    exec_reward = min(last_step_reward / 0.1, 1.0)

    fix_reward = 0.0
    if len(tool_rewards) >= 2:
        first_pass_rate = min(tool_rewards[0] / 0.1, 1.0)
        if first_pass_rate < 1.0 and exec_reward >= 1.0:
            fix_reward = 0.2

    total = exec_reward + fix_reward

    return {
        "score": total,
        "exec_reward": exec_reward,
        "fix_reward": fix_reward,
        "num_tool_calls": len(tool_rewards),
    }
