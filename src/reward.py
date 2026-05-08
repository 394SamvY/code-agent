"""Custom compute_score for OJ-like two-action GRPO training."""

from __future__ import annotations

import re


_SUBMIT_OBSERVATION_RE = re.compile(
    r"submit_solution:\s*([a-z_]+)\.\s*(\d+)/(\d+)\s+tests passed\.",
    re.MULTILINE,
)


def _submit_reward(verdict: str, passed: int, total: int) -> float:
    if verdict == "accepted":
        return 1.0
    if verdict in {"wrong_answer", "runtime_error", "time_limit_exceeded"}:
        return 0.2 * (passed / total if total else 0.0)
    return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Compute the OJ-like scalar reward and eval-friendly metrics.

    The stable correctness metric is based on the final ``submit_solution``
    verdict, not the best historical tool reward.  The scalar ``score`` is the
    current training reward and can be adjusted independently later.
    """
    if extra_info is None:
        extra_info = {}

    tool_rewards = extra_info.get("tool_rewards", [])
    matches = list(_SUBMIT_OBSERVATION_RE.finditer(solution_str or ""))

    if not matches:
        return {
            "score": 0.0,
            "reward": 0.0,
            "num_tool_calls": len(tool_rewards),
            "acc": 0.0,
        }

    last_submit = matches[-1]
    verdict = last_submit.group(1)
    passed = int(last_submit.group(2))
    total = int(last_submit.group(3))
    reward = _submit_reward(verdict, passed, total)
    accepted = 1.0 if verdict == "accepted" else 0.0

    return {
        "score": reward,
        "reward": reward,
        "num_tool_calls": len(tool_rewards),
        "acc": accepted,
    }
