"""
Prompt 模板
===========

- Phase 1 (one-shot): 纯文本 prompt，无工具
- Phase 2 (agentic): 利用 Qwen 原生 tool calling，通过
  apply_chat_template(tools=TOOLS_SCHEMA) 自动注入工具描述
"""

from __future__ import annotations

from src.env.tools import TOOLS_SCHEMA

# ---------------------------------------------------------------------------
# Multi-turn Agentic Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_AGENTIC = (
    "You are an expert Python programmer. You have exactly 4 tools: "
    "write_code, run_tests, debug, submit. You MUST only use these tools.\n\n"
    "Workflow:\n"
    "1. Use write_code to save your complete Python function.\n"
    "2. Use run_tests to check if all tests pass.\n"
    "3. If tests fail, fix your code and use write_code again.\n"
    "4. When all tests pass, use submit to finalize."
)

USER_PROMPT_TEMPLATE = (
    "Solve the following Python programming problem:\n\n"
    "{problem_description}\n\n"
    "Write a Python function that satisfies the above requirements. "
    "Start by writing your solution with write_code, then test it."
)


# ---------------------------------------------------------------------------
# One-shot Prompt (Phase 1, no tools)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_ONE_SHOT = (
    "You are an expert Python programmer. Write a correct Python function "
    "to solve the given problem. Output only the function code, no explanation."
)

USER_PROMPT_ONE_SHOT = """\
{problem_description}

Write the solution function in Python:

```python
"""


# ---------------------------------------------------------------------------
# Helper: build chat messages
# ---------------------------------------------------------------------------

def build_agentic_messages(problem_description: str) -> list[dict[str, str]]:
    """构建 multi-turn agent 的初始 messages（不含 tools 注入）.

    tools 注入由 tokenizer.apply_chat_template(messages, tools=TOOLS_SCHEMA)
    在编码时自动完成。
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_AGENTIC},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            problem_description=problem_description
        )},
    ]


def build_one_shot_prompt(problem_description: str) -> str:
    """构建 one-shot 的 user message 文本."""
    return USER_PROMPT_ONE_SHOT.format(
        problem_description=problem_description
    )
