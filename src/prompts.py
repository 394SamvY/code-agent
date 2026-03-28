"""
Prompt 模板
===========

- Phase 1 (one-shot): 纯文本 prompt,无工具
- Phase 2 (agentic): 利用 Qwen 原生 tool calling,通过
  apply_chat_template(tools=TOOLS_SCHEMA) 自动注入工具描述
"""

from __future__ import annotations



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



SYSTEM_PROMPT_AGENTIC_PLAIN = (
    "You are an expert Python programmer. "
    "Write a solution, then call the execute_code tool to test it. "
    "If any test fails, analyze the error, fix your code, and call execute_code again."
)

USER_PROMPT_TEMPLATE = (
    "Solve the following Python programming problem:\n\n"
    "{problem_description}"
)


def build_agentic_messages(problem_description: str) -> list[dict]:
    """构建 multi-turn agent 的初始 messages（system + user）。

    tools 注入由 tokenizer.apply_chat_template(messages, tools=TOOLS_SCHEMA)
    在编码时自动完成。
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_AGENTIC_PLAIN},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            problem_description=problem_description
        )},
    ]


def build_one_shot_prompt(problem_description: str) -> str:
    """构建 one-shot 的 user message 文本."""
    return USER_PROMPT_ONE_SHOT.format(
        problem_description=problem_description
    )
