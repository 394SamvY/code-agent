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
    "You must test your code by calling the execute_code tool."
)

USER_PROMPT_TEMPLATE = (
    "Solve the following Python programming problem:\n\n"
    "{problem_description}"
)


# Few-shot 示例：教模型正确的工具调用格式
FEW_SHOT_EXAMPLE = [
    {
        "role": "user",
        "content": (
            "Solve the following Python programming problem:\n\n"
            "Write a function to find the sum of two numbers.\n"
            "assert add(2, 3) == 5"
        ),
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": '{"code": "def add(a, b):\\n    return a + b"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "content": "3/3 tests passed. All tests passed!",
    },
    {
        "role": "assistant",
        "content": "All tests passed. The function correctly adds two numbers.",
    },
]


def build_agentic_messages(problem_description: str) -> list[dict[str, str]]:
    """构建 multi-turn agent 的初始 messages（含 few-shot 示例）.

    tools 注入由 tokenizer.apply_chat_template(messages, tools=TOOLS_SCHEMA)
    在编码时自动完成。few-shot 示例教模型用正确的 <tool_call> 格式。
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_AGENTIC_PLAIN},
        *FEW_SHOT_EXAMPLE,
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            problem_description=problem_description
        )},
    ]


def build_one_shot_prompt(problem_description: str) -> str:
    """构建 one-shot 的 user message 文本."""
    return USER_PROMPT_ONE_SHOT.format(
        problem_description=problem_description
    )
