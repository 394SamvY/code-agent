"""
Prompt 模板
===========

- Phase 1 (one-shot): 纯文本 prompt,无工具
- Phase 2 (agentic): 利用 Qwen 原生 tool calling,通过
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
    "4. When all tests pass, use submit to finalize.\n\n"
    "IMPORTANT: Always start by calling write_code. Never output code as plain text. "
    "Never call run_tests before write_code."
)

USER_PROMPT_TEMPLATE = (
    "Solve the following Python programming problem:\n\n"
    "{problem_description}\n\n"
    "Write a Python function that satisfies the above requirements. "
    "Start by writing your solution with write_code, then test it."
)

# ---------------------------------------------------------------------------
# Few-shot 示例：教模型正确的工具调用流程
# ---------------------------------------------------------------------------

# 示例 1：一次通过的简单流程
_FEW_SHOT_USER_1 = (
    "Solve the following Python programming problem:\n\n"
    "Write a function to find the sum of all elements in a list.\n"
    "assert list_sum([1, 2, 3]) == 6\n\n"
    "Write a Python function that satisfies the above requirements. "
    "Start by writing your solution with write_code, then test it."
)

_FEW_SHOT_ASSISTANT_1 = (
    '<tool_call>\n'
    '{"name": "write_code", "arguments": {"code": "def list_sum(lst):\\n'
    '    return sum(lst)"}}\n'
    '</tool_call>'
)

_FEW_SHOT_TOOL_1A = "Code saved successfully. Syntax OK."

_FEW_SHOT_ASSISTANT_1B = (
    '<tool_call>\n'
    '{"name": "run_tests", "arguments": {}}\n'
    '</tool_call>'
)

_FEW_SHOT_TOOL_1B = "3 passed, 0 failed out of 3 tests. All tests passed! You can now submit."

_FEW_SHOT_ASSISTANT_1C = (
    '<tool_call>\n'
    '{"name": "submit", "arguments": {}}\n'
    '</tool_call>'
)

_FEW_SHOT_TOOL_1C = "Accepted! All tests passed."

# 示例 2：先失败后修复的流程
_FEW_SHOT_USER_2 = (
    "Solve the following Python programming problem:\n\n"
    "Write a function to reverse a string.\n"
    "assert reverse_string('hello') == 'olleh'\n\n"
    "Write a Python function that satisfies the above requirements. "
    "Start by writing your solution with write_code, then test it."
)

_FEW_SHOT_ASSISTANT_2 = (
    '<tool_call>\n'
    '{"name": "write_code", "arguments": {"code": "def reverse_string(s):\\n'
    '    return s[::-2]"}}\n'
    '</tool_call>'
)

_FEW_SHOT_TOOL_2A = "Code saved successfully. Syntax OK."

_FEW_SHOT_ASSISTANT_2B = (
    '<tool_call>\n'
    '{"name": "run_tests", "arguments": {}}\n'
    '</tool_call>'
)

_FEW_SHOT_TOOL_2B = (
    "0 passed, 1 failed out of 1 tests.\n\n"
    "Failure details:\n"
    "Test 1: assert reverse_string('hello') == 'olleh'\n"
    "AssertionError"
)

_FEW_SHOT_ASSISTANT_2C = (
    "The step should be -1, not -2. Let me fix it.\n\n"
    '<tool_call>\n'
    '{"name": "write_code", "arguments": {"code": "def reverse_string(s):\\n'
    '    return s[::-1]"}}\n'
    '</tool_call>'
)

_FEW_SHOT_TOOL_2C = "Code saved successfully. Syntax OK."

_FEW_SHOT_ASSISTANT_2D = (
    '<tool_call>\n'
    '{"name": "run_tests", "arguments": {}}\n'
    '</tool_call>'
)

_FEW_SHOT_TOOL_2D = "1 passed, 0 failed out of 1 tests. All tests passed! You can now submit."

_FEW_SHOT_ASSISTANT_2E = (
    '<tool_call>\n'
    '{"name": "submit", "arguments": {}}\n'
    '</tool_call>'
)

_FEW_SHOT_TOOL_2E = "Accepted! All tests passed."


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
    """构建 multi-turn agent 的初始 messages（含 few-shot 示例，不含 tools 注入）.

    tools 注入由 tokenizer.apply_chat_template(messages, tools=TOOLS_SCHEMA)
    在编码时自动完成。

    Few-shot 示例教模型两种关键流程：
    1. write_code → run_tests → submit（一次通过）
    2. write_code → run_tests(失败) → write_code(修复) → run_tests → submit（调试修复）
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_AGENTIC},
        # ---- Few-shot 示例 1：一次通过 ----
        {"role": "user", "content": _FEW_SHOT_USER_1},
        {"role": "assistant", "content": _FEW_SHOT_ASSISTANT_1},
        {"role": "tool", "content": _FEW_SHOT_TOOL_1A},
        {"role": "assistant", "content": _FEW_SHOT_ASSISTANT_1B},
        {"role": "tool", "content": _FEW_SHOT_TOOL_1B},
        {"role": "assistant", "content": _FEW_SHOT_ASSISTANT_1C},
        {"role": "tool", "content": _FEW_SHOT_TOOL_1C},
        # ---- Few-shot 示例 2：调试修复 ----
        {"role": "user", "content": _FEW_SHOT_USER_2},
        {"role": "assistant", "content": _FEW_SHOT_ASSISTANT_2},
        {"role": "tool", "content": _FEW_SHOT_TOOL_2A},
        {"role": "assistant", "content": _FEW_SHOT_ASSISTANT_2B},
        {"role": "tool", "content": _FEW_SHOT_TOOL_2B},
        {"role": "assistant", "content": _FEW_SHOT_ASSISTANT_2C},
        {"role": "tool", "content": _FEW_SHOT_TOOL_2C},
        {"role": "assistant", "content": _FEW_SHOT_ASSISTANT_2D},
        {"role": "tool", "content": _FEW_SHOT_TOOL_2D},
        {"role": "assistant", "content": _FEW_SHOT_ASSISTANT_2E},
        {"role": "tool", "content": _FEW_SHOT_TOOL_2E},
        # ---- 实际题目 ----
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            problem_description=problem_description
        )},
    ]


def build_one_shot_prompt(problem_description: str) -> str:
    """构建 one-shot 的 user message 文本."""
    return USER_PROMPT_ONE_SHOT.format(
        problem_description=problem_description
    )
