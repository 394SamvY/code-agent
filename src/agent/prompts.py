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


def build_agentic_messages(problem_description: str, few_shot: bool = False) -> list[dict[str, str]]:
    """构建 multi-turn agent 的初始 messages（不含 tools 注入）.

    Args:
        problem_description: 题目描述
        few_shot: 是否加入 few-shot 示例

    tools 注入由 tokenizer.apply_chat_template(messages, tools=TOOLS_SCHEMA)
    在编码时自动完成。
    """
    if few_shot:
        # 注意：
        # 将包含大量 role 切换的多轮示例直接塞进 Qwen 的 tool-calling chat template，
        # 可能导致部分模型退化为首轮直接 EOS。
        # 这里改为在单条 user 消息中放一个紧凑示例轨迹，以提升稳定性。
        few_shot_prefix = (
            "You must follow this tool-calling style exactly.\n\n"
            "Example trajectory:\n"
            "Problem: Write a function to reverse a string.\n"
            "Assistant:\n"
            "<tool_call>\n"
            "{\"name\": \"write_code\", \"arguments\": {\"code\": \"def reverse_string(s):\\n"
            "    return s[::-2]\"}}\n"
            "</tool_call>\n"
            "Tool:\n"
            "Code saved successfully. Syntax OK.\n"
            "Assistant:\n"
            "<tool_call>\n"
            "{\"name\": \"run_tests\", \"arguments\": {}}\n"
            "</tool_call>\n"
            "Tool:\n"
            "0 passed, 1 failed out of 1 tests.\n"
            "Assistant:\n"
            "<tool_call>\n"
            "{\"name\": \"write_code\", \"arguments\": {\"code\": \"def reverse_string(s):\\n"
            "    return s[::-1]\"}}\n"
            "</tool_call>\n"
            "Tool:\n"
            "Code saved successfully. Syntax OK.\n"
            "Assistant:\n"
            "<tool_call>\n"
            "{\"name\": \"run_tests\", \"arguments\": {}}\n"
            "</tool_call>\n"
            "Tool:\n"
            "1 passed, 0 failed out of 1 tests. All tests passed! You can now submit.\n"
            "Assistant:\n"
            "<tool_call>\n"
            "{\"name\": \"submit\", \"arguments\": {}}\n"
            "</tool_call>\n\n"
            "Now solve the real problem below using the same pattern.\n\n"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_AGENTIC_PLAIN},
            {"role": "user", "content": few_shot_prefix + USER_PROMPT_TEMPLATE.format(
                problem_description=problem_description
            )},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_AGENTIC_PLAIN},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                problem_description=problem_description
            )},
        ]
    return messages


def build_one_shot_prompt(problem_description: str) -> str:
    """构建 one-shot 的 user message 文本."""
    return USER_PROMPT_ONE_SHOT.format(
        problem_description=problem_description
    )
