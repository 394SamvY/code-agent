"""
Prompt 模板
===========

- Phase 1 (one-shot): 纯文本 prompt,无工具
- Phase 2 (agentic): 利用 Qwen 原生 tool calling,通过
  apply_chat_template(tools=TOOLS_SCHEMA) 自动注入工具描述
"""

from __future__ import annotations

from src.data.dataset import CodeProblem, OJTestCase


# ---------------------------------------------------------------------------
# One-shot Prompt (no tools)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_ONE_SHOT = (
    "You are an expert Python programmer. Write a complete Python program "
    "that reads from stdin and writes to stdout. Output only code, no explanation."
)

USER_PROMPT_ONE_SHOT = """\
{problem_text}

Write the complete Python solution program. Output only code:

```python
"""



SYSTEM_PROMPT_AGENTIC_PLAIN = (
    "You are an expert Python programmer. "
    "Write complete Python programs that read from stdin and write to stdout. "
    "Use run_public_tests to debug against public tests. "
    "After the public tests look correct, use submit_solution for the full judge. "
    "If any test fails, analyze the feedback, fix the full program, and try again."
)

USER_PROMPT_TEMPLATE = (
    "Solve the following OJ-style programming problem:\n\n"
    "{problem_text}"
)


def _format_test_case(test: OJTestCase, index: int) -> str:
    return (
        f"Public test {index}\n"
        f"Input:\n{test.input.rstrip()}\n"
        f"Output:\n{test.output.rstrip()}"
    )


def format_problem_prompt(
    problem: CodeProblem | str,
    max_public_tests: int = 3,
) -> str:
    """Format a CodeProblem for model input."""
    if isinstance(problem, str):
        return problem

    parts: list[str] = []
    if problem.title:
        parts.append(f"Title: {problem.title}")
    parts.append(problem.problem_statement.strip())

    if problem.starter_code.strip():
        parts.append("Starter code:\n```python\n" + problem.starter_code.strip() + "\n```")

    public_tests = problem.public_tests[:max_public_tests]
    if public_tests:
        rendered = "\n\n".join(
            _format_test_case(test, index)
            for index, test in enumerate(public_tests, start=1)
        )
        parts.append("Public tests:\n" + rendered)

    parts.append(
        "Write a complete Python 3 program that reads from stdin and writes to stdout."
    )
    return "\n\n".join(part for part in parts if part)


def build_agentic_messages(problem: CodeProblem | str) -> list[dict]:
    """构建 multi-turn agent 的初始 messages（system + user）。

    tools 注入由 tokenizer.apply_chat_template(messages, tools=TOOLS_SCHEMA)
    在编码时自动完成。
    """
    problem_text = format_problem_prompt(problem)
    return [
        {"role": "system", "content": SYSTEM_PROMPT_AGENTIC_PLAIN},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            problem_text=problem_text
        )},
    ]


def build_one_shot_prompt(problem: CodeProblem | str) -> str:
    """构建 one-shot 的 user message 文本."""
    problem_text = format_problem_prompt(problem)
    return USER_PROMPT_ONE_SHOT.format(
        problem_text=problem_text
    )
