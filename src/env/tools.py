"""
Agent 工具集定义
====================================

单工具设计：execute_code
接收完整代码 → 语法检查 → 跑全部测试 → 返回结果。
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .sandbox import execute_with_tests, execute_code

# ---------------------------------------------------------------------------
# TOOLS_SCHEMA: OpenAI function calling 格式，供 Qwen apply_chat_template 使用
# ---------------------------------------------------------------------------

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": (
                "Execute Python code and run all test cases. "
                "Returns pass/fail count with detailed error messages and traceback for failures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The complete Python function code.",
                    }
                },
                "required": ["code"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 工具执行函数
# ---------------------------------------------------------------------------


def _try_get_actual_value(code: str, test: str) -> str | None:
    """尝试从 assert 语句中提取函数调用，执行并返回实际值。

    例如 test = "assert foo(3) == 5"
    → 提取 foo(3)，执行后返回 repr(结果)
    """
    # 匹配 assert expr == expected 的模式
    match = re.match(r"assert\s+(.+?)\s*==\s*(.+)", test.strip())
    if not match:
        return None

    expr = match.group(1).strip()
    expected = match.group(2).strip()

    # 执行函数调用，捕获实际返回值
    eval_code = f"{code}\n\ntry:\n    _actual = {expr}\n    print(repr(_actual))\nexcept Exception as e:\n    print(f'<error: {{e}}>')"
    result = execute_code(eval_code, timeout=3)

    actual = result.stdout.strip() if result.stdout else None
    if actual:
        return f"  Your output: {actual}\n  Expected:    {expected}"
    return None


def tool_execute_code(env_state: dict[str, Any], code: str) -> str:
    """写入代码 + 跑全部测试，一步完成。"""
    # 1. 存储代码
    env_state["current_code"] = code
    env_state["last_traceback"] = ""

    # 2. 语法检查
    try:
        compile(code, "<solution>", "exec")
    except SyntaxError as e:
        return f"Syntax error at line {e.lineno}: {e.msg}. Please fix it."

    # 3. 跑测试
    test_list: list[str] = env_state.get("test_list", [])
    if not test_list:
        return "Code saved. No test cases available."

    passed, failed = 0, 0
    failure_details: list[str] = []

    for i, test in enumerate(test_list):
        result = execute_with_tests(
            code, test, timeout=env_state.get("timeout", 5)
        )
        if result.success:
            passed += 1
        else:
            failed += 1

            # 构造错误信息
            detail_parts = [f"Test {i + 1} FAILED:"]
            detail_parts.append(f"  {test.strip()}")

            # 提取错误类型
            stderr = result.stderr.strip() if result.stderr else ""
            if result.timed_out:
                detail_parts.append(f"  Error: Execution timed out")
            elif stderr:
                # 提取最后一行的错误类型（如 AssertionError, TypeError 等）
                last_line = stderr.split("\n")[-1].strip()
                detail_parts.append(f"  Error: {last_line}")

                # 尝试获取实际返回值 vs 期望值
                actual_vs_expected = _try_get_actual_value(code, test)
                if actual_vs_expected:
                    detail_parts.append(actual_vs_expected)

                # traceback：第一个失败测试给完整的，后续截断
                if not failure_details:
                    # 第一个失败：保留完整 traceback
                    tb_lines = stderr.split("\n")
                    if len(tb_lines) > 15:
                        tb = "\n".join(tb_lines[:5] + ["  ..."] + tb_lines[-8:])
                    else:
                        tb = stderr
                    detail_parts.append(f"  Traceback:\n{tb}")
                # 后续失败：不附 traceback，节省 token
            else:
                detail_parts.append(f"  Error: Unknown error")

            # stdout 信息（模型可能有 print 调试）
            if result.stdout and result.stdout.strip():
                detail_parts.append(f"  Print output: {result.stdout.strip()[:200]}")

            failure_details.append("\n".join(detail_parts))
            env_state["last_traceback"] = stderr

    # 记录测试结果
    if "test_results_history" in env_state:
        env_state["test_results_history"].append({"passed": passed, "total": len(test_list)})

    summary = f"{passed}/{len(test_list)} tests passed."
    if failure_details:
        summary += "\n\n" + "\n---\n".join(failure_details)
    else:
        summary += " All tests passed!"
    return summary


TOOL_EXECUTORS: dict[str, Callable] = {
    "execute_code": tool_execute_code,
}


# ---------------------------------------------------------------------------
# 生成 verl tool_config.yaml
# ---------------------------------------------------------------------------

_TOOL_NAME_TO_VERL_CLASS: dict[str, str] = {
    "execute_code": "src.verl_tools.execute_code_tool.ExecuteCodeTool",
}


def generate_tool_config_yaml() -> str:
    """从 TOOLS_SCHEMA 自动生成 verl 的 tool_config.yaml 内容。"""
    import yaml

    tools = []
    for schema in TOOLS_SCHEMA:
        name = schema["function"]["name"]
        tools.append({
            "class_name": _TOOL_NAME_TO_VERL_CLASS[name],
            "config": {"type": "native"},
            "tool_schema": schema,
        })
    return yaml.dump({"tools": tools}, default_flow_style=False, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    from pathlib import Path

    output = Path(__file__).resolve().parents[2] / "configs" / "verl" / "tool_config.yaml"
    content = generate_tool_config_yaml()
    output.write_text(content)
    print(f"Generated {output}")
