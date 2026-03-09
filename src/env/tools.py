"""
Agent 工具集定义
====================================

4 种核心工具：write_code / run_tests / debug / submit
覆盖完整的 写代码 → 测试 → 调试 → 提交 工作流。

设计原则：
- 少而精，降低小模型的决策空间
- 每个工具返回丰富信息，减少不必要的交互轮数
- write_code 内置语法检查，run_tests 内置完整 traceback
"""

from __future__ import annotations

from typing import Any, Callable

from .sandbox import execute_code, execute_with_tests

# ---------------------------------------------------------------------------
# TOOLS_SCHEMA: OpenAI function calling 格式，供 Qwen apply_chat_template 使用
# ---------------------------------------------------------------------------

# 遵循 OpenAI Function Calling / Tool Calling 格式（已成为 LLM 工具调用的事实标准）。
# 该列表会传给 tokenizer.apply_chat_template(tools=TOOLS_SCHEMA)，
# 让模型知道有哪些工具可以调用、每个工具接受什么参数。
#
# 每个工具的结构：
#   "type": "function"            — 工具类型
#   "function.name"               — 工具名，模型输出中会引用此名称来发起调用
#   "function.description"        — 功能描述，模型据此判断何时使用该工具
#   "function.parameters"         — 遵循 JSON Schema 规范，定义参数类型和约束
#
# 设计上只保留 4 个工具（write_code / run_tests / debug / submit），
# 覆盖 "写码 → 测试 → 调试 → 提交" 的完整循环，降低小模型的决策复杂度。
TOOLS_SCHEMA = [
    # ---- write_code ----
    # 将完整代码存入 env_state["current_code"]（纯内存操作，不涉及文件路径）。
    # 每次调用都是全量替换，而非增量 diff，这样对小模型更友好且不会出现 patch 错位。
    # 写入后自动做 compile() 语法检查，有语法错误会立刻反馈给模型。
    {
        "type": "function",
        "function": {
            "name": "write_code",
            "description": (
                "Write or replace the current solution code. "
                "Automatically checks syntax and reports errors."
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
    # ---- run_tests ----
    # 无参数。取出 env_state 中的当前代码和测试用例列表，
    # 逐条拼接后通过 sandbox 子进程执行，汇总 pass/fail 统计和失败的 traceback。
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run all test cases against the current code. "
                "Returns pass/fail count with detailed error messages and traceback."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    # ---- debug ----
    # 将当前代码 + 模型提供的调试片段拼接后在 sandbox 中执行，
    # 模型可以通过 print() 来检查中间变量，方便定位 bug。
    {
        "type": "function",
        "function": {
            "name": "debug",
            "description": (
                "Execute a code snippet for debugging. "
                "The current solution is available in scope. Use print() to inspect variables."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code with print() calls for debugging.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    # ---- submit ----
    # 无参数。提交当前代码作为最终答案，会重新跑一遍全部测试来判定通过与否。
    # 调用后 env_state["submitted"] 置为 True，episode 结束。
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Submit the current code as the final answer. "
                "This ends the episode. Make sure all tests pass before submitting."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 工具执行函数：每个 tool 接收 (env_state, **kwargs) -> str
# ---------------------------------------------------------------------------


def tool_write_code(env_state: dict[str, Any], code: str) -> str:
    """写入代码并自动检查语法。"""
    # 全量替换内存中的代码（不写磁盘），只有 run_tests/debug 时才会写临时文件执行
    env_state["current_code"] = code
    env_state["last_traceback"] = ""

    try:
        # compile() 仅做语法检查（不会真正执行），快速反馈语法错误
        compile(code, "<solution>", "exec")
        return "Code saved successfully. Syntax OK."
    except SyntaxError as e:
        return f"Code saved but has syntax error at line {e.lineno}: {e.msg}. Please fix it."


def tool_run_tests(env_state: dict[str, Any]) -> str:
    """运行全部测试，返回详细结果（含 traceback）。"""
    code = env_state.get("current_code", "")
    if not code.strip():
        return "Error: No code written yet. Use write_code first."

    test_list: list[str] = env_state.get("test_list", [])
    if not test_list:
        return "Error: No test cases available."

    passed, failed = 0, 0
    failure_details: list[str] = []

    # 逐条测试：每条 test 与当前代码拼接后在 sandbox 子进程中执行
    for i, test in enumerate(test_list):
        result = execute_with_tests(
            code, test, timeout=env_state.get("timeout", 5)
        )
        if result.success:
            passed += 1
        else:
            failed += 1
            tb = result.stderr.strip() if result.stderr else "Unknown error"
            # traceback 过长时截断，只保留首尾关键行，避免输出 token 爆炸
            tb_lines = tb.split("\n")
            if len(tb_lines) > 10:
                tb = "\n".join(tb_lines[:3] + ["  ..."] + tb_lines[-5:])
            failure_details.append(
                f"Test {i + 1}: {test.strip()}\n{tb}"
            )
            env_state["last_traceback"] = result.stderr

    # 汇总结果：通过/失败数 + 失败详情（含 traceback），给模型足够的调试信息
    summary = f"{passed} passed, {failed} failed out of {len(test_list)} tests."
    if failure_details:
        summary += "\n\nFailure details:\n" + "\n---\n".join(failure_details)
    else:
        summary += " All tests passed! You can now submit."
    return summary


def tool_debug(env_state: dict[str, Any], code: str) -> str:
    """在当前代码作用域中执行调试代码片段。"""
    current = env_state.get("current_code", "")
    # 将当前代码 + 调试片段拼接，这样调试代码可以直接调用已定义的函数/变量
    full_code = f"{current}\n\n{code}"
    result = execute_code(full_code, timeout=env_state.get("timeout", 5))
    if result.timed_out:
        return "Debug execution timed out."
    output = result.stdout.strip()
    if result.returncode != 0:
        err_last = result.stderr.strip().split("\n")[-1] if result.stderr else ""
        if output:
            return f"Debug output:\n{output}\nError: {err_last}"
        return f"Error: {err_last}"
    return f"Debug output:\n{output}" if output else "(no output)"


def tool_submit(env_state: dict[str, Any]) -> str:
    """提交最终答案，结束 episode。"""
    code = env_state.get("current_code", "")
    if not code.strip():
        env_state["submitted"] = True
        env_state["submit_passed"] = False
        return "Submission failed: No code written."

    # 重新跑一遍全部测试来做最终判定（不复用之前 run_tests 的结果，防止代码被改过）
    test_list: list[str] = env_state.get("test_list", [])
    all_passed = True
    for test in test_list:
        result = execute_with_tests(
            code, test, timeout=env_state.get("timeout", 5)
        )
        if not result.success:
            all_passed = False
            break  # 快速失败，不需要继续跑剩余测试

    # 标记 episode 结束，外层 rollout 循环会据此终止
    env_state["submitted"] = True
    env_state["submit_passed"] = all_passed
    if all_passed:
        return "Accepted! All tests passed."
    return "Submission failed: Not all tests passed."


# ---------------------------------------------------------------------------
# 工具名 → 执行函数 的映射（供 CodeEnvironment 使用）
# ---------------------------------------------------------------------------

# 工具名 → 执行函数 的映射。
# 当模型输出工具调用（如 {"name": "write_code", "arguments": {...}}）时，
# 外层代码通过此字典查找对应函数并执行，将返回值作为 tool response 回传给模型。
TOOL_EXECUTORS: dict[str, Callable] = {
    "write_code": tool_write_code,
    "run_tests": tool_run_tests,
    "debug": tool_debug,
    "submit": tool_submit,
}
