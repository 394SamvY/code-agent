"""
Agent 工具集定义
====================================

3 种核心工具：write_code / run_tests / submit
覆盖完整的 写代码 → 测试 → 提交 工作流。

设计原则：
- 少而精，降低模型的决策空间
- 每个工具返回丰富信息，减少不必要的交互轮数
- write_code 内置语法检查，run_tests 内置完整 traceback
"""

from __future__ import annotations

from typing import Any, Callable

from .sandbox import execute_with_tests

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
# 只保留 3 个工具（write_code / run_tests / submit），
# 覆盖 "写码 → 测试 → 提交" 的核心循环，最小化决策空间。
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
    # 全量替换内存中的代码（不写磁盘），只有 run_tests 时才会写临时文件执行
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

    # 记录本次测试结果（供 reward 计算使用）
    if "test_results_history" in env_state:
        env_state["test_results_history"].append({"passed": passed, "total": len(test_list)})

    summary = f"{passed} passed, {failed} failed out of {len(test_list)} tests."
    if failure_details:
        summary += "\n\nFailure details:\n" + "\n---\n".join(failure_details)
    else:
        summary += " All tests passed! You can now submit."
    return summary


def tool_submit(env_state: dict[str, Any]) -> str:
    """提交最终答案，结束 episode。"""
    code = env_state.get("current_code", "")
    if not code.strip():
        env_state["submitted"] = True
        env_state["submit_passed"] = False
        env_state["submit_pass_ratio"] = 0.0
        return "Submission failed: No code written."

    # 重新跑一遍全部测试来做最终判定（逐条计数以获得 pass_ratio）
    test_list: list[str] = env_state.get("test_list", [])
    passed = 0
    for test in test_list:
        result = execute_with_tests(
            code, test, timeout=env_state.get("timeout", 5)
        )
        if result.success:
            passed += 1

    total = len(test_list) if test_list else 1
    all_passed = passed == total and total > 0

    env_state["submitted"] = True
    env_state["submit_passed"] = all_passed
    env_state["submit_pass_ratio"] = passed / total if total > 0 else 0.0

    if all_passed:
        return "Accepted! All tests passed."
    return f"Submission failed: {passed}/{total} tests passed."


# ---------------------------------------------------------------------------
# 工具名 → 执行函数 的映射（供 CodeEnvironment 使用）
# ---------------------------------------------------------------------------

# 工具名 → 执行函数 的映射。
# 当模型输出工具调用（如 {"name": "write_code", "arguments": {...}}）时，
# 外层代码通过此字典查找对应函数并执行，将返回值作为 tool response 回传给模型。
TOOL_EXECUTORS: dict[str, Callable] = {
    "write_code": tool_write_code,
    "run_tests": tool_run_tests,
    "submit": tool_submit,
}


# ---------------------------------------------------------------------------
# 工具名 → verl BaseTool 类路径 的映射（用于生成 tool_config.yaml）
# ---------------------------------------------------------------------------

_TOOL_NAME_TO_VERL_CLASS: dict[str, str] = {
    "write_code": "src.verl_tools.write_code_tool.WriteCodeTool",
    "run_tests": "src.verl_tools.run_tests_tool.RunTestsTool",
    "submit": "src.verl_tools.submit_tool.SubmitTool",
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
