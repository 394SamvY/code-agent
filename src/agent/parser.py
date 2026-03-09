"""
Tool Call 解析器
================

从模型输出中提取工具调用。支持两种格式：
1. Qwen 原生 <tool_call>...</tool_call> 特殊 token 格式
2. Markdown ```json 代码块格式（1.5B 等小模型常见的 fallback）

格式 2 的存在是因为：Qwen2.5-Coder-1.5B-Instruct 虽然知道工具名和参数结构，
但经常输出 ```json 而非 <tool_call> 特殊 token。parser 兼容两种格式，
让 RL 训练能有非零的初始 reward 信号。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ToolCall:
    """解析后的 tool 调用."""

    name: str
    arguments: dict[str, str]


# ---- Pattern 1: Qwen 原生 <tool_call> 格式 ----
# <tool_call>
# {"name": "write_code", "arguments": {"code": "def foo(): ..."}}
# </tool_call>
_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*\})\s*</tool_call>", re.DOTALL
)

# ---- Pattern 2: Markdown ```json 代码块格式 ----
# ```json
# {"name": "write_code", "arguments": {"code": "..."}}
# ```
# 使用贪婪匹配 .* 而非 .*?，因为 JSON 中有嵌套的 {}（如 arguments 字典）
# 外层的 ```json ... ``` 定界符保证不会过度匹配
_JSON_BLOCK_PATTERN = re.compile(
    r"```json\s*\n(\{.*\})\s*\n?```", re.DOTALL
)

# 合法的工具名集合，用于从 JSON 中过滤出真正的 tool call
_VALID_TOOL_NAMES = {"write_code", "run_tests", "debug", "submit"}


def _fix_triple_quotes(raw: str) -> str:
    """修复模型输出中的 Python 三重引号。

    Qwen2.5-Coder-1.5B 经常在 JSON 中用 Python 的 \"\"\"...\"\"\" 包裹代码，
    而不是 JSON 合法的 "..." 加转义。例如：
        {"name": "write_code", "arguments": {"code": \"\"\"
    def add(a, b):
        return a + b
    \"\"\"}}

    此函数将三重引号及其内容替换为正确转义的 JSON 字符串。
    """
    pattern = re.compile(r'"""\s*\n?(.*?)\n?\s*"""', re.DOTALL)

    def replacer(m):
        content = m.group(1)
        # 转义 JSON 特殊字符
        content = content.replace("\\", "\\\\")
        content = content.replace('"', '\\"')
        content = content.replace("\n", "\\n")
        content = content.replace("\t", "\\t")
        return f'"{content}"'

    return pattern.sub(replacer, raw)


def _parse_tool_json(raw: str) -> ToolCall | None:
    """从 JSON 字符串中解析 ToolCall，返回 None 表示解析失败.

    先尝试直接解析，失败后尝试修复三重引号再解析。
    """
    for attempt_raw in [raw, _fix_triple_quotes(raw)]:
        try:
            obj = json.loads(attempt_raw)
            name = obj.get("name", "")
            if name not in _VALID_TOOL_NAMES:
                return None
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            return ToolCall(name=name, arguments=args)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def parse_tool_calls(text: str) -> list[ToolCall]:
    """从模型输出中解析所有 tool_call.

    优先匹配 <tool_call> 格式，如果没有再尝试 ```json 格式。
    """
    results: list[ToolCall] = []

    # 先尝试 <tool_call> 格式
    for match in _TOOL_CALL_PATTERN.finditer(text):
        tc = _parse_tool_json(match.group(1))
        if tc:
            results.append(tc)

    if results:
        return results

    # fallback: 尝试 ```json 格式
    for match in _JSON_BLOCK_PATTERN.finditer(text):
        tc = _parse_tool_json(match.group(1))
        if tc:
            results.append(tc)

    return results


def parse_first_tool_call(text: str) -> ToolCall | None:
    """解析第一个 tool_call，没有则返回 None."""
    calls = parse_tool_calls(text)
    return calls[0] if calls else None


def extract_code_from_completion(text: str) -> str:
    """从 one-shot completion 中提取代码.

    提取策略：
    1. 优先匹配 ```python ... ``` 代码块
    2. fallback：返回整段文本
    """
    pattern = re.compile(r"```python\s*\n?(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3].rstrip()
    return text.strip()


def extract_last_written_code(text: str) -> str:
    """从多轮轨迹文本中提取最后一次 write_code 调用的代码.

    同时搜索 <tool_call> 和 ```json 两种格式。
    """
    last_code = ""

    # 搜索 <tool_call> 格式
    for m in _TOOL_CALL_PATTERN.finditer(text):
        tc = _parse_tool_json(m.group(1))
        if tc and tc.name == "write_code":
            code = tc.arguments.get("code", "")
            if code:
                last_code = code

    # 搜索 ```json 格式
    for m in _JSON_BLOCK_PATTERN.finditer(text):
        tc = _parse_tool_json(m.group(1))
        if tc and tc.name == "write_code":
            code = tc.arguments.get("code", "")
            if code:
                last_code = code

    return last_code
