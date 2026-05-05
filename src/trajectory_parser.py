"""Parse verl tool-agent decoded output into readable trajectory structures."""

from __future__ import annotations

import json
import re
from typing import Any


_TAG_RE = re.compile(
    r"(<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>)",
    re.DOTALL,
)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
_ROLE_PREFIX_RE = re.compile(r"^(?:assistant|user|tool)\s*\n", re.IGNORECASE)


def _strip_role_prefix(text: str) -> str:
    value = text.strip()
    if value.lower() in {"assistant", "user", "tool"}:
        return ""
    while True:
        stripped = _ROLE_PREFIX_RE.sub("", value, count=1).strip()
        if stripped.lower() in {"assistant", "user", "tool"}:
            return ""
        if stripped == value:
            return value
        value = stripped


def _parse_tool_call(raw: str) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "tool_call",
        "raw": raw,
        "name": None,
        "arguments": None,
        "parse_error": None,
    }
    try:
        payload = json.loads(raw)
        event["name"] = payload.get("name")
        event["arguments"] = payload.get("arguments")
    except Exception as exc:
        event["parse_error"] = str(exc)
    return event


def to_messages(
    output: str,
    *,
    initial_messages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert decoded tool-agent output to OpenAI/HuggingFace chat messages."""
    messages = list(initial_messages or [])
    pending_tool_call_id: str | None = None
    tool_call_index = 0

    for part in _TAG_RE.split(output or ""):
        if not part:
            continue

        tool_call_match = _TOOL_CALL_RE.fullmatch(part)
        if tool_call_match:
            tool_call = _parse_tool_call(tool_call_match.group(1).strip())
            name = tool_call.get("name")
            arguments = tool_call.get("arguments")
            if not name or tool_call.get("parse_error"):
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"<tool_call>\n{tool_call.get('raw', '')}\n</tool_call>",
                    }
                )
                pending_tool_call_id = None
                continue

            tool_call_id = f"call_{tool_call_index}"
            tool_call_index += 1
            pending_tool_call_id = tool_call_id
            content = None
            if (
                messages
                and messages[-1].get("role") == "assistant"
                and "tool_calls" not in messages[-1]
                and messages[-1].get("content")
            ):
                content = messages.pop()["content"]
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(
                                    arguments or {},
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                }
            )
            continue

        tool_response_match = _TOOL_RESPONSE_RE.fullmatch(part)
        if tool_response_match:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": pending_tool_call_id or f"call_{tool_call_index}",
                    "content": tool_response_match.group(1).strip(),
                }
            )
            pending_tool_call_id = None
            continue

        text = _strip_role_prefix(part)
        if text:
            messages.append(
                {
                    "role": "assistant",
                    "content": text,
                }
            )

    return messages


def add_messages(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a generation record with standard chat messages."""
    enriched = dict(record)
    enriched["messages"] = to_messages(str(record.get("output", "")))
    return enriched
