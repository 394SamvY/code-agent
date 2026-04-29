"""Parse verl tool-agent decoded output into a readable event structure."""

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


def parse_tool_output(output: str) -> dict[str, Any]:
    """Parse decoded multi-turn tool-agent output into ordered events.

    The parser is intentionally presentation-oriented. It does not affect reward
    or rollout execution; it only makes verl generation dumps easier to inspect.
    """
    events: list[dict[str, Any]] = []

    for part in _TAG_RE.split(output or ""):
        if not part:
            continue

        tool_call = _TOOL_CALL_RE.fullmatch(part)
        if tool_call:
            events.append(_parse_tool_call(tool_call.group(1).strip()))
            continue

        tool_response = _TOOL_RESPONSE_RE.fullmatch(part)
        if tool_response:
            events.append(
                {
                    "type": "tool_response",
                    "content": tool_response.group(1).strip(),
                }
            )
            continue

        text = _strip_role_prefix(part)
        if text:
            events.append(
                {
                    "type": "assistant_text",
                    "content": text,
                }
            )

    return {
        "format": "code_agent_tool_events_v1",
        "events": events,
        "num_events": len(events),
        "num_tool_calls": sum(1 for event in events if event["type"] == "tool_call"),
        "num_tool_responses": sum(
            1 for event in events if event["type"] == "tool_response"
        ),
    }


def add_structured_output(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a generation record with parsed `structured_output`."""
    enriched = dict(record)
    enriched["structured_output"] = parse_tool_output(str(record.get("output", "")))
    return enriched
