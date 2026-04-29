"""Tests for converting verl tool-agent generation text to messages."""

from __future__ import annotations

import json
import sys
import tempfile
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.trajectory_parser import add_messages, to_messages

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "parse_verl_generations.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("parse_verl_generations", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT_MODULE = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)
parse_file = _SCRIPT_MODULE.parse_file


def test_to_messages_extracts_ordered_tool_messages():
    output = """I will test the program.
<tool_call>
{"name": "run_public_tests", "arguments": {"code": "print(1)"}}
</tool_call>user
<tool_response>
run_public_tests: wrong_answer. 0/1 tests passed.
</tool_response>assistant
Now I will submit.
<tool_call>
{"name": "submit_solution", "arguments": {"code": "print(2)"}}
</tool_call>"""

    messages = to_messages(output)

    assert [message["role"] for message in messages] == [
        "assistant",
        "assistant",
        "tool",
        "assistant",
        "assistant",
    ]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "run_public_tests"
    assert (
        json.loads(messages[1]["tool_calls"][0]["function"]["arguments"])["code"]
        == "print(1)"
    )
    assert "wrong_answer" in messages[2]["content"]
    assert messages[4]["tool_calls"][0]["function"]["name"] == "submit_solution"

    print("[PASS] test_to_messages_extracts_ordered_tool_messages")


def test_to_messages_keeps_malformed_tool_call_as_text():
    messages = to_messages("<tool_call>\nnot json\n</tool_call>")

    assert messages[0]["role"] == "assistant"
    assert "not json" in messages[0]["content"]
    assert "tool_calls" not in messages[0]

    print("[PASS] test_to_messages_keeps_malformed_tool_call_as_text")


def test_add_messages_keeps_original_record():
    record = {
        "output": "<tool_call>\n{\"name\":\"submit_solution\",\"arguments\":{}}\n</tool_call>",
        "score": 0.0,
    }

    enriched = add_messages(record)

    assert record.keys() == {"output", "score"}
    assert enriched["score"] == 0.0
    assert enriched["messages"][0]["role"] == "assistant"
    assert enriched["messages"][0]["tool_calls"][0]["id"] == "call_0"
    assert "structured_output" not in enriched

    # Ensure it remains JSON serializable for jsonl dumping.
    json.dumps(enriched, ensure_ascii=False)

    print("[PASS] test_add_messages_keeps_original_record")


def test_parse_file_writes_messages_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "partial_0.jsonl"
        input_path.write_text(
            json.dumps(
                {
                    "output": "<tool_call>\n{\"name\":\"submit_solution\",\"arguments\":{}}\n</tool_call>",
                    "score": 0.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        output_path = parse_file(input_path)
        record = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.name == "partial_0_messages.jsonl"
    assert record["messages"][0]["tool_calls"][0]["function"]["name"] == "submit_solution"
    assert "structured_output" not in record

    print("[PASS] test_parse_file_writes_messages_jsonl")


def test_to_messages_converts_tool_calls_and_responses():
    output = """Thinking.
<tool_call>
{"name": "run_public_tests", "arguments": {"code": "print(1)"}}
</tool_call>
<tool_response>
run_public_tests: accepted. 1/1 tests passed.
</tool_response>"""

    messages = to_messages(output)

    assert [message["role"] for message in messages] == [
        "assistant",
        "assistant",
        "tool",
    ]
    assert messages[1]["content"] is None
    assert messages[1]["tool_calls"][0]["id"] == "call_0"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "run_public_tests"
    assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {
        "code": "print(1)"
    }
    assert messages[2]["tool_call_id"] == "call_0"

    print("[PASS] test_to_messages_converts_tool_calls_and_responses")


if __name__ == "__main__":
    test_to_messages_extracts_ordered_tool_messages()
    test_to_messages_keeps_malformed_tool_call_as_text()
    test_add_messages_keeps_original_record()
    test_parse_file_writes_messages_jsonl()
    test_to_messages_converts_tool_calls_and_responses()
    print("\nAll tests passed!")
