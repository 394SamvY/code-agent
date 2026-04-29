"""Tests for structured parsing of verl tool-agent generation text."""

from __future__ import annotations

import json
import sys
import tempfile
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.trajectory_parser import add_structured_output, parse_tool_output

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "parse_verl_generations.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("parse_verl_generations", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT_MODULE = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)
parse_file = _SCRIPT_MODULE.parse_file


def test_parse_tool_output_extracts_ordered_events():
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

    parsed = parse_tool_output(output)

    assert parsed["format"] == "code_agent_tool_events_v1"
    assert parsed["num_tool_calls"] == 2
    assert parsed["num_tool_responses"] == 1
    assert [event["type"] for event in parsed["events"]] == [
        "assistant_text",
        "tool_call",
        "tool_response",
        "assistant_text",
        "tool_call",
    ]
    assert parsed["events"][1]["name"] == "run_public_tests"
    assert parsed["events"][1]["arguments"]["code"] == "print(1)"
    assert "wrong_answer" in parsed["events"][2]["content"]
    assert parsed["events"][4]["name"] == "submit_solution"

    print("[PASS] test_parse_tool_output_extracts_ordered_events")


def test_parse_tool_output_keeps_malformed_tool_call():
    parsed = parse_tool_output("<tool_call>\nnot json\n</tool_call>")

    event = parsed["events"][0]
    assert event["type"] == "tool_call"
    assert event["name"] is None
    assert event["arguments"] is None
    assert event["parse_error"]

    print("[PASS] test_parse_tool_output_keeps_malformed_tool_call")


def test_add_structured_output_keeps_original_record():
    record = {
        "output": "<tool_call>\n{\"name\":\"submit_solution\",\"arguments\":{}}\n</tool_call>",
        "score": 0.0,
    }

    enriched = add_structured_output(record)

    assert record.keys() == {"output", "score"}
    assert enriched["score"] == 0.0
    assert enriched["structured_output"]["num_tool_calls"] == 1

    # Ensure it remains JSON serializable for jsonl dumping.
    json.dumps(enriched, ensure_ascii=False)

    print("[PASS] test_add_structured_output_keeps_original_record")


def test_parse_file_writes_structured_jsonl():
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

    assert output_path.name == "partial_0_structured.jsonl"
    assert record["structured_output"]["num_tool_calls"] == 1

    print("[PASS] test_parse_file_writes_structured_jsonl")


if __name__ == "__main__":
    test_parse_tool_output_extracts_ordered_events()
    test_parse_tool_output_keeps_malformed_tool_call()
    test_add_structured_output_keeps_original_record()
    test_parse_file_writes_structured_jsonl()
    print("\nAll tests passed!")
