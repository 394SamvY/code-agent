"""
Lightweight OJ-like e2e protocol tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import CodeProblem, OJTestCase
from src.data.verl_dataset import problem_to_verl_record
from src.eval.evaluate import evaluate_multi_turn, evaluate_one_shot


SOLUTION = "import sys\nx = int(sys.stdin.read())\nprint(x + 1)"


class _Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return "prompt"


class _Choice:
    def __init__(self, text: str):
        self.text = text


class _Response:
    def __init__(self, text: str):
        self.choices = [_Choice(text)]


class _Completions:
    def __init__(self, responses: list[str]):
        self._responses = responses
        self._index = 0

    def create(self, **kwargs):
        text = self._responses[self._index]
        self._index += 1
        return _Response(text)


class _Client:
    def __init__(self, responses: list[str]):
        self.completions = _Completions(responses)


def _problem() -> CodeProblem:
    return CodeProblem(
        task_id="codecontests/e2e",
        dataset="codecontests",
        title="Add One",
        problem_statement="Read one integer and print it plus one.",
        public_tests=[OJTestCase(input="1\n", output="2\n")],
        private_tests=[OJTestCase(input="41\n", output="42\n")],
        time_limit_seconds=2.0,
    )


def _tool_call(name: str, code: str) -> str:
    return "<tool_call>\n" + json.dumps(
        {"name": name, "arguments": {"code": code}}
    ) + "\n</tool_call>"


def test_problem_to_verl_record_contains_create_kwargs():
    problem = _problem()

    record = problem_to_verl_record(problem)
    create_kwargs = record["extra_info"]["create_kwargs"]

    assert record["data_source"] == "codecontests"
    assert record["ability"] == "code"
    assert create_kwargs["max_submissions"] == 5
    assert create_kwargs["time_limit_seconds"] == 2.0
    assert create_kwargs["public_tests"] == [{"input": "1\n", "output": "2\n"}]
    assert create_kwargs["private_tests"] == [{"input": "41\n", "output": "42\n"}]

    print("[PASS] test_problem_to_verl_record_contains_create_kwargs")


def test_fake_one_shot_submits_private_tests():
    result = evaluate_one_shot(
        client=_Client(["```python\n" + SOLUTION + "\n```"]),
        tokenizer=_Tokenizer(),
        problems=[_problem()],
    )

    assert result["pass@1"] == 1.0
    assert result["results"][0]["verdict"] == "accepted"
    assert result["results"][0]["submission_history"][0]["passed"] == 1

    print("[PASS] test_fake_one_shot_submits_private_tests")


def test_fake_multi_turn_public_then_submit():
    result = evaluate_multi_turn(
        client=_Client([
            _tool_call("run_public_tests", SOLUTION),
            _tool_call("submit_solution", SOLUTION),
        ]),
        tokenizer=_Tokenizer(),
        problems=[_problem()],
        max_tool_calls=4,
    )

    item = result["results"][0]
    assert result["pass@1"] == 1.0
    assert item["verdict"] == "accepted"
    assert len(item["public_results_history"]) == 1
    assert len(item["submission_history"]) == 1
    assert item["num_tool_calls"] == 2

    print("[PASS] test_fake_multi_turn_public_then_submit")


if __name__ == "__main__":
    test_problem_to_verl_record_contains_create_kwargs()
    test_fake_one_shot_submits_private_tests()
    test_fake_multi_turn_public_then_submit()
    print("\nAll tests passed!")
