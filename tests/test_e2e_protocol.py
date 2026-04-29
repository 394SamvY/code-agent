"""
Lightweight OJ-like e2e protocol tests.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import CodeProblem, OJTestCase
from src.data.verl_dataset import prepare_verl_datasets, problem_to_verl_record
from src.env.code_env import CodeEnvironment


SOLUTION = "import sys\nx = int(sys.stdin.read())\nprint(x + 1)"


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


def test_problem_to_verl_record_contains_create_kwargs():
    problem = _problem()

    record = problem_to_verl_record(problem)
    tools_kwargs = record["extra_info"]["tools_kwargs"]
    public_create_kwargs = tools_kwargs["run_public_tests"]["create_kwargs"]
    submit_create_kwargs = tools_kwargs["submit_solution"]["create_kwargs"]

    assert record["data_source"] == "codecontests"
    assert record["ability"] == "code"
    assert public_create_kwargs["max_submissions"] == 5
    assert public_create_kwargs["max_public_test_calls"] == 15
    assert public_create_kwargs["time_limit_seconds"] == 2.0
    assert public_create_kwargs["public_tests"] == [{"input": "1\n", "output": "2\n"}]
    assert "private_tests" not in public_create_kwargs
    assert submit_create_kwargs["max_submissions"] == 5
    assert submit_create_kwargs["time_limit_seconds"] == 2.0
    assert submit_create_kwargs["private_tests"] == [{"input": "41\n", "output": "42\n"}]
    assert "public_tests" not in submit_create_kwargs

    print("[PASS] test_problem_to_verl_record_contains_create_kwargs")


def test_prepare_verl_datasets_writes_explicit_source_split_files():
    codecontests_calls = []
    livecodebench_calls = []
    written_files = []

    def _fake_codecontests_loader(**kwargs):
        codecontests_calls.append(kwargs)
        return [_problem()]

    def _fake_livecodebench_loader(**kwargs):
        livecodebench_calls.append(kwargs)
        return [_problem()]

    def _fake_writer(problems, output_path, max_submissions=5, max_public_test_calls=15):
        written_files.append(
            (Path(output_path).name, max_submissions, max_public_test_calls, len(problems))
        )
        return Path(output_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.data.verl_dataset.load_codecontests", side_effect=_fake_codecontests_loader), \
             patch("src.data.verl_dataset.load_livecodebench", side_effect=_fake_livecodebench_loader), \
             patch("src.data.verl_dataset.problems_to_verl_parquet", side_effect=_fake_writer):
            paths = prepare_verl_datasets(
                output_dir=tmpdir,
                max_train_samples=2,
                max_valid_samples=3,
                max_codecontests_test_samples=4,
                max_livecodebench_test_samples=5,
                max_submissions=7,
            )

    assert [call["split"] for call in codecontests_calls] == ["train", "valid", "test"]
    assert [call["max_samples"] for call in codecontests_calls] == [2, 3, 4]
    assert [call["split"] for call in livecodebench_calls] == ["test"]
    assert [call["max_samples"] for call in livecodebench_calls] == [5]
    assert set(paths) == {
        "codecontests_train",
        "codecontests_valid",
        "codecontests_test",
        "livecodebench_test",
    }
    assert [name for name, _, _, _ in written_files] == [
        "codecontests_train.parquet",
        "codecontests_valid.parquet",
        "codecontests_test.parquet",
        "livecodebench_test.parquet",
    ]
    assert all(max_submissions == 7 for _, max_submissions, _, _ in written_files)
    assert all(max_public_test_calls == 15 for _, _, max_public_test_calls, _ in written_files)

    print("[PASS] test_prepare_verl_datasets_writes_explicit_source_split_files")


def test_code_environment_submit_uses_private_tests():
    env = CodeEnvironment(_problem())

    result = env.submit_solution(SOLUTION)

    assert result["verdict"] == "accepted"
    assert result["passed"] == 1
    assert env.is_accepted is True

    print("[PASS] test_code_environment_submit_uses_private_tests")


def test_code_environment_public_then_submit_tool_flow():
    env = CodeEnvironment(_problem())

    public_observation = env.execute_tool("run_public_tests", code=SOLUTION)
    submit_observation = env.execute_tool("submit_solution", code=SOLUTION)

    assert "run_public_tests: accepted" in public_observation
    assert "submit_solution: accepted" in submit_observation
    assert env.tool_history == ["run_public_tests", "submit_solution"]
    assert len(env.public_results_history) == 1
    assert len(env.submission_history) == 1

    print("[PASS] test_code_environment_public_then_submit_tool_flow")


if __name__ == "__main__":
    test_problem_to_verl_record_contains_create_kwargs()
    test_prepare_verl_datasets_writes_explicit_source_split_files()
    test_code_environment_submit_uses_private_tests()
    test_code_environment_public_then_submit_tool_flow()
    print("\nAll tests passed!")
