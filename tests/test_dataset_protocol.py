"""
数据 schema 本地验证
===================

验证新的 OJ-like v1 数据协议：
- 结构化测试用例 OJTestCase
- CodeProblem 默认字段
- LiveCodeBench 公开/隐藏测试解析
- CodeContests 过滤策略与 generated_tests 合并行为
- 统一 loader registry
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.data.dataset as dataset_mod
from src.data.dataset import CodeProblem, OJTestCase, DATASET_LOADERS


def _fake_load_dataset(*args, **kwargs):
    name = args[0]
    split = kwargs.get("split")

    if name == "deepmind/code_contests":
        assert split in {"train", "valid", "test"}
        return Dataset.from_dict(
            {
                "name": [
                    "Keep me",
                    "No Python solution",
                    "File IO problem",
                ],
                "description": [
                    "Solve contest problem 1",
                    "Solve contest problem 2",
                    "Solve contest problem 3",
                ],
                "public_tests": [
                    {"input": ["1\n"], "output": ["2\n"]},
                    {"input": ["10\n"], "output": ["11\n"]},
                    {"input": ["100\n"], "output": ["101\n"]},
                ],
                "private_tests": [
                    {"input": ["3\n"], "output": ["4\n"]},
                    {"input": ["12\n"], "output": ["13\n"]},
                    {"input": ["102\n"], "output": ["103\n"]},
                ],
                "generated_tests": [
                    {"input": ["5\n"], "output": ["6\n"]},
                    {"input": ["14\n"], "output": ["15\n"]},
                    {"input": ["104\n"], "output": ["105\n"]},
                ],
                "solutions": [
                    {
                        "language": [3, 2, 1],
                        "solution": [
                            "print('py3 solution')",
                            "// cpp",
                            "print 'py2 solution'",
                        ],
                    },
                    {
                        "language": [2, 4],
                        "solution": ["// cpp only", "// java only"],
                    },
                    {
                        "language": [3],
                        "solution": ["print('would be filtered by file io')"],
                    },
                ],
                "difficulty": [2, 1, 3],
                "source": [2, 6, 1],
                "time_limit": [
                    {"seconds": 1, "nanos": 500_000_000},
                    {"seconds": 2, "nanos": 0},
                    {"seconds": 3, "nanos": 0},
                ],
                "memory_limit_bytes": [256_000_000, 128_000_000, 64_000_000],
                "input_file": ["", "", "input.txt"],
                "output_file": ["", "", "output.txt"],
                "cf_contest_id": [1001, 1002, 1003],
                "cf_index": ["A", "B", "C"],
                "cf_points": [500.0, 750.0, 1000.0],
                "cf_rating": [1200, 1400, 1600],
                "cf_tags": [["math"], ["graphs"], ["dp"]],
                "is_description_translated": [False, False, False],
                "untranslated_description": ["", "", ""],
            }
        )

    if name == "livecodebench/code_generation_lite":
        assert split == "test"
        assert kwargs.get("version_tag") == "release_v6"
        return Dataset.from_dict(
            {
                "question_title": ["A. Short Sort"],
                "question_content": ["Sort three characters."],
                "platform": ["codeforces"],
                "question_id": ["1873_A"],
                "contest_id": ["1873"],
                "contest_date": ["2023-08-21T00:00:00"],
                "starter_code": [""],
                "difficulty": ["easy"],
                "public_test_cases": [
                    json.dumps(
                        [{"input": "6\nabc\n", "output": "YES\n", "testtype": "stdin"}]
                    )
                ],
                "private_test_cases": [
                    json.dumps(
                        [{"input": "1\ncba\n", "output": "YES\n", "testtype": "stdin"}]
                    )
                ],
                "metadata": ['{"source_release":"release_v6"}'],
            }
        )

    raise AssertionError(f"Unexpected dataset request: args={args}, kwargs={kwargs}")


def test_codeproblem_defaults():
    problem = CodeProblem(
        task_id="codecontests/example",
        dataset="codecontests",
        problem_statement="Solve x.",
    )

    assert problem.title is None
    assert problem.starter_code == ""
    assert problem.public_tests == []
    assert problem.private_tests == []
    assert problem.reference_solutions == []
    assert problem.metadata == {}

    print("[PASS] test_codeproblem_defaults")


def test_load_livecodebench_parses_structured_tests():
    with patch.object(dataset_mod, "load_dataset", side_effect=_fake_load_dataset):
        problems = dataset_mod.load_livecodebench(version_tag="release_v6")

    assert len(problems) == 1
    problem = problems[0]
    assert problem.dataset == "livecodebench"
    assert problem.task_id == "livecodebench/1873_A"
    assert problem.title == "A. Short Sort"
    assert problem.problem_statement == "Sort three characters."
    assert problem.public_tests == [OJTestCase(input="6\nabc\n", output="YES\n")]
    assert problem.private_tests == [OJTestCase(input="1\ncba\n", output="YES\n")]
    assert problem.metadata["contest_id"] == "1873"
    assert problem.metadata["version_tag"] == "release_v6"
    assert problem.metadata["original_split"] == "test"
    assert problem.metadata["source"] == "codeforces"
    assert not hasattr(problem, "memory_limit_bytes")

    print("[PASS] test_load_livecodebench_parses_structured_tests")


def test_load_codecontests_filters_and_merges_generated_tests():
    with patch.object(dataset_mod, "load_dataset", side_effect=_fake_load_dataset):
        problems = dataset_mod.load_codecontests(split="valid")

    assert len(problems) == 1
    problem = problems[0]
    assert problem.dataset == "codecontests"
    assert problem.title == "Keep me"
    assert problem.problem_statement == "Solve contest problem 1"
    assert problem.public_tests == [OJTestCase(input="1\n", output="2\n")]
    assert problem.private_tests == [
        OJTestCase(input="3\n", output="4\n"),
        OJTestCase(input="5\n", output="6\n"),
    ]
    assert problem.reference_solutions == [
        "print('py3 solution')",
        "print 'py2 solution'",
    ]
    assert problem.time_limit_seconds == 1.5
    assert problem.difficulty == "MEDIUM"
    assert problem.metadata["source"] == "CODEFORCES"
    assert problem.metadata["original_split"] == "valid"
    assert problem.metadata["memory_limit_bytes"] == 256_000_000
    assert problem.metadata["num_generated_tests"] == 1

    print("[PASS] test_load_codecontests_filters_and_merges_generated_tests")


def test_load_codecontests_uses_valid_split():
    seen_splits: list[str] = []

    def _capturing_loader(*args, **kwargs):
        seen_splits.append(kwargs["split"])
        return _fake_load_dataset(*args, **kwargs)

    with patch.object(dataset_mod, "load_dataset", side_effect=_capturing_loader):
        dataset_mod.load_codecontests(split="valid", max_samples=1)

    assert seen_splits == ["valid"]

    print("[PASS] test_load_codecontests_uses_valid_split")


def test_dataset_loader_registry():
    assert set(DATASET_LOADERS) == {"codecontests", "livecodebench"}
    assert DATASET_LOADERS["codecontests"] is dataset_mod.load_codecontests
    assert DATASET_LOADERS["livecodebench"] is dataset_mod.load_livecodebench

    print("[PASS] test_dataset_loader_registry")


if __name__ == "__main__":
    test_codeproblem_defaults()
    test_load_livecodebench_parses_structured_tests()
    test_load_codecontests_filters_and_merges_generated_tests()
    test_load_codecontests_uses_valid_split()
    test_dataset_loader_registry()
    print("\nAll tests passed!")
