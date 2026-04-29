"""
verl parquet export for the OJ-like v1 protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.dataset import CodeProblem, load_codecontests, load_livecodebench
from src.env.tools import DEFAULT_MAX_PUBLIC_TEST_CALLS, serialize_oj_tests
from src.prompts import build_agentic_messages


def _dataset_local_path(data_dir: str | None, name: str) -> str | None:
    if data_dir is None:
        return None
    candidate = Path(data_dir) / name
    return str(candidate) if candidate.exists() else None


def _create_kwargs(
    problem: CodeProblem,
    max_submissions: int,
    max_public_test_calls: int = DEFAULT_MAX_PUBLIC_TEST_CALLS,
) -> dict[str, Any]:
    return {
        "public_tests": serialize_oj_tests(problem.public_tests),
        "private_tests": serialize_oj_tests(problem.private_tests),
        "time_limit_seconds": problem.time_limit_seconds,
        "max_submissions": max_submissions,
        "max_public_test_calls": max_public_test_calls,
    }


def _tool_create_kwargs(
    problem: CodeProblem,
    max_submissions: int,
    max_public_test_calls: int = DEFAULT_MAX_PUBLIC_TEST_CALLS,
) -> dict[str, dict[str, Any]]:
    common = {
        "time_limit_seconds": problem.time_limit_seconds,
        "max_submissions": max_submissions,
    }
    return {
        "run_public_tests": {
            **common,
            "public_tests": serialize_oj_tests(problem.public_tests),
            "max_public_test_calls": max_public_test_calls,
        },
        "submit_solution": {
            **common,
            "private_tests": serialize_oj_tests(problem.private_tests),
        },
    }


def problem_to_verl_record(
    problem: CodeProblem,
    max_submissions: int = 5,
    max_public_test_calls: int = DEFAULT_MAX_PUBLIC_TEST_CALLS,
) -> dict[str, Any]:
    """Convert a CodeProblem into a verl multi-turn training record."""
    tool_create_kwargs = _tool_create_kwargs(
        problem,
        max_submissions=max_submissions,
        max_public_test_calls=max_public_test_calls,
    )
    # `extra_info` 不是给模型看的 prompt；verl 会把它传给 tool layer，
    # 让同一套工具在不同题目上使用不同的测试、时间限制和提交次数限制。
    extra_info = {
        "task_id": problem.task_id,
        "dataset": problem.dataset,
        "tools_kwargs": {
            name: {"create_kwargs": kwargs}
            for name, kwargs in tool_create_kwargs.items()
        },
    }
    return {
        "data_source": problem.dataset,
        "prompt": build_agentic_messages(problem),
        "ability": "code",
        "reward_model": {
            "style": "rule",
            # OJ 任务的正确性由测试执行产生的 tool reward 决定。
            # 这里保留 task_id，只作为 verl 的 ground-truth handle。
            "ground_truth": {
                "task_id": problem.task_id,
            },
        },
        "extra_info": extra_info,
    }


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _serialize_record_for_parquet(record: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(record)
    serialized["prompt"] = _json_dumps(record["prompt"])
    serialized["reward_model"] = _json_dumps(record["reward_model"])
    serialized["extra_info"] = _json_dumps(record["extra_info"])
    return serialized


def problems_to_verl_parquet(
    problems: list[CodeProblem],
    output_path: str | Path,
    max_submissions: int = 5,
    max_public_test_calls: int = DEFAULT_MAX_PUBLIC_TEST_CALLS,
    batch_size: int = 128,
) -> Path:
    """Write CodeProblem records to a verl-compatible parquet file."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Install pyarrow to export verl parquet files."
        ) from e

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_name(output.name + ".tmp")
    writer = None
    try:
        for start in range(0, len(problems), batch_size):
            batch = [
                _serialize_record_for_parquet(
                    problem_to_verl_record(
                        problem,
                        max_submissions=max_submissions,
                        max_public_test_calls=max_public_test_calls,
                    )
                )
                for problem in problems[start : start + batch_size]
            ]
            if not batch:
                continue
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(temp_output, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    temp_output.replace(output)
    return output


def prepare_verl_datasets(
    output_dir: str = "./data/verl",
    data_dir: str | None = None,
    max_train_samples: int | None = None,
    max_valid_samples: int | None = None,
    max_codecontests_test_samples: int | None = None,
    max_livecodebench_test_samples: int | None = None,
    max_submissions: int = 5,
    max_public_test_calls: int = DEFAULT_MAX_PUBLIC_TEST_CALLS,
    livecodebench_version_tag: str = "release_v6",
) -> dict[str, Path]:
    """Prepare explicit CodeContests and LiveCodeBench parquet files."""
    output = Path(output_dir)
    codecontests_path = _dataset_local_path(data_dir, "codecontests")
    livecodebench_path = _dataset_local_path(data_dir, "livecodebench")

    codecontests_train = load_codecontests(
        split="train",
        max_samples=max_train_samples,
        local_path=codecontests_path,
    )
    codecontests_valid = load_codecontests(
        split="valid",
        max_samples=max_valid_samples,
        local_path=codecontests_path,
    )
    codecontests_test = load_codecontests(
        split="test",
        max_samples=max_codecontests_test_samples,
        local_path=codecontests_path,
    )
    livecodebench_test = load_livecodebench(
        split="test",
        max_samples=max_livecodebench_test_samples,
        local_path=livecodebench_path,
        version_tag=livecodebench_version_tag,
    )

    paths = {
        "codecontests_train": problems_to_verl_parquet(
            codecontests_train,
            output / "codecontests_train.parquet",
            max_submissions=max_submissions,
            max_public_test_calls=max_public_test_calls,
        ),
        "codecontests_valid": problems_to_verl_parquet(
            codecontests_valid,
            output / "codecontests_valid.parquet",
            max_submissions=max_submissions,
            max_public_test_calls=max_public_test_calls,
        ),
        "codecontests_test": problems_to_verl_parquet(
            codecontests_test,
            output / "codecontests_test.parquet",
            max_submissions=max_submissions,
            max_public_test_calls=max_public_test_calls,
        ),
        "livecodebench_test": problems_to_verl_parquet(
            livecodebench_test,
            output / "livecodebench_test.parquet",
            max_submissions=max_submissions,
            max_public_test_calls=max_public_test_calls,
        ),
    }
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OJ-like verl parquet datasets")
    parser.add_argument("--output_dir", default="./data/verl")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_valid_samples", type=int, default=None)
    parser.add_argument("--max_codecontests_test_samples", type=int, default=None)
    parser.add_argument("--max_livecodebench_test_samples", type=int, default=None)
    parser.add_argument("--max_submissions", type=int, default=5)
    parser.add_argument(
        "--max_public_test_calls",
        type=int,
        default=DEFAULT_MAX_PUBLIC_TEST_CALLS,
    )
    parser.add_argument("--livecodebench_version_tag", default="release_v6")
    args = parser.parse_args()

    paths = prepare_verl_datasets(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
        max_codecontests_test_samples=args.max_codecontests_test_samples,
        max_livecodebench_test_samples=args.max_livecodebench_test_samples,
        max_submissions=args.max_submissions,
        max_public_test_calls=args.max_public_test_calls,
        livecodebench_version_tag=args.livecodebench_version_tag,
    )
    for split, path in paths.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
