"""
verl parquet export for the OJ-like v1 protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.data.dataset import CodeProblem, load_codecontests, load_livecodebench
from src.env.tools import serialize_oj_tests
from src.prompts import build_agentic_messages


def _dataset_local_path(data_dir: str | None, name: str) -> str | None:
    if data_dir is None:
        return None
    candidate = Path(data_dir) / name
    return str(candidate) if candidate.exists() else None


def _create_kwargs(problem: CodeProblem, max_submissions: int) -> dict[str, Any]:
    return {
        "public_tests": serialize_oj_tests(problem.public_tests),
        "private_tests": serialize_oj_tests(problem.private_tests),
        "time_limit_seconds": problem.time_limit_seconds,
        "max_submissions": max_submissions,
    }


def problem_to_verl_record(
    problem: CodeProblem,
    max_submissions: int = 5,
) -> dict[str, Any]:
    """Convert a CodeProblem into a verl multi-turn training record."""
    create_kwargs = _create_kwargs(problem, max_submissions=max_submissions)
    extra_info = {
        "task_id": problem.task_id,
        "dataset": problem.dataset,
        "title": problem.title,
        "difficulty": problem.difficulty,
        "metadata": problem.metadata,
        "public_tests": create_kwargs["public_tests"],
        "private_tests": create_kwargs["private_tests"],
        "time_limit_seconds": problem.time_limit_seconds,
        "max_submissions": max_submissions,
        "create_kwargs": create_kwargs,
        "tools_kwargs": {
            "run_public_tests": {"create_kwargs": create_kwargs},
            "submit_solution": {"create_kwargs": create_kwargs},
        },
    }
    return {
        "data_source": problem.dataset,
        "prompt": build_agentic_messages(problem),
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "task_id": problem.task_id,
                "private_tests": create_kwargs["private_tests"],
            },
        },
        "extra_info": extra_info,
    }


def problems_to_verl_parquet(
    problems: list[CodeProblem],
    output_path: str | Path,
    max_submissions: int = 5,
) -> Path:
    """Write CodeProblem records to a verl-compatible parquet file."""
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Install pandas and pyarrow to export verl parquet files."
        ) from e

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [
        problem_to_verl_record(problem, max_submissions=max_submissions)
        for problem in problems
    ]
    pd.DataFrame(records).to_parquet(output, index=False)
    return output


def prepare_verl_datasets(
    output_dir: str = "./data/verl",
    data_dir: str | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    max_submissions: int = 5,
    livecodebench_version_tag: str = "release_v6",
) -> dict[str, Path]:
    """Prepare CodeContests train/val and LiveCodeBench test parquet files."""
    output = Path(output_dir)
    codecontests_path = _dataset_local_path(data_dir, "codecontests")
    livecodebench_path = _dataset_local_path(data_dir, "livecodebench")

    train = load_codecontests(
        split="train",
        max_samples=max_train_samples,
        local_path=codecontests_path,
    )
    val = load_codecontests(
        split="validation",
        max_samples=max_val_samples,
        local_path=codecontests_path,
    )
    test = load_livecodebench(
        split="test",
        max_samples=max_test_samples,
        local_path=livecodebench_path,
        version_tag=livecodebench_version_tag,
    )

    paths = {
        "train": problems_to_verl_parquet(
            train,
            output / "train.parquet",
            max_submissions=max_submissions,
        ),
        "val": problems_to_verl_parquet(
            val,
            output / "val.parquet",
            max_submissions=max_submissions,
        ),
        "test": problems_to_verl_parquet(
            test,
            output / "test.parquet",
            max_submissions=max_submissions,
        ),
    }
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OJ-like verl parquet datasets")
    parser.add_argument("--output_dir", default="./data/verl")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--max_submissions", type=int, default=5)
    parser.add_argument("--livecodebench_version_tag", default="release_v6")
    args = parser.parse_args()

    paths = prepare_verl_datasets(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        max_submissions=args.max_submissions,
        livecodebench_version_tag=args.livecodebench_version_tag,
    )
    for split, path in paths.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
