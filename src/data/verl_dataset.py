"""
verl 数据格式转换
==================

将 CodeProblem 列表转换为 verl RLHFDataset 所需的 parquet 格式。

verl 的数据集要求每条记录包含:
- prompt: chat messages 列表 (JSON 序列化字符串)
- 其余字段作为 extra columns, 会通过 non_tensor_batch 传给 tool.create()
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.prompts import SYSTEM_PROMPT_AGENTIC_PLAIN, USER_PROMPT_TEMPLATE
from src.data.dataset import CodeProblem, load_mbpp, load_humaneval, load_apps


def problem_to_verl_record(problem: CodeProblem) -> dict:
    """将单个 CodeProblem 转换为 verl 训练记录。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_AGENTIC_PLAIN},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            problem_description=problem.prompt
        )},
    ]
    return {
        "prompt": json.dumps(messages, ensure_ascii=False),
        "task_id": problem.task_id,
        "test_list": json.dumps(problem.test_list, ensure_ascii=False),
        "entry_point": problem.entry_point,
    }


def problems_to_verl_parquet(
    problems: list[CodeProblem],
    output_path: str | Path,
) -> Path:
    """将 CodeProblem 列表转换为 verl 格式的 parquet 文件。"""
    records = [problem_to_verl_record(p) for p in problems]
    df = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(records)} records to {output_path}")
    return output_path


def prepare_verl_datasets(
    output_dir: str = "./data/verl",
    use_apps: bool = False,
    apps_difficulty: str = "introductory",
    apps_max_samples: int = 3000,
) -> dict[str, Path]:
    """准备所有 verl 训练和验证数据集。

    Returns:
        字典: {split_name: parquet_path}
    """
    output_dir = Path(output_dir)
    paths: dict[str, Path] = {}

    print("Loading MBPP train...")
    train_problems = load_mbpp(version="full", split="train")
    print(f"  MBPP train: {len(train_problems)} problems")

    if use_apps:
        print("Loading APPS...")
        apps_problems = load_apps(
            split="train",
            difficulty=apps_difficulty,
            max_samples=apps_max_samples,
        )
        train_problems.extend(apps_problems)
        print(f"  + APPS: {len(apps_problems)}, total: {len(train_problems)}")

    paths["train"] = problems_to_verl_parquet(
        train_problems, output_dir / "train.parquet"
    )

    print("Loading MBPP validation...")
    val_problems = load_mbpp(version="full", split="validation")
    paths["val"] = problems_to_verl_parquet(
        val_problems, output_dir / "val.parquet"
    )

    print("Loading MBPP test...")
    test_problems = load_mbpp(version="full", split="test")
    paths["test"] = problems_to_verl_parquet(
        test_problems, output_dir / "test.parquet"
    )

    print("Loading HumanEval...")
    he_problems = load_humaneval()
    paths["humaneval"] = problems_to_verl_parquet(
        he_problems, output_dir / "humaneval.parquet"
    )

    return paths


if __name__ == "__main__":
    paths = prepare_verl_datasets()
    print("\nAll datasets prepared:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
