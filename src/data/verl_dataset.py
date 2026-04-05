"""
verl 数据格式转换
==================

将 CodeProblem 列表转换为 verl RLHFDataset 所需的 parquet 格式。

verl 的数据集要求每条记录包含:
- prompt: chat messages 列表（直接存 list，不是 JSON 字符串）
- agent_name: "tool_agent"（verl 用它路由到 ToolAgentLoop）
- data_source: 数据来源标识
- extra_info: 包含 tools_kwargs，verl 通过它把参数传给 tool.create()

数据流:
  parquet extra_info.tools_kwargs.execute_code.create_kwargs
    → verl _call_tool()
    → tool.create(create_kwargs={"test_list": [...], "entry_point": "..."})
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.prompts import SYSTEM_PROMPT_AGENTIC_PLAIN, USER_PROMPT_TEMPLATE
from src.data.dataset import CodeProblem, load_mbpp, load_humaneval, load_apps


def problem_to_verl_record(problem: CodeProblem) -> dict:
    """将单个 CodeProblem 转换为 verl 训练记录。

    对齐 verl 官方 GSM8K tool agent 示例的数据格式：
    - prompt: list[dict]（不是 JSON 字符串）
    - agent_name: "tool_agent"
    - extra_info.tools_kwargs: 按工具名组织的 create_kwargs
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_AGENTIC_PLAIN},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            problem_description=problem.prompt
        )},
    ]

    return {
        "data_source": f"code_agent/{problem.task_id.split('/')[0]}",
        "agent_name": "tool_agent",
        "prompt": messages,
        "reward_model": {
            "ground_truth": problem.test_list,
        },
        "extra_info": {
            "task_id": problem.task_id,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "execute_code": {
                    "create_kwargs": {
                        "test_list": problem.test_list,
                        "entry_point": problem.entry_point,
                    },
                },
            },
        },
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
    data_dir: str | None = None,
    use_apps: bool = True,
    apps_difficulty: str = "introductory",
    apps_max_samples: int = 3000,
) -> dict[str, Path]:
    """准备所有 verl 训练和验证数据集。

    Args:
        output_dir: parquet 输出目录
        data_dir: 本地数据集根目录（含 mbpp_full/ humaneval/ 子目录），
                  无外网环境使用
        use_apps: 是否加载 APPS 数据集（需要网络或本地缓存）
        apps_difficulty: APPS 难度过滤
        apps_max_samples: APPS 最大样本数

    Returns:
        字典: {split_name: parquet_path}
    """
    output_dir = Path(output_dir)
    paths: dict[str, Path] = {}

    mbpp_local = os.path.join(data_dir, "mbpp_full") if data_dir else None
    humaneval_local = os.path.join(data_dir, "humaneval") if data_dir else None

    print("Loading MBPP train...")
    train_problems = load_mbpp(version="full", split="train", local_path=mbpp_local)
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
    val_problems = load_mbpp(version="full", split="validation", local_path=mbpp_local)
    paths["val"] = problems_to_verl_parquet(
        val_problems, output_dir / "val.parquet"
    )

    print("Loading MBPP test...")
    test_problems = load_mbpp(version="full", split="test", local_path=mbpp_local)
    paths["test"] = problems_to_verl_parquet(
        test_problems, output_dir / "test.parquet"
    )

    print("Loading HumanEval...")
    he_problems = load_humaneval(local_path=humaneval_local)
    paths["humaneval"] = problems_to_verl_parquet(
        he_problems, output_dir / "humaneval.parquet"
    )

    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None,
                        help="本地数据集根目录（含 mbpp_full/ humaneval/ 子目录）")
    parser.add_argument("--output_dir", type=str, default="./data/verl")
    parser.add_argument("--no_apps", action="store_true",
                        help="跳过 APPS 数据集（无网络或无本地缓存时使用）")
    args = parser.parse_args()

    paths = prepare_verl_datasets(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        use_apps=not args.no_apps,
    )
    print("\nAll datasets prepared:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
