"""
数据集加载
==========

加载 MBPP / HumanEval / APPS 数据集并转换为统一格式。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


@dataclass
class CodeProblem:
    """统一的编程题格式."""

    task_id: str
    prompt: str            # 题目描述
    test_list: list[str]   # 测试用例（assert 语句列表）
    entry_point: str       # 函数入口名（HumanEval 有，MBPP 可能为空）
    canonical_solution: str = ""  # 标准答案（仅评估时参考）


# ---------------------------------------------------------------------------
# MBPP
# ---------------------------------------------------------------------------

def load_mbpp(
    version: str = "full",
    split: str = "train",
    max_samples: int | None = None,
    local_path: str | None = None,
) -> list[CodeProblem]:
    """加载 MBPP 数据集.

    Args:
        version: "full" 或 "sanitized"
        split: "train" / "validation" / "test"
        max_samples: 限制样本数
        local_path: 本地缓存路径（无外网环境使用）

    Returns:
        CodeProblem 列表
    """
    if local_path and os.path.exists(local_path):
        ds_all = load_from_disk(local_path)
        ds = ds_all[split] if isinstance(ds_all, DatasetDict) else ds_all
    else:
        ds = load_dataset("google-research-datasets/mbpp", version, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    problems: list[CodeProblem] = []
    for row in ds:
        desc = row.get("text") or row.get("prompt", "")
        tests = row.get("test_list", [])
        # 将第一个测试用例加入 prompt，让模型知道函数名和参数格式
        # 参考 bigcode-evaluation-harness 的做法
        if tests:
            desc = f"{desc}\n{tests[0]}"
        problems.append(CodeProblem(
            task_id=f"mbpp/{row['task_id']}",
            prompt=desc,
            test_list=tests,
            entry_point="",
            canonical_solution=row.get("code", ""),
        ))
    return problems


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------

def load_humaneval(
    max_samples: int | None = None,
    local_path: str | None = None,
) -> list[CodeProblem]:
    """加载 HumanEval 数据集.

    Args:
        max_samples: 限制样本数
        local_path: 本地缓存路径（无外网环境使用）

    Returns:
        CodeProblem 列表
    """
    if local_path and os.path.exists(local_path):
        ds_all = load_from_disk(local_path)
        ds = ds_all["test"] if isinstance(ds_all, DatasetDict) else ds_all
    else:
        ds = load_dataset("openai/openai_humaneval", split="test")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    problems: list[CodeProblem] = []
    for row in ds:
        test_code = row["test"]
        entry = row["entry_point"]
        test_calls = f"{test_code}\ncheck({entry})"
        problems.append(CodeProblem(
            task_id=row["task_id"],
            prompt=row["prompt"],
            test_list=[test_calls],
            entry_point=entry,
            canonical_solution=row.get("canonical_solution", ""),
        ))
    return problems


# ---------------------------------------------------------------------------
# APPS
# ---------------------------------------------------------------------------

def load_apps(
    split: str = "train",
    difficulty: str | None = None,
    max_samples: int | None = None,
    local_path: str | None = None,
) -> list[CodeProblem]:
    """加载 APPS 数据集（~10000 题，含测试用例）.

    APPS 数据集比 MBPP 大 25+ 倍，用于缓解 RL 训练数据不足的问题。

    Args:
        split: "train" / "test"
        difficulty: 可选过滤难度 "introductory" / "interview" / "competition"
        max_samples: 限制样本数
        local_path: 本地 JSONL 文件路径（如 /data/apps/train.jsonl）

    Returns:
        CodeProblem 列表
    """
    # codeparrot/apps 的 loading script 已废弃，直接加载 JSONL 文件
    if local_path and os.path.exists(local_path):
        ds = load_dataset("json", data_files=local_path, split="train")
    else:
        ds = load_dataset(
            "json",
            data_files=f"hf://datasets/codeparrot/apps/{split}.jsonl",
            split="train",
        )

    if difficulty is not None:
        # JSONL 中 difficulty 字段值为字符串: "introductory" / "interview" / "competition"
        ds = ds.filter(lambda x: str(x.get("difficulty", "")) == difficulty)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    problems: list[CodeProblem] = []
    for idx, row in enumerate(ds):
        question = row.get("question", "")
        if not question.strip():
            continue

        # APPS 的测试用例存在 input_output 字段中（JSON 格式）
        test_list = _parse_apps_tests(row, idx)
        if not test_list:
            continue

        solutions = row.get("solutions", "")
        canonical = ""
        if solutions:
            try:
                import json as _json
                sol_list = _json.loads(solutions)
                if sol_list:
                    canonical = sol_list[0]
            except (TypeError, _json.JSONDecodeError):
                pass

        problems.append(CodeProblem(
            task_id=f"apps/{row.get('problem_id', idx)}",
            prompt=question,
            test_list=test_list,
            entry_point="",
            canonical_solution=canonical,
        ))
    return problems


def _parse_apps_tests(row: dict, idx: int) -> list[str]:
    """从 APPS 的 input_output 字段解析出可执行的 assert 测试.

    APPS 测试格式有两种：
    1. fn_name 存在：函数调用式，可转为 assert 语句
    2. fn_name 不存在：stdin/stdout 式，转为 IO 测试代码
    """
    import json as _json

    io_str = row.get("input_output", "")
    if not io_str:
        return []

    try:
        io_data = _json.loads(io_str)
    except (TypeError, _json.JSONDecodeError):
        return []

    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])
    fn_name = io_data.get("fn_name")

    if not inputs or not outputs or len(inputs) != len(outputs):
        return []

    test_list: list[str] = []

    if fn_name:
        # 函数调用式测试：assert fn_name(*args) == expected
        for inp, out in zip(inputs, outputs):
            if isinstance(inp, list):
                args_str = ", ".join(repr(a) for a in inp)
            else:
                args_str = repr(inp)
            test_list.append(f"assert {fn_name}({args_str}) == {repr(out)}")
    else:
        # stdin/stdout 式测试：构造 IO 测试代码
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            inp_str = inp if isinstance(inp, str) else str(inp)
            out_str = out if isinstance(out, str) else str(out)
            # 生成可执行的 IO 测试
            test_code = (
                f"import sys, io\n"
                f"sys.stdin = io.StringIO({repr(inp_str)})\n"
                f"_captured = io.StringIO()\n"
                f"sys.stdout = _captured\n"
                f"exec(open('<solution>').read()) if False else None\n"
                f"# IO test {i}: expected output check\n"
                f"assert _captured.getvalue().strip() == {repr(out_str.strip())}"
            )
            test_list.append(test_code)

    return test_list


# ---------------------------------------------------------------------------
# 转为 HuggingFace Dataset（TRL GRPOTrainer 需要）
# ---------------------------------------------------------------------------

def problems_to_hf_dataset(
    problems: list[CodeProblem],
    prompt_formatter=None,
) -> Dataset:
    """将 CodeProblem 列表转为 HuggingFace Dataset.

    Args:
        problems: 编程题列表
        prompt_formatter: 可选，将 problem.prompt 转换为训练用 prompt 字符串。
            默认直接使用 problem.prompt。

    Returns:
        HuggingFace Dataset，包含 "prompt" 和 "test_list" 列
    """
    records = []
    for p in problems:
        formatted = prompt_formatter(p.prompt) if prompt_formatter else p.prompt
        records.append({
            "prompt": formatted,
            "task_id": p.task_id,
            "test_list": p.test_list,
            "entry_point": p.entry_point,
        })
    return Dataset.from_list(records)
