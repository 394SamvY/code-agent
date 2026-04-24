"""
OJ-like 数据 schema 与 loader
============================

Phase 1 只为两套主数据集建立统一协议：

- CodeContests：训练 / 开发验证
- LiveCodeBench：最终测试

当前 v1 schema 明确采用结构化测试用例表示，并且只支持 stdin/stdout。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import zlib
from dataclasses import dataclass, field
from typing import Any, Literal

try:
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
except ModuleNotFoundError:
    Dataset = Any

    class DatasetDict(dict):
        pass

    def load_dataset(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError("Install the 'datasets' package to load datasets.")

    def load_from_disk(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError("Install the 'datasets' package to load datasets.")


DatasetName = Literal["codecontests", "livecodebench"]
DatasetSplit = Literal["train", "validation", "test"]

_CODECONTESTS_SOURCE_MAP = {
    0: "UNKNOWN_SOURCE",
    1: "CODECHEF",
    2: "CODEFORCES",
    3: "HACKEREARTH",
    4: "CODEJAM",
    6: "ATCODER",
    7: "AIZU",
}

_CODECONTESTS_DIFFICULTY_MAP = {
    0: "UNKNOWN_DIFFICULTY",
    1: "EASY",
    2: "MEDIUM",
    3: "HARD",
    4: "HARDER",
    5: "HARDEST",
    6: "EXTERNAL",
    7: "A",
    8: "B",
    9: "C",
    10: "D",
    11: "E",
    12: "F",
    13: "G",
    14: "H",
    15: "I",
    16: "J",
    17: "K",
    19: "L",
    20: "M",
    21: "N",
    22: "O",
    23: "P",
    24: "Q",
    25: "R",
    26: "S",
    27: "T",
    28: "U",
    29: "V",
}

_CODECONTESTS_LANGUAGE_MAP = {
    0: "UNKNOWN_LANGUAGE",
    1: "PYTHON",
    2: "CPP",
    3: "PYTHON3",
    4: "JAVA",
}


@dataclass(frozen=True)
class OJTestCase:
    """结构化 OJ 测试用例，仅支持 stdin/stdout."""

    input: str
    output: str


@dataclass
class CodeProblem:
    """统一后的 OJ 编程题 schema."""

    task_id: str
    dataset: DatasetName
    problem_statement: str
    title: str | None = None
    starter_code: str = ""
    public_tests: list[OJTestCase] = field(default_factory=list)
    private_tests: list[OJTestCase] = field(default_factory=list)
    time_limit_seconds: float | None = None
    reference_solutions: list[str] = field(default_factory=list)
    difficulty: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _select_split(
    dataset_name: str,
    split: str,
    local_path: str | None = None,
    **load_kwargs: Any,
) -> Dataset:
    """优先从本地读取，否则从 HF 加载指定 split."""
    if local_path and os.path.exists(local_path):
        ds_all = load_from_disk(local_path)
        if isinstance(ds_all, DatasetDict):
            if split in ds_all:
                return ds_all[split]
            if split == "valid" and "validation" in ds_all:
                return ds_all["validation"]
            if split == "validation" and "valid" in ds_all:
                return ds_all["valid"]
            raise KeyError(f"Split '{split}' not found in local dataset: {list(ds_all.keys())}")
        return ds_all

    return load_dataset(dataset_name, split=split, **load_kwargs)


def _normalize_codecontests_split(split: DatasetSplit) -> str:
    if split == "validation":
        return "valid"
    return split


def _stable_task_id(prefix: str, *parts: str) -> str:
    joined = "\n".join(parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}/{digest}"


def _normalize_label(value: Any, value_map: dict[int, str]) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value_map.get(value, str(value))
    if isinstance(value, str):
        return value
    return str(value)


def _parse_duration_seconds(raw: Any) -> float | None:
    if raw in (None, "", {}):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict):
        seconds = float(raw.get("seconds", 0) or 0)
        nanos = float(raw.get("nanos", 0) or 0)
        return seconds + nanos / 1_000_000_000
    raise TypeError(f"Unsupported time_limit payload: {type(raw)!r}")


def _parse_test_dict(raw: Any) -> list[OJTestCase]:
    """解析 CodeContests 风格的测试表示."""
    if raw in (None, "", []):
        return []

    if isinstance(raw, dict):
        inputs = raw.get("input", []) or []
        outputs = raw.get("output", []) or []
        if len(inputs) != len(outputs):
            raise ValueError("Mismatched CodeContests test lengths between input and output")
        return [OJTestCase(input=str(inp), output=str(out)) for inp, out in zip(inputs, outputs)]

    if isinstance(raw, list):
        cases: list[OJTestCase] = []
        for item in raw:
            if not isinstance(item, dict):
                raise TypeError(f"Unsupported CodeContests test case item: {type(item)!r}")
            cases.append(
                OJTestCase(
                    input=str(item.get("input", "")),
                    output=str(item.get("output", "")),
                )
            )
        return cases

    raise TypeError(f"Unsupported CodeContests tests payload: {type(raw)!r}")


def _decode_json_string(raw: str) -> Any:
    stripped = raw.strip()
    if not stripped:
        return []

    if stripped.startswith("[") or stripped.startswith("{"):
        return json.loads(stripped)

    # 某些 LCB 变体会把隐藏测试存为 base64(zlib(json)) 字符串。
    decoded = base64.b64decode(stripped)
    inflated = zlib.decompress(decoded).decode("utf-8")
    return json.loads(inflated)


def _parse_lcb_tests(raw: Any) -> list[OJTestCase]:
    """解析 LiveCodeBench 的 public/private test strings."""
    if raw in (None, "", []):
        return []

    if isinstance(raw, str):
        raw = _decode_json_string(raw)

    if isinstance(raw, dict):
        raw = [raw]

    if not isinstance(raw, list):
        raise TypeError(f"Unsupported LiveCodeBench tests payload: {type(raw)!r}")

    cases: list[OJTestCase] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError(f"Unsupported LiveCodeBench test case item: {type(item)!r}")
        testtype = item.get("testtype", "stdin")
        if testtype != "stdin":
            raise ValueError(f"Unsupported LiveCodeBench test type in v1 schema: {testtype}")
        cases.append(
            OJTestCase(
                input=str(item.get("input", "")),
                output=str(item.get("output", "")),
            )
        )
    return cases


def _iter_solutions(raw: Any) -> list[tuple[str | None, str]]:
    """提取 (language, solution) 序列，兼容 dict-of-lists / list-of-dicts."""
    if raw in (None, "", []):
        return []

    if isinstance(raw, dict):
        languages = raw.get("language", []) or []
        solutions = raw.get("solution", []) or []
        if len(languages) != len(solutions):
            raise ValueError("Mismatched solutions lengths between language and solution")
        pairs = zip(languages, solutions)
    elif isinstance(raw, list):
        pairs = ((item.get("language"), item.get("solution", "")) for item in raw)
    else:
        raise TypeError(f"Unsupported solutions payload: {type(raw)!r}")

    normalized: list[tuple[str | None, str]] = []
    for language, solution in pairs:
        if solution in (None, ""):
            continue
        if isinstance(language, int):
            label = _CODECONTESTS_LANGUAGE_MAP.get(language, str(language))
        elif isinstance(language, str):
            label = language.upper()
        else:
            label = str(language) if language is not None else None
        normalized.append((label, str(solution)))
    return normalized


def _extract_python_references(raw: Any) -> list[str]:
    solutions = _iter_solutions(raw)
    py3 = [solution for language, solution in solutions if language == "PYTHON3"]
    py2 = [solution for language, solution in solutions if language == "PYTHON"]
    return py3 + py2


def _is_stdio_problem(row: dict[str, Any]) -> bool:
    return not row.get("input_file") and not row.get("output_file")


def load_codecontests(
    split: DatasetSplit = "train",
    max_samples: int | None = None,
    local_path: str | None = None,
) -> list[CodeProblem]:
    """加载 CodeContests 并映射到统一 schema."""
    hf_split = _normalize_codecontests_split(split)
    ds = _select_split("deepmind/code_contests", split=hf_split, local_path=local_path)

    problems: list[CodeProblem] = []
    for row in ds:
        if not _is_stdio_problem(row):
            continue

        reference_solutions = _extract_python_references(row.get("solutions"))
        if not reference_solutions:
            continue

        public_tests = _parse_test_dict(row.get("public_tests"))
        private_tests = _parse_test_dict(row.get("private_tests"))
        generated_tests = _parse_test_dict(row.get("generated_tests"))

        title = row.get("name") or None
        description = row.get("description", "")
        task_id = _stable_task_id("codecontests", title or "", description)

        problems.append(
            CodeProblem(
                task_id=task_id,
                dataset="codecontests",
                title=title,
                problem_statement=description,
                starter_code="",
                public_tests=public_tests,
                private_tests=private_tests + generated_tests,
                time_limit_seconds=_parse_duration_seconds(row.get("time_limit")),
                reference_solutions=reference_solutions,
                difficulty=_normalize_label(row.get("difficulty"), _CODECONTESTS_DIFFICULTY_MAP),
                metadata={
                    "original_split": split,
                    "source": _normalize_label(row.get("source"), _CODECONTESTS_SOURCE_MAP),
                    "memory_limit_bytes": row.get("memory_limit_bytes"),
                    "cf_contest_id": row.get("cf_contest_id"),
                    "cf_index": row.get("cf_index"),
                    "cf_points": row.get("cf_points"),
                    "cf_rating": row.get("cf_rating"),
                    "cf_tags": row.get("cf_tags", []),
                    "is_description_translated": row.get("is_description_translated"),
                    "untranslated_description": row.get("untranslated_description"),
                    "num_generated_tests": len(generated_tests),
                },
            )
        )

        if max_samples is not None and len(problems) >= max_samples:
            break

    return problems


def load_livecodebench(
    split: DatasetSplit = "test",
    max_samples: int | None = None,
    local_path: str | None = None,
    version_tag: str = "release_v6",
) -> list[CodeProblem]:
    """加载 LiveCodeBench code generation lite 并映射到统一 schema."""
    if split != "test":
        raise ValueError("LiveCodeBench v1 loader currently only supports split='test'")

    ds = _select_split(
        "livecodebench/code_generation_lite",
        split="test",
        local_path=local_path,
        version_tag=version_tag,
    )

    problems: list[CodeProblem] = []
    for row in ds:
        task_id = f"livecodebench/{row['question_id']}"
        raw_metadata = row.get("metadata", "")
        try:
            parsed_metadata = _decode_json_string(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
        except Exception:
            parsed_metadata = {"raw_metadata": raw_metadata}

        problems.append(
            CodeProblem(
                task_id=task_id,
                dataset="livecodebench",
                title=row.get("question_title") or None,
                problem_statement=row.get("question_content", ""),
                starter_code=row.get("starter_code", "") or "",
                public_tests=_parse_lcb_tests(row.get("public_test_cases")),
                private_tests=_parse_lcb_tests(row.get("private_test_cases")),
                time_limit_seconds=None,
                reference_solutions=[],
                difficulty=row.get("difficulty"),
                metadata={
                    "original_split": "test",
                    "source": row.get("platform"),
                    "question_id": row.get("question_id"),
                    "contest_id": row.get("contest_id"),
                    "contest_date": row.get("contest_date"),
                    "version_tag": version_tag,
                    "dataset_metadata": parsed_metadata if isinstance(parsed_metadata, dict) else {"value": parsed_metadata},
                },
            )
        )

        if max_samples is not None and len(problems) >= max_samples:
            break

    return problems


DATASET_LOADERS = {
    "codecontests": load_codecontests,
    "livecodebench": load_livecodebench,
}
