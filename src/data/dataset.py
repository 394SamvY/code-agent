"""
OJ-like 数据 schema 与 loader
============================

本文件是当前数据协议的 source of truth：所有外部数据集都先被映射成
`CodeProblem` / `OJTestCase`，后续 prompt、env、eval、verl 导出都只依赖这套
内部 schema，而不直接依赖原始数据集字段。

当前只保留两套主数据集：

- CodeContests：训练 / 开发验证
- LiveCodeBench：最终测试

主入口是 `load_codecontests()` 和 `load_livecodebench()`。它们负责把两个数据集
各自的原始字段、测试格式、metadata 统一到 OJ-like v1 schema。

v1 只支持完整 Python stdin/stdout 程序：

- 不支持文件 I/O 题
- 不支持 interactive problems
- 不支持 special judge
- 不在数据层预拼 prompt
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import zlib
from dataclasses import dataclass, field
from typing import Any, Literal

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


DatasetName = Literal["codecontests", "livecodebench"]
DatasetSplit = Literal["train", "valid", "test"]

# CodeContests 官方 proto enum -> 可读字符串。
# 这里只做 metadata 归一化，不参与过滤、训练 reward 或评测判定。
_CODECONTESTS_SOURCE_MAP = {
    0: "UNKNOWN_SOURCE",
    1: "CODECHEF",
    2: "CODEFORCES",
    3: "HACKEREARTH",
    4: "CODEJAM",
    6: "ATCODER",
    7: "AIZU",
}

# CodeContests 官方 proto Difficulty enum -> 可读字符串。
# 注意：这些值来自不同平台，不能当作全局单调递增难度。
# 对 Codeforces，metadata["cf_rating"] 可用时比 difficulty 更适合排序/分桶。
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

# CodeContests solutions.language 的官方 enum。
# 当前 loader 只保留 PYTHON3/PYTHON 参考解，其他语言不会进入 reference_solutions。
_CODECONTESTS_LANGUAGE_MAP = {
    0: "UNKNOWN_LANGUAGE",
    1: "PYTHON",
    2: "CPP",
    3: "PYTHON3",
    4: "JAVA",
}


@dataclass(frozen=True)
class OJTestCase:
    """结构化 OJ 测试用例，仅支持 stdin/stdout。

    `input` 是一次程序运行时写入 stdin 的完整文本。
    `output` 是期望从 stdout 得到的完整文本。

    judge 层会统一换行符后比较 `stdout.rstrip()` 和 `output.rstrip()`；
    数据层只负责保存原始测试，不做输出比较。
    """

    input: str
    output: str


@dataclass
class CodeProblem:
    """统一后的 OJ 编程题 schema。

    设计原则：

    - 顶层字段只放训练/评测主链路会稳定使用的内容。
    - 数据集特有字段、分析字段、不稳定字段放入 `metadata`。
    - `problem_statement` 保留原始题面，prompt 构造由 `src/prompts.py` 负责。
    - `public_tests` 给 `run_public_tests` 使用。
    - `private_tests` 给 `submit_solution` / full judge 使用。
    """

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
    """优先从本地读取，否则从 Hugging Face Hub 加载指定 split。

    `local_path` 用于服务器上已经缓存好的 dataset。它可能是单个 `Dataset`，
    也可能是包含多个 split 的 `DatasetDict`，所以这里统一做 split 选择。

    这里不做 split 名翻译；调用方直接使用真实数据集里的 `train` / `valid` / `test`。
    """
    if local_path and os.path.exists(local_path):
        ds_all = load_from_disk(local_path)
        if isinstance(ds_all, DatasetDict):
            if split in ds_all:
                return ds_all[split]
            raise KeyError(f"Split '{split}' not found in local dataset: {list(ds_all.keys())}")
        return ds_all

    return load_dataset(dataset_name, split=split, **load_kwargs)


def _stable_task_id(prefix: str, *parts: str) -> str:
    """根据题面稳定生成 task id。

    CodeContests 没有一个跨镜像稳定、适合直接暴露的唯一 id。
    这里用 title + description 做 sha1，保证同一份数据重复导出时 id 稳定。
    """
    joined = "\n".join(parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}/{digest}"


def _normalize_label(value: Any, value_map: dict[int, str]) -> str | None:
    """把 HF class label / proto enum / 字符串统一成可读字符串。"""
    if value is None:
        return None
    if isinstance(value, int):
        return value_map.get(value, str(value))
    if isinstance(value, str):
        return value
    return str(value)


def _parse_duration_seconds(raw: Any) -> float | None:
    """把数据集里的 time_limit 统一成秒。

    CodeContests 原始字段常见形式是 `{seconds, nanos}`，HF 变体或测试里也可能
    直接给 int/float。没有时间限制时返回 None，由 env 默认 timeout 接管。
    """
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
    """解析 CodeContests 风格的测试表示。

    CodeContests 常见格式是 dict-of-lists：

    ```
    {"input": ["1\\n", "2\\n"], "output": ["2\\n", "3\\n"]}
    ```

    有些测试或镜像也可能提供 list-of-dicts：

    ```
    [{"input": "1\\n", "output": "2\\n"}]
    ```

    两种都会统一成 `list[OJTestCase]`。
    """
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
    """解析 LiveCodeBench 中可能出现的 JSON 字符串或压缩 JSON 字符串。"""
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
    """解析 LiveCodeBench 的 public/private tests。

    LiveCodeBench 的测试字段可能已经是 list/dict，也可能是 JSON 字符串，
    某些变体还会用 base64(zlib(json)) 存 hidden tests。

    v1 只接受 `testtype == "stdin"`。遇到非 stdin 测试直接报错，而不是静默
    降级，因为 file I/O / function-call 测试会破坏当前 OJ 协议。
    """
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
    """提取 `(language, solution)` 序列。

    CodeContests 的 `solutions` 和 tests 类似，也常见 dict-of-lists：

    ```
    {"language": [3, 2], "solution": ["...", "..."]}
    ```

    这里兼容 list-of-dicts，是为了让单元测试和可能的 HF 镜像格式更容易接入。
    """
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
    """提取 Python 参考解。

    当前训练/eval 目标只支持 Python 程序，所以 CodeContests 默认过滤掉没有
    Python 参考解的问题。优先返回 `PYTHON3`，再返回 `PYTHON`。
    """
    solutions = _iter_solutions(raw)
    py3 = [solution for language, solution in solutions if language == "PYTHON3"]
    py2 = [solution for language, solution in solutions if language == "PYTHON"]
    return py3 + py2


def _is_stdio_problem(row: dict[str, Any]) -> bool:
    """判断 CodeContests 题目是否是 stdin/stdout 题。

    CodeContests 有些题要求从特定文件读写；v1 不支持这类题，loader 会过滤掉。
    """
    return not row.get("input_file") and not row.get("output_file")


def load_codecontests(
    split: DatasetSplit = "train",
    max_samples: int | None = None,
    local_path: str | None = None,
) -> list[CodeProblem]:
    """加载 CodeContests 并映射到统一 OJ schema。

    这是训练/验证数据的主 loader。处理流程：

    1. 从 `local_path` 或 Hugging Face 读取 `train` / `valid` / `test` split。
    2. 过滤 file I/O 题，只保留 stdin/stdout 题。
    3. 过滤没有 Python 参考解的问题。
    4. 解析 `public_tests`、`private_tests`、`generated_tests`。
    5. 将 `generated_tests` 合并进 `private_tests`，作为 full judge 的一部分。
    6. 将 source、cf_rating、memory_limit_bytes 等数据集特有字段下沉到 metadata。

    返回值只包含统一后的 `CodeProblem`，后续 env/eval 不再直接依赖 CodeContests
    原始字段。
    """
    ds = _select_split("deepmind/code_contests", split=split, local_path=local_path)

    problems: list[CodeProblem] = []
    for row in ds:
        if not _is_stdio_problem(row):
            continue

        # 只保留有 Python 参考解的问题。当前 agent 只生成 Python stdin/stdout 程序，
        # 没有 Python reference 的题先不进入训练数据。
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
                # generated_tests 在 CodeContests 中是由已有测试变体生成并经正确解验证的测试。
                # v1 不把它暴露成单独 action，而是并入 private/full judge。
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
    """加载 LiveCodeBench code generation lite 并映射到统一 OJ schema。

    这是最终 eval/test 的主 loader。当前只支持 `split="test"`，因为项目把
    LiveCodeBench 定位为最终测试集，不作为训练数据。

    处理流程：

    1. 从 `local_path` 或 Hugging Face 读取指定 `version_tag`。
    2. 使用 `question_id` 构造稳定 task id。
    3. 保留原始 `question_content` 作为 `problem_statement`。
    4. 解析 `public_test_cases` / `private_test_cases` 为 `OJTestCase`。
    5. 将平台、contest、release metadata 等下沉到 `metadata`。

    LiveCodeBench 当前不提供我们需要的 Python reference solutions，所以
    `reference_solutions` 为空。
    """
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
            # LCB 的 metadata 在不同镜像里可能是 JSON 字符串，也可能已经是 dict。
            # 解析失败时保留原始值，避免因为分析字段坏掉而丢弃整道题。
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


# 外部按名称取 loader 的 registry。eval / verl_dataset 通过这里避免硬编码函数名。
DATASET_LOADERS = {
    "codecontests": load_codecontests,
    "livecodebench": load_livecodebench,
}
