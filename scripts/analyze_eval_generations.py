#!/usr/bin/env python3
"""Summarize OJ-like verl validation generations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
from statistics import mean, median
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _avg(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _p50(values: list[float]) -> float:
    return median(values) if values else 0.0


def _is_unclosed_think(output: str) -> bool:
    last_open = output.rfind("<think>")
    last_close = output.rfind("</think>")
    return last_open != -1 and last_open > last_close


def _accepted_tail_chars(output: str) -> int:
    marker = "submit_solution: accepted"
    index = output.find(marker)
    if index == -1:
        return 0
    return max(0, len(output) - (index + len(marker)))


def _summarize_gpu_csv(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    if not rows:
        return {}

    by_gpu: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_gpu.setdefault(row["gpu_id"].strip(), []).append(row)

    summary: dict[str, Any] = {}
    for gpu_id, gpu_rows in sorted(by_gpu.items()):
        mem_used = [float(row["mem_used_mb"]) for row in gpu_rows]
        util = [float(row["gpu_util%"]) for row in gpu_rows]
        timestamps = [
            datetime.strptime(row["timestamp"].strip(), "%Y-%m-%d %H:%M:%S")
            for row in gpu_rows
        ]
        duration_seconds = (
            (timestamps[-1] - timestamps[0]).total_seconds()
            if len(timestamps) > 1
            else 0.0
        )
        sample_interval = duration_seconds / (len(timestamps) - 1) if len(timestamps) > 1 else 0.0
        active_util = [
            float(row["gpu_util%"])
            for row in gpu_rows
            if float(row["mem_used_mb"]) > 20_000
        ]
        summary[gpu_id] = {
            "samples": len(gpu_rows),
            "duration_seconds": duration_seconds,
            "active_seconds": len(active_util) * sample_interval,
            "max_mem_gb": max(mem_used) / 1024,
            "avg_util": _avg(util),
            "avg_active_util": _avg(active_util),
        }
    return summary


def summarize(records: list[dict[str, Any]], gpu_summary: dict[str, Any]) -> str:
    outputs = [str(record.get("output", "")) for record in records]
    scores = [float(record.get("score", 0.0) or 0.0) for record in records]
    tool_calls = [
        int(record.get("num_tool_calls", output.count("<tool_call>")) or 0)
        for record, output in zip(records, outputs)
    ]
    accepted = [score >= 1.0 for score in scores]
    submit_samples = ["submit_solution" in output for output in outputs]
    unclosed_thinks = [_is_unclosed_think(output) for output in outputs]
    output_chars = [len(output) for output in outputs]
    accepted_tails = [
        _accepted_tail_chars(output)
        for ok, output in zip(accepted, outputs)
        if ok and _accepted_tail_chars(output) > 0
    ]

    n = len(records)
    lines = [
        "==== eval generation summary ====",
        f"samples: {n}",
        f"score_mean: {_avg(scores):.4f}",
        f"accepted_rate: {_pct(sum(accepted) / n) if n else '0.0%'}",
        f"zero_score_rate: {_pct(sum(score == 0.0 for score in scores) / n) if n else '0.0%'}",
        f"unclosed_think_rate: {_pct(sum(unclosed_thinks) / n) if n else '0.0%'}",
        f"num_tool_calls_zero_rate: {_pct(sum(value == 0 for value in tool_calls) / n) if n else '0.0%'}",
        f"submit_sample_rate: {_pct(sum(submit_samples) / n) if n else '0.0%'}",
        f"avg_tool_calls: {_avg([float(value) for value in tool_calls]):.2f}",
        f"max_tool_calls: {max(tool_calls) if tool_calls else 0}",
        f"avg_output_chars: {_avg([float(value) for value in output_chars]):.0f}",
        f"p50_output_chars: {_p50([float(value) for value in output_chars]):.0f}",
        f"max_output_chars: {max(output_chars) if output_chars else 0}",
        f"accepted_tail_avg_chars: {_avg([float(value) for value in accepted_tails]):.0f}",
        f"accepted_tail_max_chars: {max(accepted_tails) if accepted_tails else 0}",
    ]

    if gpu_summary:
        lines.append("")
        lines.append("==== gpu summary ====")
        for gpu_id, item in gpu_summary.items():
            lines.append(
                "gpu {gpu}: max_mem={max_mem_gb:.1f}GB avg_util={avg_util:.1f}% "
                "avg_active_util={avg_active_util:.1f}% duration={duration_seconds:.0f}s "
                "active={active_seconds:.0f}s sec_per_sample={sec_per_sample:.1f} "
                "samples={samples}".format(
                    gpu=gpu_id,
                    sec_per_sample=(float(item["duration_seconds"]) / n) if n else 0.0,
                    **item,
                )
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--gpu-csv", type=Path, default=None)
    args = parser.parse_args()

    records = _load_jsonl(args.jsonl)
    gpu_summary = _summarize_gpu_csv(args.gpu_csv)
    print(summarize(records, gpu_summary))


if __name__ == "__main__":
    main()
