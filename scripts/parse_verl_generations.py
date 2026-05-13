#!/usr/bin/env python3
"""Add parsed chat messages to verl generation JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.trajectory_parser import add_messages


def _output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_messages{input_path.suffix}")


def parse_file(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    source = Path(input_path)
    target = Path(output_path) if output_path is not None else _output_path(source)

    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            dst.write(json.dumps(add_messages(record), ensure_ascii=False) + "\n")

    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input verl generation JSONL path.")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL path.")
    args = parser.parse_args()

    output_path = parse_file(args.input, args.output)
    print(output_path)


if __name__ == "__main__":
    main()
