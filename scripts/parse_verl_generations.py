"""Add structured tool-event views to verl generation jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.trajectory_parser import add_structured_output


def parse_file(input_path: Path, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_structured.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w",
        encoding="utf-8",
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            dst.write(json.dumps(add_structured_output(record), ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse verl generation output text into structured tool events."
    )
    parser.add_argument("input", type=Path, help="Input generations jsonl file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output jsonl path. Defaults to *_structured.jsonl next to input.",
    )
    args = parser.parse_args()

    output = parse_file(args.input, args.output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
