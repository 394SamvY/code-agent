# CLAUDE.md

This file is the Claude Code-specific entry point for this repository. The shared project rules live elsewhere; do not duplicate them here.

## Read First

Before changing code, read:

1. `README.md`
2. `AGENTS.md`
3. `docs/project_status.md`

Then follow any task-specific documents linked from those files, especially `docs/specs/env_protocol.md` and `docs/specs/verl_parquet_dataset_analysis.md`.

## Claude-Specific Guidance

- Treat `AGENTS.md` as the shared agent operating guide and source of project constraints.
- Treat `docs/project_status.md` as the current progress and handoff entry point.
- Keep this file thin. If project rules, data status, baseline status, or command defaults change, update `AGENTS.md`, `README.md`, or `docs/project_status.md` instead of copying the full content here.
- Do not reintroduce paths or commands that no longer exist. Check the repository before adding tool-specific quick commands.
- 临时脚本运行时使用项目根目录下的 `.venv` 环境（如 `.venv/bin/python3 script.py`）。

## Current High-Level Context

This is an OJ-like code agent project:

- Training data is `CodeContests`; final test target is `LiveCodeBench`.
- The environment exposes `run_public_tests` and `submit_solution`.
- Current baseline work should use `scripts/evaluate_baseline_with_verl.sh` and the verl validation / agent loop path.

For the current status, blocker list, verification state, and next commands, read `docs/project_status.md`.
