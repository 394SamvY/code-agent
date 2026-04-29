---
name: debug-cleanup
description: Clean up a repository after an extended debugging session. Use when the user manually asks to run this skill, clean up legacy debug attempts, remove fallback logic, preserve useful failed experiments, delete throwaway artifacts, reconcile docs with current code, or leave a workspace in a clean self-consistent state for the next agent.
---

# Debug Cleanup

## Goal

Leave the workspace easier for the next agent to understand: one current implementation path, self-consistent docs, no stale fallback logic, and no leftover exploratory artifacts unless they are deliberately preserved.

This skill is cleanup-oriented, not feature-oriented. Do not introduce new product behavior except where a small edit is required to remove obsolete debug scaffolding or make the surviving code path coherent.

## Operating Principles

- Preserve user work and unrelated changes. Never revert, delete, or move files you did not understand.
- Favor the current agreed solution over broad backward compatibility from failed attempts.
- Remove dead fallback branches, duplicate implementations, stale TODOs, temporary logs, copied snippets, and notes that contradict the current code.
- Preserve experiments that teach a reusable lesson, document a non-obvious failure mode, or explain why an attractive approach was rejected.
- Delete artifacts that are mechanical, trivial, misleading, environment-specific, or only useful as transient scratch work.
- Keep preserved notes concise and dated. They should help future reasoning without becoming a second source of truth.
- Make docs match the current repository state. If docs and code disagree, either update the docs or flag the disagreement before finalizing.

## Workflow

1. Build a change inventory.
   - Run `git status --short`.
   - Inspect untracked files, modified docs, modified tests, and recently touched implementation files.
   - Use `git diff --stat`, targeted `git diff`, and `rg` searches for debug markers such as `TODO`, `FIXME`, `hack`, `fallback`, `legacy`, `debug`, `tmp`, `experiment`, `old`, `v2`, and obsolete tool or API names relevant to the repo.

2. Reconstruct the intended final path.
   - Read the current project guidance first, including repository `AGENTS.md` and any explicitly named source-of-truth docs.
   - Identify which implementation is the accepted one from the latest user decision, tests, docs, and code structure.
   - If the accepted path is ambiguous and cleanup could destroy useful work, ask the user a short clarifying question before editing.

3. Classify every suspect artifact.
   - **Keep current**: needed by the accepted implementation, tests, or current docs.
   - **Preserve**: useful failed experiment, benchmark result, design note, migration context, or bug investigation that future agents should be able to learn from.
   - **Delete**: throwaway scratch file, obsolete generated output, redundant fallback code, misleading docs, dead branch, temporary print/logging, abandoned shim, or stale copy of code.
   - **Defer**: unrelated user change or ambiguous file. Leave it untouched and mention it in the final response.

4. Clean code first.
   - Remove obsolete branches and fallback paths that are no longer part of the agreed protocol.
   - Delete unused helpers, imports, flags, compatibility aliases, and tests that only exercise removed behavior.
   - Keep edits narrow. Do not perform broad refactors unless they are necessary to make the surviving path self-consistent.
   - For this repository, respect OJ-like v1 constraints: tools are `run_public_tests` and `submit_solution`; do not restore `execute_code`, `test_list`, SFT paths, or function-benchmark protocols.

5. Clean docs second.
   - Update docs that describe the current behavior.
   - Remove or rewrite sections that document discarded approaches as if they are still active.
   - Keep top-level docs concise. If the repo has a documented source-of-truth policy, follow it.

6. Preserve valuable experiments.
   - Put concise debug notes, failure analyses, and investigation records under the repository's debug/history docs convention, preferring `docs/debug/` when no convention exists.
   - Put bulky generated outputs, retired datasets, old training assets, and no-longer-mainline experiments under the repository's legacy convention, preferring `docs/legacy/` when no convention exists.
   - Use dated names such as `docs/debug/YYYY-MM-DD-<topic>.md` or `docs/legacy/YYYY-MM-DD-<topic>/`.
   - Add a short note with: context, attempted approach, why it was rejected, what was learned, and which current path replaced it.
   - Do not preserve bulky generated outputs unless the user asked for them or they are uniquely valuable.

7. Verify consistency.
   - Run focused tests, linters, type checks, or smoke commands appropriate to the files changed.
   - Re-run `git status --short` and review the final diff.
   - Search once more for obsolete names or stale markers tied to the cleanup target.

## Final Response

Report:

- What was deleted.
- What was preserved and where.
- What docs or code were updated for consistency.
- What verification ran and whether it passed.
- Any unrelated or ambiguous changes intentionally left alone.

Keep the final answer brief and concrete. Do not describe every inspected file unless it changed or affected a decision.
