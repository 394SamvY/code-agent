---
name: project-context-sync
description: Synchronize a project's current context into its documentation after meaningful work, before handoff, before context compaction, or when docs feel scattered. Use when the user asks to update progress docs, create or refresh a handoff, reconcile README/AGENTS/CLAUDE/docs status, standardize project documentation, record decisions, or make the repository easier for the next agent to understand.
---

# Project Context Sync

## Goal

Make the repository's written context match the real project state, with fewer places to look and fewer stale claims. The next agent should quickly understand what is current, what changed, what is blocked, what was verified, and where the canonical information lives.

This skill updates context and documentation. It does not clean debug leftovers, remove code paths, or preserve failed experiments unless the user explicitly asks for that too. For cleanup of stale attempts, use a cleanup-oriented skill first, then run this skill.

## Core Principles

- Reduce documentation entry points instead of creating more of them.
- Record current facts, decisions, verification status, blockers, and next steps; avoid chat transcripts and speculative history.
- Treat existing project instructions as discoverable conventions, not immutable law. Improve them when the user wants standardization.
- Keep canonical docs canonical. Do not duplicate long sections across README, agent docs, tool-specific docs, and progress docs.
- Prefer one project status or handoff document when the repo lacks a clear progress entry point.
- If a claim says work is complete, fixed, passing, or current, back it with fresh evidence or label it unverified.
- Preserve user work. Do not delete or rewrite large docs blindly; propose a structure first when changes are broad.

## Workflow

1. Discover the documentation map.
   - Read repository guidance files such as `AGENTS.md`, `CLAUDE.md`, `.cursor/rules`, `.github/copilot-instructions.md`, `README.md`, and obvious `docs/**/*.md`.
   - Use `rg --files -g '*.md'` to find documentation. Avoid deep-reading legacy material unless it is clearly the active source.
   - Run `git status --short` to understand what changed and what may need synchronization.

2. Identify document roles.
   - **Entry doc**: human-facing overview, usually `README.md`.
   - **Agent doc**: agent-facing operating rules, often `AGENTS.md`, `CLAUDE.md`, or tool-specific instruction files.
   - **Status doc**: current phase, progress, blockers, validation, next steps.
   - **Protocol/spec docs**: durable behavior, APIs, schemas, architecture, data contracts.
   - **Operations docs**: how to run, train, evaluate, deploy, and troubleshoot.
   - **Reference docs**: source reading guides, design background, and external references.
   - **Decision docs**: durable rationale for choices that future agents should not reopen casually.
   - **Debug docs**: dated investigations, failure analyses, and historical bug processes.
   - **Legacy assets**: large retired outputs, old experiments, superseded datasets, or inactive training artifacts.

3. Choose a sync mode.
   - **Light sync**: update existing status/handoff docs and small summaries when the document structure is already coherent.
   - **Normalize docs**: propose and then implement a cleaner documentation layout when progress is scattered, files duplicate each other, or tool-specific docs conflict.
   - **Decision capture**: write or update concise decision notes when the main missing context is rationale.

4. Establish or update the status entry point.
   - If the repo already has a clear status/handoff file, update it.
   - If not, create one conventional file, favoring `docs/project_status.md` unless the repo uses another pattern.
   - Include only high-signal sections:
     - Current phase
     - Current working state
     - Completed since last sync
     - Active decisions and constraints
     - Verification status
     - Blockers or risks
     - Next steps
     - Pointers to canonical docs

5. Reconcile overlapping docs.
   - Move volatile progress out of README and agent instruction files when it makes them noisy.
   - Keep README focused on project overview and common commands.
   - Keep agent docs focused on operating constraints, source-of-truth pointers, and project-specific gotchas.
   - Keep tool-specific docs thin when they overlap with a common agent doc; point them to the canonical source instead of copying it.
   - Update protocol/spec docs only when durable behavior has changed.

6. Capture decisions without bloating status.
   - Use existing decision records if present.
   - If none exist and the decision is important, create a concise note under `docs/decisions/` with a dated filename.
   - Record: decision, context, rejected alternatives, consequences, and links to affected docs or code.
   - Do not write a decision record for routine implementation details.

7. Verify documentation consistency.
   - Search for stale names, old commands, deprecated paths, obsolete statuses, and repeated claims.
   - Check that README, agent docs, status docs, and specs do not contradict each other.
   - If possible, run or cite the command that proves any newly written verification claim.
   - Review the final diff before responding.

## Normalization Guidance

When asked to standardize messy docs, prefer a small durable structure:

```text
README.md
AGENTS.md
CLAUDE.md or other tool-specific docs, if needed
docs/
  project_status.md
  specs/
  operations/
  references/
  decisions/
  debug/
  legacy/
```

This is a guideline, not a required layout. Follow existing repository conventions when they are already coherent.

Default meanings:

- `docs/specs/`: stable protocols, schemas, APIs, interfaces, and data contracts.
- `docs/operations/`: runbooks for running, training, evaluating, deploying, and troubleshooting.
- `docs/references/`: design background, source reading maps, and external references.
- `docs/decisions/`: durable decisions and rejected alternatives.
- `docs/debug/`: dated debug records, failure analyses, and historical bug investigations.
- `docs/legacy/`: no-longer-mainline large outputs, old experiments, retired datasets, and inactive training assets.

## What Not To Do

- Do not create multiple new progress files for one project.
- Do not copy the same project constraints into every tool-specific instruction file.
- Do not turn status docs into chronological chat logs.
- Do not mark work complete without fresh verification evidence.
- Do not move large documentation sets without explaining the proposed target structure first.
- Do not delete historical material just because it is old; classify it as active, stale, debug-worthy, or legacy-worthy.
- Do not use this skill as a substitute for code cleanup.

## Final Response

Report:

- Which docs were created or updated.
- Which document is now the primary status/handoff entry point.
- What stale or conflicting context was reconciled.
- What verification evidence supports the updated status.
- Any ambiguous or unrelated docs left unchanged.
