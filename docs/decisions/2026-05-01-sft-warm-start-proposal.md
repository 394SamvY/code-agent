# 2026-05-01 SFT Warm-Start Proposal

Status: proposed, not adopted.

## Context

The current project policy is RL-only. This document does not change that policy by itself; it records a possible direction if we decide that a small behavior warm-start is worth introducing before RL.

The recent Qwen3 thinking-budget investigation found:

- Qwen3 can generate very long reasoning before tool calls.
- SGLang's built-in `Qwen3ThinkingBudgetLogitProcessor` works for the first assistant turn, but is not a clean fit for multi-turn tool-agent rollouts because previous `</think>` tokens remain in the prompt.
- The existing two-pass runtime budget can control correctness, but it is still a runtime workaround.

The alternative is to move the base model's behavior distribution closer to the desired agent behavior: short reasoning, prompt tool use, and concise repair after tool feedback.

## Goal

Train a small SFT or LoRA warm-start adapter that teaches the model to:

- think briefly before each action;
- call `run_public_tests` before final submission when useful;
- use tool feedback to make targeted fixes;
- call `submit_solution` once public behavior looks correct;
- avoid long free-form reasoning inside code comments or after tool calls;
- preserve the current OJ-like two-tool protocol.

The warm-start should reduce dependence on complex thinking-budget intervention, but it should not remove hard safety caps such as `response_length`, per-turn `max_new_tokens`, and `max_tool_calls`.

## Non-Goals

- Do not reintroduce function-benchmark or `execute_code` style protocols.
- Do not train on final test sets or LiveCodeBench held-out data.
- Do not make SFT the main optimization objective unless a later decision explicitly changes the RL-only policy.
- Do not optimize for pretty chain-of-thought; optimize for stable agent actions and judge feedback use.

## Data Plan

Use CodeContests training problems only.

Generate trajectories with a stronger teacher model under the exact current environment protocol:

1. Input is the same OJ-style prompt and public tests.
2. Available tools are exactly `run_public_tests` and `submit_solution`.
3. Public-test calls return observations only.
4. Full submit is the only terminal reward source.
5. `max_submissions=5` remains the environment rule.

Recommended first dataset size:

- Pilot: 200-500 trajectories for format and filter validation.
- First SFT experiment: 1k-5k high-quality trajectories.
- Only scale further after behavior metrics improve on held-out CodeContests validation.

## Target Trajectory Format

Each assistant turn should follow this shape:

```text
<think>
Short plan or diagnosis, normally 128-512 tokens.
</think>

<tool_call>
{"name": "run_public_tests", "arguments": {"code": "..."}}
</tool_call>
```

After public-test failure:

```text
<think>
Briefly identify the failing behavior and the concrete fix.
</think>

<tool_call>
{"name": "run_public_tests", "arguments": {"code": "...fixed..."}}
</tool_call>
```

Final accepted attempt:

```text
<think>
Public behavior is consistent; submit the final version.
</think>

<tool_call>
{"name": "submit_solution", "arguments": {"code": "..."}}
</tool_call>
```

## Filtering Rules

Keep trajectories only if:

- all tool calls parse as JSON;
- all tool names are in `{run_public_tests, submit_solution}`;
- every tool call has a string `code` argument;
- there is at least one final `submit_solution`;
- terminal semantics match the environment;
- no assistant turn exceeds the chosen per-turn token budget;
- no thinking span is unclosed;
- code is a complete stdin/stdout Python program;
- the trajectory either reaches accepted or demonstrates a clear repair pattern from public feedback.

Prefer accepted trajectories. Keep a smaller controlled slice of failed-but-instructive trajectories only if they show valid tool use and useful repair behavior.

## Training Plan

Start with LoRA SFT against the current Qwen3-8B base.

Suggested first configuration:

- train on assistant tokens only;
- include tool responses in context, but mask them from loss;
- preserve `<think>`, `</think>`, `<tool_call>`, and `</tool_call>` in the target;
- use short sequence packing only after verifying it does not merge independent tool trajectories incorrectly;
- train for 1 epoch first;
- keep the adapter separate from the RL path until evaluation passes.

If the repo later adopts this direction, add a dedicated script rather than modifying the main GRPO entrypoint.

## Evaluation Gates

Evaluate the adapter with the same verl validation path before any RL use.

Primary behavior metrics:

- tool-call parse success rate;
- average thinking tokens per assistant turn;
- fraction of turns with closed thinking spans;
- average tool calls per problem;
- fraction of trajectories that call `submit_solution`;
- public-fail-to-fix rate;
- accepted rate on CodeContests validation.

Required correctness gates:

- P0 audit still passes.
- `run_public_tests` accepted never terminates the trajectory.
- only `submit_solution` accepted or submission-limit terminates.
- no regression to legacy tools or function-completion protocol.

## Risks

- SFT may reduce exploration too much for RL if the dataset is narrow.
- Teacher trajectories can leak brittle solution styles or overfit public tests.
- Short thinking may improve tool hygiene but hurt hard-problem reasoning.
- Adding SFT changes the project policy and should be recorded as a separate decision before becoming the default path.

## Proposed Rollout

1. Keep current two-pass runtime budget for baseline correctness.
2. Build a small teacher-trajectory generation script against CodeContests train.
3. Generate and audit a 200-500 trajectory pilot.
4. Train a LoRA SFT adapter.
5. Run 32-sample and then 500-sample CodeContests validation through the existing verl eval path.
6. Compare against the no-SFT baseline on behavior metrics and accepted rate.
7. If clearly better, write a new decision changing the project from RL-only to SFT warm-start + RL.
