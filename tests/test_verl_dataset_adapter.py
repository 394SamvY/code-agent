"""Focused tests for the verl dataset adapter."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.uint8 = "uint8"
    torch_stub.tensor = lambda value, dtype=None: value
    sys.modules["torch"] = torch_stub

try:
    import verl  # noqa: F401
except ModuleNotFoundError:
    verl_stub = types.ModuleType("verl")
    utils_stub = types.ModuleType("verl.utils")
    dataset_pkg_stub = types.ModuleType("verl.utils.dataset")
    rl_dataset_stub = types.ModuleType("verl.utils.dataset.rl_dataset")
    tokenizer_stub = types.ModuleType("verl.utils.tokenizer")

    class FakeRLHFDataset:
        def _build_messages(self, example):
            return example["prompt"]

    rl_dataset_stub.RLHFDataset = FakeRLHFDataset
    tokenizer_stub.normalize_token_ids = lambda token_ids: token_ids
    sys.modules["verl"] = verl_stub
    sys.modules["verl.utils"] = utils_stub
    sys.modules["verl.utils.dataset"] = dataset_pkg_stub
    sys.modules["verl.utils.dataset.rl_dataset"] = rl_dataset_stub
    sys.modules["verl.utils.tokenizer"] = tokenizer_stub

from src.verl_dataset_adapter import OJLikeRLHFDataset


class FakeTokenizer:
    def __init__(self) -> None:
        self.seen_messages = []

    def apply_chat_template(self, messages, **kwargs):
        if not isinstance(messages, list):
            raise TypeError("messages must be decoded before tokenization")
        self.seen_messages.append(messages)
        content = messages[0]["content"]
        token_count = len(content.split())
        if kwargs.get("tools"):
            token_count += 1
        return list(range(token_count))


def test_filter_decodes_json_prompt_when_filtering_enabled():
    try:
        from datasets import Dataset
    except ModuleNotFoundError:
        print("[SKIP] test_filter_decodes_json_prompt_when_filtering_enabled: datasets not installed")
        return

    adapter = OJLikeRLHFDataset.__new__(OJLikeRLHFDataset)
    adapter.processor = None
    adapter.filter_overlong_prompts = True
    adapter.tokenizer = FakeTokenizer()
    adapter.prompt_key = "prompt"
    adapter.max_prompt_length = 3
    adapter.apply_chat_template_kwargs = {}
    adapter.tool_schemas = [{"type": "function", "function": {"name": "dummy"}}]
    adapter.num_workers = None

    dataframe = Dataset.from_list(
        [
            {"prompt": json.dumps([{"role": "user", "content": "short"}])},
            {
                "prompt": json.dumps(
                    [{"role": "user", "content": "this prompt is too long"}]
                )
            },
        ]
    )

    filtered = adapter.maybe_filter_out_long_prompts(dataframe)

    assert len(filtered) == 1
    assert len(adapter.tokenizer.seen_messages) == 2
    assert all(isinstance(messages, list) for messages in adapter.tokenizer.seen_messages)

    print("[PASS] test_filter_decodes_json_prompt_when_filtering_enabled")


def test_decode_row_does_not_apply_prompt_style_overrides():
    adapter = OJLikeRLHFDataset.__new__(OJLikeRLHFDataset)
    original_env = os.environ.get("CODE_AGENT_PROMPT_STYLE")
    os.environ["CODE_AGENT_PROMPT_STYLE"] = "short_thinking"

    try:
        prompt = [
            {"role": "system", "content": "base system prompt"},
            {"role": "user", "content": "solve"},
        ]
        row = {"prompt": json.dumps(prompt)}

        decoded = adapter._decode_row(row)

        assert decoded["prompt"] == prompt
    finally:
        if original_env is None:
            os.environ.pop("CODE_AGENT_PROMPT_STYLE", None)
        else:
            os.environ["CODE_AGENT_PROMPT_STYLE"] = original_env

    print("[PASS] test_decode_row_does_not_apply_prompt_style_overrides")


if __name__ == "__main__":
    test_filter_decodes_json_prompt_when_filtering_enabled()
    test_decode_row_does_not_apply_prompt_style_overrides()
    print("\nAll tests passed!")
