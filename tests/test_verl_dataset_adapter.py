"""Focused tests for the verl dataset adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


if __name__ == "__main__":
    test_filter_decodes_json_prompt_when_filtering_enabled()
    print("\nAll tests passed!")
