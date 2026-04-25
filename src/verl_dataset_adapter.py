"""verl dataset adapter for the OJ-like parquet v1 schema.

The repo's source-of-truth parquet schema stores `prompt`, `reward_model`, and
`extra_info` as JSON strings for stable pyarrow/pandas interoperability. Current
verl versions expect these fields to already be nested Python objects, so this
adapter decodes them at dataset-read time without changing the parquet files.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import torch

from verl.utils.dataset.rl_dataset import RLHFDataset


logger = logging.getLogger(__name__)


def _json_load_if_str(value: Any, *, field_name: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Failed to decode JSON field %s; keeping raw string", field_name)
        return value


class OJLikeRLHFDataset(RLHFDataset):
    """Decode OJ-like v1 JSON-string parquet fields for verl."""

    def _decode_row(self, row_dict: dict[str, Any]) -> dict[str, Any]:
        row_dict = dict(row_dict)
        for field_name in ("prompt", "reward_model", "extra_info"):
            if field_name in row_dict:
                row_dict[field_name] = _json_load_if_str(
                    row_dict[field_name],
                    field_name=field_name,
                )
        return row_dict

    def _build_messages(self, example: dict):
        example = self._decode_row(example)
        return super()._build_messages(example)

    def __getitem__(self, item):
        """Mirror RLHFDataset.__getitem__ after decoding JSON-string fields."""
        row_dict: dict = self._decode_row(self.dataframe[item])
        row_dict["raw_prompt"] = self._build_messages(row_dict)

        # Keep DataProto.batch non-empty, matching verl's default RLHFDataset.
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}
        if not isinstance(row_dict["extra_info"], dict):
            logger.warning("extra_info is %s after decode; replacing with empty dict", type(row_dict["extra_info"]))
            row_dict["extra_info"] = {}

        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict
