"""verl dataset adapter for the OJ-like parquet v1 schema.

动机：项目 parquet 文件的 `prompt`、`reward_model`、`extra_info` 三个字段存的是 JSON 字符串
（为了跨 pyarrow/pandas 版本兼容），但 verl 0.7.1 的 RLHFDataset 期望这些字段已经是 Python
dict/list 对象。这个 adapter 在读数据时动态 decode，不改 parquet 文件本身。

注入方式：通过 eval 脚本传入 Hydra override ——
  data.custom_cls.path=src/verl_dataset_adapter.py
  data.custom_cls.name=OJLikeRLHFDataset
verl 的 get_dataset_class() 会通过 load_extern_object 动态加载本类，替代默认 RLHFDataset。
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any

import torch

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import normalize_token_ids


logger = logging.getLogger(__name__)


def _json_load_if_str(value: Any, *, field_name: str) -> Any:
    """如果 value 是 JSON 格式的字符串就 decode，否则原样返回。

    比直接 json.loads 更安全：非字符串或 decode 失败都不抛异常，
    确保 adapter 不会因为意外的数据格式而让整个数据集加载崩溃。
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logger.warning(
            "Failed to decode JSON field %s; keeping raw string", field_name
        )
        return value


class OJLikeRLHFDataset(RLHFDataset):
    """继承 verl 的 RLHFDataset，在读 parquet 时将 JSON 字符串字段解码为 Python 对象。

    调用链回顾（详见 verl/utils/dataset/rl_dataset.py 的 get_dataset_class）：
      1. eval 脚本传 custom_cls.path / custom_cls.name
      2. main_ppo 的 create_rl_dataset → get_dataset_class → load_extern_object
      3. 加载本类，替代默认 RLHFDataset
      4. __init__ → _read_files_and_tokenize → maybe_filter_out_long_prompts
         （本类覆盖该方法，确保过滤阶段按 decode 后的真实 chat messages 计数）
      5. rollout 时 __getitem__ 逐条取 sample，本类 decode 后再组装 raw_prompt / tools_kwargs 等字段
    """

    # ── 内部工具方法 ──────────────────────────────────────────────
    def _decode_row(self, row_dict: dict[str, Any]) -> dict[str, Any]:
        """把一行的 prompt / reward_model / extra_info 从 JSON 字符串 decode 为 Python 对象。

        先 dict() 拷贝再修改，避免污染原始 dataframe 缓存。
        """
        row_dict = dict(row_dict)
        for field_name in ("prompt", "reward_model", "extra_info"):
            if field_name in row_dict:
                row_dict[field_name] = _json_load_if_str(
                    row_dict[field_name],
                    field_name=field_name,
                )
        return row_dict

    # ── 覆盖父类方法 ──────────────────────────────────────────────
    def _build_messages(self, example: dict):
        """构建 chat messages 列表，先 decode 再交给父类处理。

        调用时机：_read_files_and_tokenize → maybe_filter_out_long_prompts → doc2len
        在这里 decode 确保 token 长度计算时拿到的是真实的 list[dict] 而不是 JSON 字符串，
        否则遍历字符串会得到每个字符而不是每轮对话。
        """
        example = self._decode_row(example)
        return super()._build_messages(example)

    def maybe_filter_out_long_prompts(self, dataframe=None):
        """Filter overlong prompts after decoding JSON string fields.

        verl's text-only RLHFDataset path computes prompt length from
        ``doc[prompt_key]`` directly instead of calling ``_build_messages``.
        Our parquet stores ``prompt`` as a JSON string, so the parent method
        under-counts and leaves overlong prompts in validation batches.
        """
        if self.processor is not None or not self.filter_overlong_prompts:
            return super().maybe_filter_out_long_prompts(dataframe)

        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        def doc2len(doc) -> int:
            try:
                doc = self._decode_row(doc)
                apply_kwargs = dict(**self.apply_chat_template_kwargs)
                if self.tool_schemas is not None:
                    apply_kwargs["tools"] = self.tool_schemas

                apply_kwargs.pop("tokenize", None)
                apply_kwargs.pop("return_dict", None)
                apply_kwargs.pop("return_tensors", None)

                tokenized_prompt = tokenizer.apply_chat_template(
                    doc[prompt_key],
                    add_generation_prompt=True,
                    tokenize=True,
                    **apply_kwargs,
                )
                return len(normalize_token_ids(tokenized_prompt))
            except Exception:
                print("Error processing one of the samples, skipping...")
                traceback.print_exc()
                return self.max_prompt_length + 1

        dataframe = dataframe.filter(
            lambda doc: doc2len(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )
        print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def __getitem__(self, item):
        """rollout / validation 时 verl 逐条取 sample 的入口。

        流程：
        1. 从 dataframe 取该行，调用 _decode_row 解码 JSON 字符串字段
        2. 调 _build_messages 构建 raw_prompt（multi-turn chat messages 列表）
        3. 加 dummy_tensor 保持 DataProto.batch 非空（verl 的硬性要求）
        4. 从 extra_info 中提取 tools_kwargs / interaction_kwargs / index，
           这些是 verl agent loop 执行 tool call 和 reward 计算所必需的元数据
        """
        row_dict: dict = self._decode_row(self.dataframe[item])
        row_dict["raw_prompt"] = self._build_messages(row_dict)

        # Keep DataProto.batch non-empty, matching verl's default RLHFDataset.
        # verl 内部依赖 DataProto.batch 至少有一个 tensor，否则后续 concat 会崩溃。
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        # extra_info 是业务元数据的容器，包含该题目的工具参数和测试用例。
        # 防御性处理：不存在或被 decode 成非 dict 时用空 dict 替代。
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}
        if not isinstance(row_dict["extra_info"], dict):
            logger.warning(
                "extra_info is %s after decode; replacing with empty dict",
                type(row_dict["extra_info"]),
            )
            row_dict["extra_info"] = {}

        # 从 extra_info 中提取 agent loop 需要的字段：
        #   index:             样本序号
        #   tools_kwargs:      每个题目配套的工具参数（public/private tests、time limit 等）
        #                      传给 verl BaseTool 的 create() 方法，动态构建测试环境
        #   interaction_kwargs:多轮交互控制参数（如 max_turns）
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get(
            "interaction_kwargs", {}
        )
        need_tools_kwargs = row_dict.get("extra_info", {}).get(
            "need_tools_kwargs", self.need_tools_kwargs
        )
        if need_tools_kwargs and not tools_kwargs:
            logger.warning(
                "tools_kwargs is empty for index %s, data source: %s",
                index,
                row_dict["data_source"],
            )
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict
