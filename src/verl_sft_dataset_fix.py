"""
修复 MultiTurnSFTDataset 逐条调用 chat_template 导致 <think> 被丢弃的问题。

== 场景 ==
一条典型的训练轨迹包含 6 条消息:
  [0] system
  [1] user        (题目)
  [2] assistant   (<think>分析</think><tool_call>{"name":"run_public_tests",...}</tool_call>)
  [3] tool        (run_public_tests 结果)
  [4] assistant   (<think>提交</think><tool_call>{"name":"submit_solution",...}</tool_call>)
  [5] tool        (submit_solution 结果)

== verl 的处理方式 (MultiTurnSFTDataset._process_single_message) ==
为了精确控制 loss_mask（assistant 参与训练，其他角色不参与），verl 对每条消息
单独调用一次 tokenizer.apply_chat_template(messages=[单条消息], ...)，拿到各段
token 序列后 torch.cat 拼接。

== Qwen Thinking chat_template 的处理逻辑 ==
Template 分两步:

1. 找 "最后一个真正的用户提问" 的位置 (chat_template.jinja:17-24):
   last_query_index 初始 = messages 总数 - 1
   从后往前扫描，遇到 role=user 且不是 <tool_response> 包装的消息就记录其索引

   设计意图: 多轮对话中，只有 "最后用户提问之后" 的 assistant 才需要渲染 <think>，
   之前的 assistant 是历史上下文，不应渲染。

2. 渲染 assistant 消息 (chat_template.jinja:38-51):
   先拆出 reasoning_content（<think> 和 </think> 之间的内容）和 content（</think> 之后的部分）。
   然后判断:
     if loop.index0 > last_query_index:  → 渲染带 <think> 包装的完整格式
     else:                               → 丢弃 reasoning_content，只输出 content

== 问题 ==
verl 逐条传入 messages=[一条assistant] 时:
  - messages 长度 = 1
  - last_query_index 初始 = 0
  - 扫描找 user → 没有 → last_query_index 保持 0
  - assistant 的 loop.index0 = 0
  - 判断: 0 > 0 → False → 走 else

结果: reasoning_content（思考内容）被丢弃，训练序列变成:
  <|im_start|>assistant\n<tool_call>{"name":"run_public_tests",...}</tool_call>

模型从头到尾没见过 <think> 和 </think> 出现在 assistant 消息中（训练序列中
<think> token 出现 0 次），因此没学到 "先思考再调用工具" 的行为模式。

== 修复 ==
将 `>` 改为 `>=`。当 last_query_index 和 loop.index0 相等时（逐条传入且 assistant
是唯一消息），也渲染 <think> 包装。对于整段对话的场景，assistant 的 index 本来就
大于 last_query_index，`>=` 和 `>` 结果完全相同。
"""

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class FixedMultiTurnSFTDataset(MultiTurnSFTDataset):
    """Fix two Qwen3 chat_template issues for per-turn SFT tokenization.

    Fix 1: ``>`` → ``>=``
      In per-turn mode, each assistant message is the only message, so
      loop.index0 == last_query_index == 0.  The original ``>`` misses it.
      With ``>=``, the template treats it as if it follows the last user query
      and renders the <think> wrapper.

    Fix 2: ``loop.last or (not loop.last and reasoning_content)`` → ``True``
      Even after fix 1, the template has a second guard: for non-last
      assistant messages it only renders <think> when reasoning_content is
      non-empty.  In per-turn mode every message is ``loop.last``, so
      <think> always fires.  When verl's sanity_check applies the template
      to the full conversation at once, empty-reasoning assistants skip
      <think> — causing the per-turn concat and full-conversation token
      sequences to diverge.  Forcing ``True`` makes both paths consistent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        template = self.tokenizer.chat_template
        if isinstance(template, str):
            template = template.replace(
                "loop.index0 > ns.last_query_index",
                "loop.index0 >= ns.last_query_index",
            )
            template = template.replace(
                "loop.last or (not loop.last and reasoning_content)",
                "True",
            )
            self.tokenizer.chat_template = template
