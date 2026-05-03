# SFT Dataset 处理流程调试脚本使用说明

## 脚本功能

`debug_sft_dataset_processing.py` 用于模拟 verl SFT dataset 对训练数据的完整处理流程，展示：

1. 原始 parquet 数据的内容
2. 经过 `MultiTurnSFTDataset` 处理后的输出
3. tokenization 结果（input_ids, loss_mask, position_ids）
4. 解码后的文本内容
5. loss mask 的分布分析
6. enable_thinking 参数的影响

## 使用方法

### 在服务器上运行

```bash
# 1. 确保在项目根目录
cd /root/code/code-agent  # 或你的项目路径

# 2. 激活 verl 环境（如果需要）
# source /path/to/verl/env/bin/activate

# 3. 设置模型路径（可选，默认为 /root/autodl-tmp/models/Qwen3-8B）
export MODEL_PATH=/root/autodl-tmp/models/Qwen3-8B

# 4. 运行脚本
python3 debug_sft_dataset_processing.py
```

### 输出内容说明

脚本会输出以下信息：

1. **Tokenizer 信息**
   - tokenizer 类型
   - vocab_size
   - pad_token_id, eos_token_id

2. **原始样本内容**
   - task_id, title
   - enable_thinking 值
   - messages 数量和内容预览
   - tools 列表

3. **处理后的输出**
   - input_ids: shape, dtype, 序列长度, 前 50 个 token
   - loss_mask: shape, 需要计算 loss 的 token 数量和比例
   - position_ids: shape, 前 50 个位置

4. **解码文本**
   - 完整序列的解码文本（前 2000 字符）

5. **Loss Mask 分布**
   - 所有需要计算 loss 的区间
   - 每个区间的起止位置和文本预览

6. **统计信息**
   - 总 token 数
   - 训练 token 数 vs 忽略 token 数
   - 训练比例

7. **enable_thinking 验证**
   - 检查解码文本中是否包含 thinking 相关标签

## 预期结果

根据你的配置：
- `enable_thinking=False`
- 解码文本中**不应该**包含 `<thinking>`, `</thinking>`, `<think>`, `</think>` 标签
- loss_mask 只在 assistant 回复部分为 1（需要计算 loss）
- user 消息、system 消息、tool 调用结果部分的 loss_mask 为 0

## 依赖环境

脚本需要以下依赖：
- torch
- transformers
- pandas
- omegaconf
- verl (需要能 import verl.utils.dataset.multiturn_sft_dataset)

## 故障排查

### 如果提示找不到 verl 模块

```bash
# 确保 verl 已安装或在 PYTHONPATH 中
export PYTHONPATH=/path/to/verl:$PYTHONPATH
```

### 如果提示找不到模型

```bash
# 检查模型路径是否正确
ls -la /root/autodl-tmp/models/Qwen3-8B

# 或设置正确的模型路径
export MODEL_PATH=/your/actual/model/path
```

### 如果提示找不到数据文件

```bash
# 检查数据文件是否存在
ls -la data/verl/sft/sft_accepted_train.parquet
```

## 与训练脚本的对应关系

脚本中的配置参数对应 `scripts/train_sft_with_verl.sh` 中的：

| 脚本参数 | 训练脚本参数 |
|---------|-------------|
| `messages_key` | `data.messages_key=messages` |
| `tools_key` | `data.tools_key=tools` |
| `enable_thinking_key` | `data.enable_thinking_key=enable_thinking` |
| `enable_thinking_default` | `data.enable_thinking_default=false` |
| `pad_mode` | `data.pad_mode="$PAD_MODE"` |
| `max_length` | `data.max_length="$MAX_LENGTH"` |
| `truncation` | `data.truncation="$TRUNCATION"` |

## 下一步

运行脚本后，你可以：

1. 检查 loss_mask 的分布是否符合预期（只在 assistant 回复部分计算 loss）
2. 确认 enable_thinking=False 时没有 thinking 标签
3. 查看实际的 token 序列长度是否在合理范围内
4. 理解 verl 如何处理 multi-turn 对话和 tool calls
