#!/bin/bash
# ============================================================
# 在堡垒机上运行：下载模型、数据集
# 前提：已 module load miniforge && conda activate code-agent
# 用法: bash scripts/prepare_resources.sh
# ============================================================

set -e

BASE_DIR=~/yangenhui
MODELS_DIR=$BASE_DIR/models
DATASETS_DIR=$BASE_DIR/datasets

echo "============================================================"
echo "资源准备脚本"
echo "所有资源将保存到: $BASE_DIR"
echo "============================================================"

mkdir -p $MODELS_DIR $DATASETS_DIR

# ---- 1. 下载模型 ----
echo ""
echo "[1/2] 下载模型..."

python3 -c "
from huggingface_hub import snapshot_download

print('  下载 Qwen2.5-Coder-0.5B-Instruct（调试用，约 1GB）...')
snapshot_download('Qwen/Qwen2.5-Coder-0.5B-Instruct', local_dir='$MODELS_DIR/Qwen2.5-Coder-0.5B-Instruct')

print('  下载 Qwen2.5-Coder-7B-Instruct（正式训练，约 14GB）...')
snapshot_download('Qwen/Qwen2.5-Coder-7B-Instruct', local_dir='$MODELS_DIR/Qwen2.5-Coder-7B-Instruct')

print('  模型下载完成')
"

# ---- 2. 下载数据集 ----
echo ""
echo "[2/2] 下载数据集..."

python3 -c "
from datasets import load_dataset

print('  下载 MBPP...')
ds = load_dataset('google-research-datasets/mbpp', 'full')
ds.save_to_disk('$DATASETS_DIR/mbpp_full')
print(f'  MBPP 已保存: train={len(ds[\"train\"])}, val={len(ds[\"validation\"])}, test={len(ds[\"test\"])}')

print('  下载 HumanEval...')
ds = load_dataset('openai/openai_humaneval')
ds.save_to_disk('$DATASETS_DIR/humaneval')
print(f'  HumanEval 已保存: {len(ds[\"test\"])} 题')

print('  数据集下载完成')
"

# ---- 完成 ----
echo ""
echo "============================================================"
echo "资源准备完成！"
echo "============================================================"
echo ""
echo "$BASE_DIR/"
echo "├── models/"
echo "│   ├── Qwen2.5-Coder-0.5B-Instruct/"
echo "│   └── Qwen2.5-Coder-7B-Instruct/"
echo "├── datasets/"
echo "│   ├── mbpp_full/"
echo "│   └── humaneval/"
echo "└── code-agent/"
echo ""
echo "GPU 节点上运行 baseline："
echo "  module load miniforge/25.3.0-3 cuda/12.4"
echo "  conda activate code-agent"
echo "  cd $BASE_DIR/code-agent"
echo "  python -m src.eval.evaluate \\"
echo "      --model $MODELS_DIR/Qwen2.5-Coder-7B-Instruct \\"
echo "      --datasets mbpp_test humaneval \\"
echo "      --mode baseline \\"
echo "      --temperature 0.0 \\"
echo "      --data_dir $DATASETS_DIR"
