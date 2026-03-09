#!/bin/bash
# ============================================================
# 在堡垒机上运行此脚本，提前下载模型、数据集、pip 包
# 用法: bash scripts/prepare_resources.sh
# ============================================================

set -e

BASE_DIR=~/yangenhui
MODELS_DIR=$BASE_DIR/models
DATASETS_DIR=$BASE_DIR/datasets
PIP_DIR=$BASE_DIR/pip_packages

echo "============================================================"
echo "资源准备脚本"
echo "所有资源将保存到: $BASE_DIR"
echo "============================================================"

# ---- 创建目录 ----
mkdir -p $MODELS_DIR $DATASETS_DIR $PIP_DIR

# ---- 1. 安装下载工具 ----
echo ""
echo "[1/4] 安装下载工具..."
pip install huggingface_hub datasets -q

# ---- 2. 下载模型 ----
echo ""
echo "[2/4] 下载模型..."

# 调试用小模型（0.5B，约 1GB）
echo "  下载 Qwen2.5-Coder-0.5B-Instruct..."
huggingface-cli download Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --local-dir $MODELS_DIR/Qwen2.5-Coder-0.5B-Instruct

# 正式训练模型（7B，约 14GB）
echo "  下载 Qwen2.5-Coder-7B-Instruct..."
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
    --local-dir $MODELS_DIR/Qwen2.5-Coder-7B-Instruct

# ---- 3. 下载数据集 ----
echo ""
echo "[3/4] 下载数据集..."

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

# ---- 4. 下载 pip 包 ----
echo ""
echo "[4/4] 下载 pip 离线包..."
pip download -r $BASE_DIR/code-agent/requirements.txt -d $PIP_DIR

# ---- 完成 ----
echo ""
echo "============================================================"
echo "资源准备完成！目录结构："
echo "============================================================"
echo "$BASE_DIR/"
echo "├── models/"
echo "│   ├── Qwen2.5-Coder-0.5B-Instruct/"
echo "│   └── Qwen2.5-Coder-7B-Instruct/"
echo "├── datasets/"
echo "│   ├── mbpp_full/"
echo "│   └── humaneval/"
echo "├── pip_packages/"
echo "└── code-agent/  (git clone 到这里)"
echo ""
echo "GPU 节点上的使用方式："
echo "  # 离线安装依赖"
echo "  pip install --no-index --find-links $PIP_DIR -r requirements.txt"
echo ""
echo "  # 跑 baseline"
echo "  cd $BASE_DIR/code-agent"
echo "  python -m src.eval.evaluate \\"
echo "      --model $MODELS_DIR/Qwen2.5-Coder-7B-Instruct \\"
echo "      --datasets mbpp_test humaneval \\"
echo "      --mode baseline \\"
echo "      --temperature 0.0 \\"
echo "      --data_dir $DATASETS_DIR"
