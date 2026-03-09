#!/bin/bash
# ============================================================
# 在登录节点（ln02/ln03）运行：下载模型、数据集、安装依赖
# 前提：已设置代理（见下方说明）
#
# 代理设置（替换为你的真实账号和令牌）：
#   export https_proxy=https://你的AD用户名:你的令牌@blsc-proxy.pjlab.org.cn:13128
#   export http_proxy=https://你的AD用户名:你的令牌@blsc-proxy.pjlab.org.cn:13128
#
# 用法: bash scripts/prepare_resources.sh
# ============================================================

set -e

BASE_DIR=~/yangenhui
MODELS_DIR=$BASE_DIR/models
DATASETS_DIR=$BASE_DIR/datasets
CODE_DIR=$BASE_DIR/code-agent

echo "============================================================"
echo "资源准备脚本（宁夏超算 N40R4PJ 区）"
echo "============================================================"

# ---- 检查代理 ----
if [ -z "$https_proxy" ]; then
    echo "[ERROR] 未设置代理，请先执行："
    echo "  export https_proxy=https://用户名:令牌@blsc-proxy.pjlab.org.cn:13128"
    echo "  export http_proxy=https://用户名:令牌@blsc-proxy.pjlab.org.cn:13128"
    exit 1
fi

echo "代理已设置: $https_proxy"
echo ""

# ---- 检查网络 ----
module load curl/8.7.1 2>/dev/null || true
echo "检查网络连通性..."
curl -s --max-time 10 www.baidu.com > /dev/null && echo "  网络OK" || { echo "  [ERROR] 无法联网，请检查代理设置"; exit 1; }

mkdir -p $MODELS_DIR $DATASETS_DIR

# ---- 1. 创建 conda 环境 ----
echo ""
echo "[1/3] 创建 conda 环境..."
module load miniforge/25.3.0-3

if conda env list | grep -q "code-agent"; then
    echo "  conda 环境 code-agent 已存在，跳过创建"
else
    conda create -n code-agent python=3.11 -y
fi

source activate code-agent

# 安装依赖（不要加 --user）
echo "  安装 Python 依赖..."
pip install -r $CODE_DIR/requirements.txt

# ---- 2. 下载模型 ----
echo ""
echo "[2/3] 下载模型..."

python3 -c "
from huggingface_hub import snapshot_download

print('  下载 Qwen2.5-Coder-0.5B-Instruct（调试用，约 1GB）...')
snapshot_download('Qwen/Qwen2.5-Coder-0.5B-Instruct', local_dir='$MODELS_DIR/Qwen2.5-Coder-0.5B-Instruct')

print('  下载 Qwen2.5-Coder-7B-Instruct（正式训练，约 14GB）...')
snapshot_download('Qwen/Qwen2.5-Coder-7B-Instruct', local_dir='$MODELS_DIR/Qwen2.5-Coder-7B-Instruct')

print('  模型下载完成')
"

# ---- 3. 下载数据集 ----
echo ""
echo "[3/3] 下载数据集..."

python3 -c "
from datasets import load_dataset

print('  下载 MBPP...')
ds = load_dataset('google-research-datasets/mbpp', 'full')
ds.save_to_disk('$DATASETS_DIR/mbpp_full')
print(f'  MBPP: train={len(ds[\"train\"])}, val={len(ds[\"validation\"])}, test={len(ds[\"test\"])}')

print('  下载 HumanEval...')
ds = load_dataset('openai/openai_humaneval')
ds.save_to_disk('$DATASETS_DIR/humaneval')
print(f'  HumanEval: {len(ds[\"test\"])} 题')

print('  数据集下载完成')
"

# ---- 完成 ----
echo ""
echo "============================================================"
echo "资源准备完成！"
echo "============================================================"
echo ""
echo "目录结构："
echo "$BASE_DIR/"
echo "├── code-agent/        # 项目代码"
echo "├── models/"
echo "│   ├── Qwen2.5-Coder-0.5B-Instruct/"
echo "│   └── Qwen2.5-Coder-7B-Instruct/"
echo "└── datasets/"
echo "    ├── mbpp_full/"
echo "    └── humaneval/"
echo ""
echo "下一步：提交 GPU 任务"
echo "  sbatch -N 1 -n 6 --gres=gpu:1 -p <你的分区> -A <你的account> scripts/run_baseline_slurm.sh"
