#!/bin/bash
# ============================================================
# 下载 Qwen3-8B 模型到 models/ 目录
#
# 用法:
#   bash scripts/download_qwen3.sh              # 默认下载 Qwen3-8B
#   bash scripts/download_qwen3.sh Qwen3-4B     # 下载 Qwen3-4B
#   bash scripts/download_qwen3.sh Qwen3-1.7B   # 下载 Qwen3-1.7B
# ============================================================

set -e

# 禁用 xet 协议，避免 401 认证问题，走传统 HTTP 下载
export HF_HUB_DISABLE_XET=1

MODEL_NAME="${1:-Qwen3-8B}"
REPO_ID="Qwen/${MODEL_NAME}"
MODELS_DIR="$(cd "$(dirname "$0")/../.." && pwd)/models"
LOCAL_DIR="${MODELS_DIR}/${MODEL_NAME}"

echo "============================================"
echo "  下载模型: ${REPO_ID}"
echo "  保存到:   ${LOCAL_DIR}"
echo "============================================"

mkdir -p "${MODELS_DIR}"

python3 -c "
from huggingface_hub import snapshot_download
import os

repo_id = '${REPO_ID}'
local_dir = '${LOCAL_DIR}'

print(f'  正在下载/校验 {repo_id} (支持断点续传) ...')
snapshot_download(
    repo_id,
    local_dir=local_dir,
)
print(f'  下载完成: {local_dir}')

# 验证关键文件
for f in ['config.json', 'tokenizer_config.json']:
    path = os.path.join(local_dir, f)
    assert os.path.exists(path), f'缺少关键文件: {f}'
print('  文件校验通过')
"

echo ""
echo "下载完成！模型路径: ${LOCAL_DIR}"
echo ""
echo "下一步: 修改 SFT 配置中的 model.path 指向此路径"
