#!/usr/bin/env bash

# 后台环境指令训练启动脚本
# 使用示例：
#   chmod +x run_env_instruct_train.sh
#   nohup ./run_env_instruct_train.sh > train_env_instruct.log 2>&1 &

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo ">>> [$(date)] Start env-instruct training in $REPO_ROOT"

# 1) 初始化 conda（根据你本机安装路径，如有不同自行修改）
if [ -f "/home/exouser/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/home/exouser/miniconda3/etc/profile.d/conda.sh"
fi

# 2) 激活训练环境
conda activate tts

# 3) 设置预训练模型路径（放在大盘上）
export PRETRAINED_DIR="/media/volume/4train/pretrained_models/Fun-CosyVoice3-0.5B"

# 4) 确保仓库在 PYTHONPATH 中
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# 5) 如果你有多张 GPU，可以在这里改 CUDA_VISIBLE_DEVICES
# 例如：只用 0、1 两张卡：
# export CUDA_VISIBLE_DEVICES=0,1

# 6) 启动训练（llm -> flow -> hifigan）
bash env_instruct_pipeline/scripts/run_train_room.sh

echo ">>> [$(date)] Training script finished"

