#!/bin/bash
# 生成 5000 条「房间大小/混响」粗粒度 env-instruct 数据（train 5000 + dev 500），
# 用于训练。需先有 Kaldi 格式人声数据（见 run_build_env_instruct.sh）。
# 在 CosyVoice 仓库根目录执行: bash env_instruct_pipeline/scripts/run_room_5000.sh

export OUT_DIR="${OUT_DIR:-env_instruct_pipeline/output/env_instruct_room5000}"
export MAX_TRAIN="${MAX_TRAIN:-5000}"
export MAX_DEV="${MAX_DEV:-500}"
export INSTRUCT_MODE=coarse

bash "$(dirname "$0")/run_build_env_instruct.sh"

echo ""
echo "5000 条粗粒度数据已生成到: $OUT_DIR"
echo "训练时指定该目录，例如:"
echo "  export DATA_ROOT=$OUT_DIR"
echo "  export PRETRAINED_DIR=/media/volume/4train/pretrained_models/Fun-CosyVoice3-0.5B"
echo "  bash env_instruct_pipeline/scripts/run_train_room.sh"
echo "或使用 start_env_instruct_train.sh 前设置: export DATA_ROOT=$OUT_DIR"
