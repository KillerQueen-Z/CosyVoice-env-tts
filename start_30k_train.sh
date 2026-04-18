#!/bin/bash
# 30k 数据全 3 阶段训练启动器(llm + flow + hifigan)
# 数据必须已就绪:env_instruct_pipeline/output/env_instruct_room30k_both/
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_DIR=env_instruct_pipeline/output/env_instruct_room30k_both
if [ ! -f "$DATA_DIR/train/instruct" ]; then
  echo "[err] $DATA_DIR/train/instruct 不存在,先等 30k 生成完成"
  exit 1
fi

# 清旧 checkpoint(新数据分布不兼容)
rm -rf /media/volume/geo3/exp/env_instruct_room_both
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/train_30k_${TS}.log"

nohup bash -c "
source /home/exouser/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice
export CUDA_HOME=\$CONDA_PREFIX
export PYTHONPATH=$PWD:$PWD/third_party/Matcha-TTS:\${PYTHONPATH:-}
export DATA_ROOT=$DATA_DIR
export PRETRAINED_DIR=/media/volume/geo3/pretrained_models/Fun-CosyVoice3-0.5B
export EXP_TAG=_both
export MODELS='llm flow hifigan'
stage=0 stop_stage=0 bash env_instruct_pipeline/scripts/run_train_room.sh
stage=5 stop_stage=5 bash env_instruct_pipeline/scripts/run_train_room.sh
" > "$LOG" 2>&1 &
PID=$!
disown
echo "$PID" > .train_pid
echo "[OK] PID=$PID  log: $LOG"
