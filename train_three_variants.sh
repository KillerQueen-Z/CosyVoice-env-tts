#!/bin/bash
# 顺序训练三个 instruct 变体:both(中英混合) → en(纯英文) → zh(纯中文)
# 每个变体:Stage 0(parquet) + Stage 5(llm/flow/hifigan 训练)
# 产物分别进 /media/volume/geo3/exp/env_instruct_room_{both,en,zh}/

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export PRETRAINED_DIR="/media/volume/geo3/pretrained_models/Fun-CosyVoice3-0.5B"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

source /home/exouser/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/third_party/Matcha-TTS:${PYTHONPATH:-}"

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"

for VARIANT in both en zh; do
  DATA_ROOT_VAR="env_instruct_pipeline/output/env_instruct_room5000_${VARIANT}"
  LOG="$REPO_ROOT/logs/train_${VARIANT}_${TS}.log"
  echo "===================================================================="
  echo "[$(date)] 训练变体: $VARIANT"
  echo "  DATA_ROOT = $DATA_ROOT_VAR"
  echo "  EXP       = /media/volume/geo3/exp/env_instruct_room_${VARIANT}"
  echo "  LOG       = $LOG"
  echo "===================================================================="

  export DATA_ROOT="$DATA_ROOT_VAR"
  export EXP_TAG="_${VARIANT}"

  # 智能跳过:如果某个子模型已有 epoch_*.pt 训练产物,就跳过
  EXP_DIR="/media/volume/geo3/exp/env_instruct_room_${VARIANT}"
  TODO_MODELS=""
  for sub in llm flow hifigan; do
    if ls "$EXP_DIR/$sub/torch_ddp"/epoch_*.pt >/dev/null 2>&1; then
      echo "  [skip] $VARIANT/$sub 已有 checkpoint,跳过"
    else
      TODO_MODELS="$TODO_MODELS $sub"
    fi
  done
  TODO_MODELS=$(echo "$TODO_MODELS" | xargs)
  if [ -z "$TODO_MODELS" ]; then
    echo "  [done] $VARIANT 全部已训完,跳过"
    continue
  fi
  echo "  待训练: $TODO_MODELS"

  # Stage 0:确保 parquet 存在(已存在则跳过)
  if [ ! -f "$DATA_ROOT_VAR/train.data.list" ]; then
    stage=0 stop_stage=0 bash env_instruct_pipeline/scripts/run_train_room.sh >>"$LOG" 2>&1
  fi

  # Stage 5:只训练待训的子模型
  MODELS="$TODO_MODELS" stage=5 stop_stage=5 bash env_instruct_pipeline/scripts/run_train_room.sh >>"$LOG" 2>&1

  echo "[$(date)] 变体 $VARIANT 训练完成"
done

echo "全部三个变体训练完成。产物:"
ls -d /media/volume/geo3/exp/env_instruct_room_* 2>/dev/null
