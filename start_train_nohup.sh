#!/bin/bash
# Launch CosyVoice3 env-instruct room training with nohup.
# Produces /home/exouser/CosyVoice-env-tts/logs/train_<ts>.log and writes PID to .train_pid.
# Tail the log with:  tail -f logs/train_*.log

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export DATA_ROOT="${DATA_ROOT:-env_instruct_pipeline/output/env_instruct_room5000}"
export PRETRAINED_DIR="${PRETRAINED_DIR:-/media/volume/geo3/pretrained_models/Fun-CosyVoice3-0.5B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# skip Stage 0 by default (parquet already built); override with stage=0 to rebuild
export stage="${stage:-5}"
export stop_stage="${stop_stage:-5}"

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$REPO_ROOT/logs/train_${TS}.log"

nohup bash -c "
source /home/exouser/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice
export CUDA_HOME=\$CONDA_PREFIX
export PYTHONPATH=$REPO_ROOT:$REPO_ROOT/third_party/Matcha-TTS:\${PYTHONPATH:-}
export DATA_ROOT=$DATA_ROOT
export PRETRAINED_DIR=$PRETRAINED_DIR
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export stage=$stage
export stop_stage=$stop_stage
cd $REPO_ROOT
bash env_instruct_pipeline/scripts/run_train_room.sh
" > "$LOG" 2>&1 &

PID=$!
disown
echo "$PID" > "$REPO_ROOT/.train_pid"

echo "started training, PID=$PID"
echo "log  : $LOG"
echo "tail : tail -f $LOG"
echo "stop : kill \$(cat $REPO_ROOT/.train_pid)"
