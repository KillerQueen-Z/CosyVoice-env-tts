#!/bin/bash
# 一键完成：LibriTTS → Kaldi → 环境+instruct 合成 → parquet → JSON 预览
# 用法（先激活 tts 环境）:
#   conda activate tts
#   cd /Users/andy/Desktop/vscode/CosyVoice
#   bash env_instruct_pipeline/scripts/run_full_env_instruct_pipeline.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# 确保 parquet 步骤所需依赖存在
python -c "import tqdm" 2>/dev/null || pip install -q tqdm
python -c "import pandas" 2>/dev/null || pip install -q pandas
python -c "import pyarrow" 2>/dev/null || pip install -q pyarrow

PIPELINE="$REPO_ROOT/env_instruct_pipeline"
DATASETS="$PIPELINE/datasets"
SPEECH_LIBRITTS="$DATASETS/speech/LibriTTS/LibriTTS"
KALDI_ROOT="$DATASETS/speech/kaldi"
ENV_DIR="$DATASETS/env/RIRS_NOISES"
OUT_DIR="${OUT_DIR:-$PIPELINE/output/env_instruct}"
MAX_TRAIN="${MAX_TRAIN:--1}"
MAX_DEV="${MAX_DEV:-200}"
TARGET_SR="${TARGET_SR:-24000}"

echo "[Step 0] 检查数据集路径"
if [ ! -d "$ENV_DIR" ]; then
  echo "  环境数据不存在: $ENV_DIR"
  echo "  请先运行: bash env_instruct_pipeline/scripts/download_datasets.sh"
  exit 1
fi
if [ ! -d "$SPEECH_LIBRITTS" ]; then
  echo "  人声数据不存在: $SPEECH_LIBRITTS"
  echo "  请确认 dev-clean 已解压到 LibriTTS/LibriTTS/ 下."
  exit 1
fi

echo "[Step 1] LibriTTS → Kaldi (wav.scp/text/utt2spk)"
if [ -f "$KALDI_ROOT/dev-clean/wav.scp" ]; then
  echo "  已存在 Kaldi dev-clean，跳过转换。"
else
  mkdir -p "$KALDI_ROOT"
  python env_instruct_pipeline/scripts/prepare_kaldi_libritts.py \
    --src_dir "$SPEECH_LIBRITTS" \
    --des_dir "$KALDI_ROOT"
fi

echo "[Step 2] 合成环境 + instruct 数据 (train/dev)"
if [ -f "$OUT_DIR/train/wav.scp" ] && [ -s "$OUT_DIR/train/wav.scp" ] && [ -f "$OUT_DIR/dev/wav.scp" ] && [ -s "$OUT_DIR/dev/wav.scp" ]; then
  echo "  已存在 train/dev 合成数据，跳过。"
else
  export OUT_DIR MAX_TRAIN MAX_DEV TARGET_SR
  bash env_instruct_pipeline/scripts/run_build_env_instruct.sh
fi

echo "[Step 3] 生成 parquet 与 *.data.list"
OUT="$OUT_DIR"
mkdir -p "$OUT/train/parquet" "$OUT/dev/parquet"

if [ -f "$OUT/train.data.list" ] && [ -f "$OUT/dev.data.list" ] && [ -s "$OUT/train.data.list" ] && [ -s "$OUT/dev.data.list" ]; then
  echo "  已存在 parquet 与 data.list，跳过。"
else
  python tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 4 \
    --src_dir "$OUT/train" --des_dir "$OUT/train/parquet"

  python tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 4 \
    --src_dir "$OUT/dev" --des_dir "$OUT/dev/parquet"

  cat "$OUT/train/parquet/data.list" > "$OUT/train.data.list"
  cat "$OUT/dev/parquet/data.list"   > "$OUT/dev.data.list"
fi

echo "[Step 4] 导出 JSON 预览 (train 前 50 条)"
python env_instruct_pipeline/scripts/export_train_preview.py "$OUT"

echo ""
echo "全部完成。训练数据入口:"
echo "  train: $OUT/train.data.list"
echo "  dev  : $OUT/dev.data.list"
echo "JSON 预览: $OUT/train/train_preview.jsonl"

