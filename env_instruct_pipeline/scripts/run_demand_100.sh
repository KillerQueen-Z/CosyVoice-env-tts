#!/bin/bash
# 用 DEMAND 场景噪声合成 100 条「背景+人声」试跑数据
# 1) 若未下载 DEMAND，先下载 4 个场景（约 400MB）
# 2) 清空或使用独立输出目录，合成 train=100 条、dev=20 条
# 在 CosyVoice 仓库根目录执行:
#   conda activate tts   # 需含 torch, torchaudio, scipy, soundfile
#   bash env_instruct_pipeline/scripts/run_demand_100.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PIPELINE="$REPO_ROOT/env_instruct_pipeline"
ENV_ROOT="$PIPELINE/datasets/env"
DEMAND_DIR="$ENV_ROOT/DEMAND"
OUT_DIR="$PIPELINE/output/env_instruct_demand100"

# Step 0: 确保 DEMAND 已下载（4 个场景：街道×2 + 家庭×2，约 400MB）
if [ ! -d "$DEMAND_DIR" ] || [ -z "$(ls -A "$DEMAND_DIR" 2>/dev/null)" ]; then
  echo "[Step 0] 下载 DEMAND（4 个场景: STRAFFIC SPSQUARE DKITCHEN DLIVING）..."
  DEMAND_SCENES="STRAFFIC SPSQUARE DKITCHEN DLIVING" bash "$PIPELINE/scripts/download_demand.sh"
else
  echo "[Step 0] 已存在 DEMAND: $DEMAND_DIR"
fi

# Step 1: 删除旧试跑输出，保证用 DEMAND 重新合成
rm -rf "$OUT_DIR/train" "$OUT_DIR/dev"
mkdir -p "$OUT_DIR"

# Step 2: 合成 100 条 train + 20 条 dev（使用 DEMAND + 若有 RIR 则用 RIR）
echo "[Step 2] 合成 100 条 train / 20 条 dev（DEMAND 场景噪声 + RIR）..."
OUT_DIR="$OUT_DIR" MAX_TRAIN=100 MAX_DEV=20 bash "$PIPELINE/scripts/run_build_env_instruct.sh"

# Step 3: 生成 parquet 与 data.list（100 条较少，单 parquet 即可）
echo "[Step 3] 生成 parquet 与 data.list ..."
mkdir -p "$OUT_DIR/train/parquet" "$OUT_DIR/dev/parquet"
python tools/make_parquet_list.py --num_utts_per_parquet 200 --num_processes 1 \
  --src_dir "$OUT_DIR/train" --des_dir "$OUT_DIR/train/parquet"
python tools/make_parquet_list.py --num_utts_per_parquet 200 --num_processes 1 \
  --src_dir "$OUT_DIR/dev" --des_dir "$OUT_DIR/dev/parquet"
cat "$OUT_DIR/train/parquet/data.list" > "$OUT_DIR/train.data.list"
cat "$OUT_DIR/dev/parquet/data.list"   > "$OUT_DIR/dev.data.list"

echo ""
echo "完成。DEMAND 试跑 100 条数据:"
echo "  train: $OUT_DIR/train.data.list ($(wc -l < "$OUT_DIR/train.data.list") 条)"
echo "  dev:   $OUT_DIR/dev.data.list ($(wc -l < "$OUT_DIR/dev.data.list") 条)"
echo "  试听: $OUT_DIR/train/wavs/ 与 $OUT_DIR/train/instruct 对应"
