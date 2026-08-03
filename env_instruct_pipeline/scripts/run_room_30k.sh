#!/bin/bash
# 从 train-clean-100 生成 15000 base × 2 (zh/en) = 30000 条 env-instruct 训练数据
# 依赖:
#   - LibriTTS train-clean-100 已下载解压到 $LIBRITTS_ROOT
#   - RIRS_NOISES 已存在于 $RIR_DIR
#
# 输出:
#   env_instruct_pipeline/output/env_instruct_room30k_both/

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

COSYVOICE_STORAGE_ROOT="${COSYVOICE_STORAGE_ROOT:-$REPO_ROOT/storage}"
LIBRITTS_ROOT="${LIBRITTS_ROOT:-$COSYVOICE_STORAGE_ROOT/speech_data/LibriTTS}"
KALDI_ROOT="${KALDI_ROOT:-$COSYVOICE_STORAGE_ROOT/speech_data/kaldi}"
OUT_DIR="${OUT_DIR:-$COSYVOICE_STORAGE_ROOT/env_instruct_output/env_instruct_room30k_both}"
RIR_DIR="${RIR_DIR:-$COSYVOICE_STORAGE_ROOT/datasets/env/RIRS_NOISES}"
RIR_MANIFEST="${RIR_MANIFEST:-$COSYVOICE_STORAGE_ROOT/rt60_manifest.json}"
MAX_TRAIN="${MAX_TRAIN:-15000}"
MAX_DEV="${MAX_DEV:-1500}"
TARGET_SR="${TARGET_SR:-24000}"
SEED="${SEED:-42}"

if [ ! -d "$LIBRITTS_ROOT/train-clean-100" ]; then
  echo "[err] $LIBRITTS_ROOT/train-clean-100 不存在。先解压 train-clean-100.tar.gz"
  exit 1
fi
if [ ! -d "$RIR_DIR" ]; then
  echo "[err] $RIR_DIR 不存在"
  exit 1
fi

# Step 1: LibriTTS → Kaldi 格式
if [ ! -f "$KALDI_ROOT/train-clean-100/wav.scp" ]; then
  echo "[step1] LibriTTS → Kaldi"
  mkdir -p "$KALDI_ROOT"
  python env_instruct_pipeline/scripts/prepare_kaldi_libritts.py \
    --src_dir "$LIBRITTS_ROOT" \
    --des_dir "$KALDI_ROOT"
else
  echo "[skip step1] $KALDI_ROOT/train-clean-100 已存在"
fi

# Step 2: 从 Kaldi 切分 + 卷积 RIR 生成 env-instruct 数据
if [ ! -f "$OUT_DIR/train/instruct" ]; then
  echo "[step2] 切分 train/dev + 卷积 RIR"
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT

  python - "$KALDI_ROOT/train-clean-100" "$TMP_DIR" "$MAX_TRAIN" "$MAX_DEV" "$SEED" <<'PY'
import random, sys
from pathlib import Path
src, tmp, max_train, max_dev, seed = sys.argv[1:]
src, tmp = Path(src), Path(tmp)
max_train, max_dev, seed = int(max_train), int(max_dev), int(seed)
def read(p):
    rows = []
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        k, _, v = line.partition(" ")
        rows.append((k, v))
    return rows
scp = read(src / "wav.scp")
text = dict(read(src / "text"))
u2s = dict(read(src / "utt2spk"))
rng = random.Random(seed); rng.shuffle(scp)
nt = min(max_train, len(scp))
nd = min(max_dev, len(scp) - nt)
for name, rows in [("train", scp[:nt]), ("dev", scp[nt:nt+nd])]:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    with (d/"wav.scp").open("w") as fw, (d/"text").open("w") as ft, (d/"utt2spk").open("w") as fu:
        for utt, wp in rows:
            fw.write(f"{utt} {wp}\n")
            ft.write(f"{utt} {text.get(utt,'')}\n")
            fu.write(f"{utt} {u2s.get(utt,'spk')}\n")
    print(f"{name}: {len(rows)} utts")
PY

  for split in train dev; do
    SRC_SPLIT="$TMP_DIR/$split"
    OUT_SPLIT="$OUT_DIR/$split"
    mkdir -p "$OUT_SPLIT"
    echo "  [build] $split → $OUT_SPLIT"
    python tools/build_env_instruct_dataset.py \
      --src_dir "$SRC_SPLIT" \
      --rir_dir "$RIR_DIR" \
      --rir_manifest "$RIR_MANIFEST" \
      --out_dir "$OUT_SPLIT" \
      --target_sr "$TARGET_SR" \
      --seed "$SEED" \
      --instruct_lang zh
  done
else
  echo "[skip step2] $OUT_DIR/train/instruct 已存在"
fi

# Step 3: canonical 包装 + 中英文翻倍
echo "[step3] canonical 翻倍 → 30000 条"
python tools/rewrite_instruct_bilingual.py \
  --data_root "$OUT_DIR" \
  --duplicate  # 默认加 canonical 前缀

echo ""
echo "=== 完成 ==="
echo "训练集 $(wc -l < "$OUT_DIR/train/instruct") 条"
echo "验证集 $(wc -l < "$OUT_DIR/dev/instruct") 条"
echo ""
head -3 "$OUT_DIR/train/instruct"
