#!/bin/bash
# Split the Kaldi clean speech dir into train/dev and generate env-instruct data
# (convolving with RIRS_NOISES room-bucketed RIRs) for each split.
#
# Env vars:
#   OUT_DIR        : output root (will create $OUT_DIR/{train,dev})
#   MAX_TRAIN=100  : cap train utts (-1 = all)
#   MAX_DEV=20     : cap dev utts
#   TARGET_SR=24000
#   KALDI_SRC      : source Kaldi dir (default env_instruct_pipeline/datasets/speech/kaldi/dev-clean)
#   RIR_DIR        : RIRS_NOISES root (default env_instruct_pipeline/datasets/env/RIRS_NOISES)
#   DEV_RATIO=0.1  : fraction used for dev split if MAX_* not given
#   SEED=42

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-env_instruct_pipeline/output/env_instruct_room100}"
MAX_TRAIN="${MAX_TRAIN:-100}"
MAX_DEV="${MAX_DEV:-20}"
TARGET_SR="${TARGET_SR:-24000}"
KALDI_SRC="${KALDI_SRC:-env_instruct_pipeline/datasets/speech/kaldi/dev-clean}"
RIR_DIR="${RIR_DIR:-env_instruct_pipeline/datasets/env/RIRS_NOISES}"
SEED="${SEED:-42}"

if [ ! -f "$KALDI_SRC/wav.scp" ]; then
  echo "Source Kaldi dir missing: $KALDI_SRC"
  exit 1
fi
if [ ! -d "$RIR_DIR" ]; then
  echo "RIR dir missing: $RIR_DIR"
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Shuffle wav.scp deterministically and split into train/dev.
python - "$KALDI_SRC" "$TMP_DIR" "$MAX_TRAIN" "$MAX_DEV" "$SEED" <<'PY'
import random, sys, shutil
from pathlib import Path
src, tmp, max_train, max_dev, seed = sys.argv[1:]
src, tmp = Path(src), Path(tmp)
max_train, max_dev, seed = int(max_train), int(max_dev), int(seed)
def read(p):
    rows = []
    with p.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line: continue
            k, _, v = line.partition(" ")
            rows.append((k, v))
    return rows
scp = read(src / "wav.scp")
text = dict(read(src / "text"))
u2s = dict(read(src / "utt2spk"))
rng = random.Random(seed); rng.shuffle(scp)
nt = len(scp) if max_train < 0 else min(max_train, len(scp))
nd = max(0, (len(scp) - nt)) if max_dev < 0 else min(max_dev, len(scp) - nt)
splits = {"train": scp[:nt], "dev": scp[nt:nt+nd]}
for name, rows in splits.items():
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "wav.scp").open("w") as fw, (d / "text").open("w") as ft, (d / "utt2spk").open("w") as fu:
        for utt, wp in rows:
            fw.write(f"{utt} {wp}\n")
            ft.write(f"{utt} {text.get(utt,'')}\n")
            fu.write(f"{utt} {u2s.get(utt,'spk')}\n")
    print(f"{name}: {len(rows)} utts -> {d}")
PY

MANIFEST_ARG=""
if [ -n "${RIR_MANIFEST:-}" ] && [ -f "${RIR_MANIFEST}" ]; then
  MANIFEST_ARG="--rir_manifest $RIR_MANIFEST"
  echo "[build_env_instruct] 使用 RT60 过滤 manifest: $RIR_MANIFEST"
fi
for split in train dev; do
  SRC_SPLIT="$TMP_DIR/$split"
  OUT_SPLIT="$OUT_DIR/$split"
  mkdir -p "$OUT_SPLIT"
  echo "[build_env_instruct] $split -> $OUT_SPLIT"
  python tools/build_env_instruct_dataset.py \
    --src_dir "$SRC_SPLIT" \
    --rir_dir "$RIR_DIR" \
    $MANIFEST_ARG \
    --out_dir "$OUT_SPLIT" \
    --target_sr "$TARGET_SR" \
    --seed "$SEED"
done

echo "Done. Output at: $OUT_DIR"
