#!/usr/bin/env bash
# 一键初始化：检查命令行工具 → pip 依赖 → 下载 RIR/LibriTTS → Kaldi 列表 → CosyVoice3 预训练
#
# 用法（在仓库根目录）:
#   bash quick_init.sh
#
# 环境变量（均为可选）:
#   PRETRAINED_BACKEND=modelscope   # 或 huggingface（海外）
#   PRETRAINED_DIR=pretrained_models/Fun-CosyVoice3-0.5B
#   SKIP_PIP=1           跳过 pip install
#   SKIP_DATA=1          跳过 OpenSLR + LibriTTS 下载
#   SKIP_KALDI=1         跳过 prepare_kaldi_libritts
#   SKIP_PRETRAINED=1    跳过预训练模型下载
#   PIP_EXTRA_ARGS="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
#
# 说明：不自动创建 conda；请先: conda create -n cosyvoice python=3.10 -y && conda activate cosyvoice

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PIPELINE="$REPO_ROOT/env_instruct_pipeline"
PRETRAINED_BACKEND="${PRETRAINED_BACKEND:-modelscope}"
PRETRAINED_DIR="${PRETRAINED_DIR:-$REPO_ROOT/pretrained_models/Fun-CosyVoice3-0.5B}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"

log() { echo ">>> [quick_init] $*"; }

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "缺少命令: $1  请先安装后再运行本脚本。"
    exit 1
  fi
}

log "仓库根目录: $REPO_ROOT"

need_cmd curl
need_cmd unzip
need_cmd tar
need_cmd python

if ! python -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)"; then
  echo "建议使用 Python >= 3.10，当前: $(python -V)"
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ "${SKIP_PIP:-0}" != "1" ]]; then
  log "安装 Python 依赖 (requirements.txt) ..."
  # shellcheck disable=SC2086
  python -m pip install -U pip
  # shellcheck disable=SC2086
  python -m pip install $PIP_EXTRA_ARGS -r "$REPO_ROOT/requirements.txt"
else
  log "SKIP_PIP=1，跳过 pip install"
fi

if [[ "${SKIP_DATA:-0}" != "1" ]]; then
  log "下载环境声 + LibriTTS (download_datasets.sh) ..."
  bash "$PIPELINE/scripts/download_datasets.sh"
else
  log "SKIP_DATA=1，跳过数据集下载"
fi

resolve_libritts_src() {
  local base="$PIPELINE/datasets/speech/LibriTTS"
  if [[ -d "$base/dev-clean" ]]; then
    echo "$base"
  elif [[ -d "$base/LibriTTS/dev-clean" ]]; then
    echo "$base/LibriTTS"
  else
    echo ""
  fi
}

if [[ "${SKIP_KALDI:-0}" != "1" ]]; then
  LIB_SRC="$(resolve_libritts_src)"
  if [[ -z "$LIB_SRC" ]]; then
    echo "未找到 LibriTTS/dev-clean，请检查下载是否成功: $PIPELINE/datasets/speech/LibriTTS"
    exit 1
  fi
  log "准备 Kaldi 格式人声 (src=$LIB_SRC) ..."
  python "$PIPELINE/scripts/prepare_kaldi_libritts.py" \
    --src_dir "$LIB_SRC" \
    --des_dir "$PIPELINE/datasets/speech/kaldi"
else
  log "SKIP_KALDI=1，跳过 Kaldi 准备"
fi

if [[ "${SKIP_PRETRAINED:-0}" != "1" ]]; then
  log "下载 CosyVoice3 预训练 (backend=$PRETRAINED_BACKEND) -> $PRETRAINED_DIR ..."
  python "$PIPELINE/scripts/download_pretrained_cosyvoice3.py" \
    --backend "$PRETRAINED_BACKEND" \
    --out_dir "$PRETRAINED_DIR"
else
  log "SKIP_PRETRAINED=1，跳过预训练下载"
fi

log "初始化完成。"
echo ""
echo "后续可执行:"
echo "  export PRETRAINED_DIR=$PRETRAINED_DIR"
echo "  export PYTHONPATH=$REPO_ROOT:\$PYTHONPATH"
echo "  bash env_instruct_pipeline/scripts/run_room_100.sh   # 或 run_room_5000.sh"
echo "  bash env_instruct_pipeline/scripts/run_train_room.sh"
echo ""
echo "若仅用 dev-clean 试跑、数据在嵌套 LibriTTS/ 下，本脚本已自动识别。"
