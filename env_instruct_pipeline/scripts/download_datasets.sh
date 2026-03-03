#!/bin/bash
# 下载环境声音数据集（OpenSLR 28）和人声数据集（LibriTTS 子集）
# 用法: 在 CosyVoice 仓库根目录执行: bash env_instruct_pipeline/scripts/download_datasets.sh
# 或: cd env_instruct_pipeline/scripts && bash download_datasets.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="$PIPELINE_ROOT/datasets/env"
SPEECH_DIR="$PIPELINE_ROOT/datasets/speech"
mkdir -p "$ENV_DIR" "$SPEECH_DIR"

# OpenSLR 镜像（可选替换为 CN/EU）
BASE_OPENSLR="${OPENSLR_MIRROR:-https://www.openslr.org/resources}"

echo "=== 1. 环境声音数据集 (OpenSLR 28, RIR+Noise) ==="
if [ -d "$ENV_DIR/RIRS_NOISES" ]; then
  echo "已存在 $ENV_DIR/RIRS_NOISES，跳过下载"
else
  ZIP="$ENV_DIR/rirs_noises.zip"
  if [ ! -f "$ZIP" ]; then
    echo "下载 rirs_noises.zip (~1.3G)..."
    curl -L --progress-bar -o "$ZIP" "$BASE_OPENSLR/28/rirs_noises.zip"
  fi
  if [ ! -d "$ENV_DIR/RIRS_NOISES" ]; then
    echo "解压到 $ENV_DIR ..."
    unzip -o -q "$ZIP" -d "$ENV_DIR"
  fi
  echo "环境数据集就绪: $ENV_DIR/RIRS_NOISES"
fi

echo ""
echo "=== 2. 人声数据集 (LibriTTS, OpenSLR 60) ==="
# 先下载 dev-clean（约 337MB），可选 train-clean-100（约 6.3G）
for part in dev-clean; do
  if [ -d "$SPEECH_DIR/LibriTTS/$part" ]; then
    echo "已存在 $SPEECH_DIR/LibriTTS/$part，跳过"
  else
    TAR="$SPEECH_DIR/${part}.tar.gz"
  if [ ! -f "$TAR" ]; then
    echo "下载 $part (~337MB)..."
    curl -L --progress-bar -o "$TAR" "$BASE_OPENSLR/60/$part.tar.gz"
  fi
  if [ ! -d "$SPEECH_DIR/LibriTTS/$part" ]; then
    echo "解压 $part ..."
    mkdir -p "$SPEECH_DIR/LibriTTS"
    tar -xzf "$TAR" -C "$SPEECH_DIR/LibriTTS"
  fi
    echo "人声数据就绪: $SPEECH_DIR/LibriTTS/$part"
  fi
done

echo ""
echo "下载完成。"
echo "  环境: $ENV_DIR/RIRS_NOISES"
echo "  人声: $SPEECH_DIR/LibriTTS"
echo ""
echo "可选 - 带场景标签的 DEMAND 噪声（用于与 instruct 一致）:"
echo "  bash env_instruct_pipeline/scripts/download_demand.sh"
echo "  默认下载 6 个场景 (~600MB)。仅试跑可: DEMAND_SCENES=\"STRAFFIC DKITCHEN\" bash env_instruct_pipeline/scripts/download_demand.sh"
