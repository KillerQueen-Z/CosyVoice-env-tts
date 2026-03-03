#!/bin/bash
# 下载 DEMAND 带场景标签的噪声数据集（16k，部分场景即可试跑）
# 用法: 在 CosyVoice 仓库根目录执行: bash env_instruct_pipeline/scripts/download_demand.sh
# 可选: DEMAND_SCENES="street domestic" 只下街道+家庭两类

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="$PIPELINE_ROOT/datasets/env"
DEMAND_ROOT="$ENV_DIR/DEMAND"
ZENODO="https://zenodo.org/record/1227121/files"

# DEMAND 16k 文件名与场景对应（前缀即场景标签）
# D=domestic, N=nature, O=office, P=public, S=street, T=transportation
ALL_SCENES="DKITCHEN DLIVING DWASHING NFIELD NPARK NRIVER OHALLWAY OMEETING OOFFICE PCAFETER PRESTO PSTATION SCAFE SPSQUARE STRAFFIC TBUS TCAR TMETRO"
# 默认只下 6 个场景（约 600MB），足够试跑：街道×2、家庭×2、办公室×1、自然×1
DEFAULT_SCENES="STRAFFIC SPSQUARE DKITCHEN DLIVING OOFFICE NFIELD"
SCENES="${DEMAND_SCENES:-$DEFAULT_SCENES}"

mkdir -p "$DEMAND_ROOT"
cd "$DEMAND_ROOT"

# SCAFE 在 Zenodo 上仅有 48k，无 16k；其余场景优先 16k
try_16k() { [ "$1" != "SCAFE" ]; }

for scene in $SCENES; do
  if [ -d "$scene" ] && [ -n "$(ls -A "$scene" 2>/dev/null)" ]; then
    echo "已存在 $scene，跳过"
    continue
  fi
  for rate in 16k 48k; do
    [ "$rate" = "16k" ] && ! try_16k "$scene" && continue
    ZIP="${scene}_${rate}.zip"
    if [ ! -f "$ZIP" ]; then
      echo "下载 $ZIP ..."
      curl -L -f --progress-bar -o "$ZIP" "$ZENODO/$ZIP?download=1" || { rm -f "$ZIP"; continue; }
    fi
    echo "解压 $ZIP -> $scene/"
    mkdir -p "$scene"
    if unzip -o -q -j "$ZIP" -d "$scene" "*.wav" 2>/dev/null; then
      rm -f "$ZIP"
      break
    fi
    if unzip -o -q "$ZIP" -d "_tmp_$$" 2>/dev/null; then
      find "_tmp_$$" -name "*.wav" -exec mv {} "$scene/" \;
      rm -rf "_tmp_$$"
      rm -f "$ZIP"
      break
    fi
    rm -f "$ZIP"
  done
done

echo ""
echo "DEMAND 就绪: $DEMAND_ROOT"
echo "  场景目录: $(ls -d */ 2>/dev/null | tr -d '/')"
echo "  使用方式: build_env_instruct_dataset.py --demand_dir $DEMAND_ROOT ..."
