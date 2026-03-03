#!/bin/bash
# 生成「仅房间大小/混响」粗粒度数据（small / medium / large room + street + clean），
# 用于验证模型是否理解 room size / reverb instruct。不加载 DEMAND，只用 RIR。
# 在 CosyVoice 仓库根目录执行: bash env_instruct_pipeline/scripts/run_room_100.sh

export OUT_DIR="${OUT_DIR:-env_instruct_pipeline/output/env_instruct_room100}"
export MAX_TRAIN="${MAX_TRAIN:-100}"
export MAX_DEV="${MAX_DEV:-20}"
export INSTRUCT_MODE=coarse

bash "$(dirname "$0")/run_build_env_instruct.sh"

echo ""
echo "粗粒度数据已生成到: $OUT_DIR"
echo "下一步: 生成 parquet 与 data.list 后训练，例如:"
echo "  mkdir -p $OUT_DIR/train/parquet $OUT_DIR/dev/parquet"
echo "  python tools/make_parquet_list.py --num_utts_per_parquet 100 --num_processes 2 --src_dir $OUT_DIR/train --des_dir $OUT_DIR/train/parquet"
echo "  python tools/make_parquet_list.py --num_utts_per_parquet 20 --num_processes 1 --src_dir $OUT_DIR/dev --des_dir $OUT_DIR/dev/parquet"
echo "  cat $OUT_DIR/train/parquet/data.list > $OUT_DIR/train.data.list"
echo "  cat $OUT_DIR/dev/parquet/data.list   > $OUT_DIR/dev.data.list"
