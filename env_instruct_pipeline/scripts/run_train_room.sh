#!/bin/bash
# 用粗粒度 room 数据（env_instruct_room100）训练 CosyVoice3，支持 instruct。
# 必须：已生成 train/dev 的 wav.scp、text、utt2spk、instruct，并准备好 CosyVoice3 预训练模型。
#
# 用法（在 CosyVoice 仓库根目录）:
#   export PRETRAINED_DIR=/path/to/Fun-CosyVoice3-0.5B   # 必填：含 llm.pt, flow.pt, hifigan.pt, campplus.onnx, speech_tokenizer_v3.onnx, CosyVoice-BlankEN/
#   bash env_instruct_pipeline/scripts/run_train_room.sh
#
# 可选:
#   DATA_ROOT=env_instruct_pipeline/output/env_instruct_room100  # 数据根目录，默认即此
#   CUDA_VISIBLE_DEVICES=0
#   stage=0  stop_stage=5  # 只做 parquet 或只做训练

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# 数据目录（与 run_room_100.sh 输出一致）
DATA_ROOT="${DATA_ROOT:-env_instruct_pipeline/output/env_instruct_room100}"
PRETRAINED_DIR="${PRETRAINED_DIR:-}"

# stage: 0=只做 parquet+data.list, 5=只做训练, 默认 0 和 5 都跑
stage="${stage:-0}"
stop_stage="${stop_stage:-5}"

if [ -z "$PRETRAINED_DIR" ] || [ ! -d "$PRETRAINED_DIR" ]; then
  echo "请设置 PRETRAINED_DIR 指向 CosyVoice3 预训练目录，例如:"
  echo "  export PRETRAINED_DIR=/path/to/Fun-CosyVoice3-0.5B"
  echo "该目录需包含: llm.pt, flow.pt, hifigan.pt, campplus.onnx, speech_tokenizer_v3.onnx, CosyVoice-BlankEN/"
  exit 1
fi

if [ ! -f "$DATA_ROOT/train/wav.scp" ] || [ ! -f "$DATA_ROOT/train/instruct" ]; then
  echo "未找到数据: $DATA_ROOT/train/wav.scp 或 instruct"
  echo "请先运行: bash env_instruct_pipeline/scripts/run_room_100.sh"
  exit 1
fi

# ---------- Stage 0: 生成 parquet 与 data.list ----------
if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
  echo "[Stage 0] 生成 parquet 与 data.list ..."
  mkdir -p "$DATA_ROOT/train/parquet" "$DATA_ROOT/dev/parquet"

  python tools/make_parquet_list.py \
    --num_utts_per_parquet 100 \
    --num_processes 2 \
    --src_dir "$DATA_ROOT/train" \
    --des_dir "$DATA_ROOT/train/parquet"

  python tools/make_parquet_list.py \
    --num_utts_per_parquet 50 \
    --num_processes 1 \
    --src_dir "$DATA_ROOT/dev" \
    --des_dir "$DATA_ROOT/dev/parquet"

  cat "$DATA_ROOT/train/parquet/data.list" > "$DATA_ROOT/train.data.list"
  cat "$DATA_ROOT/dev/parquet/data.list"   > "$DATA_ROOT/dev.data.list"
  echo "  train: $(wc -l < "$DATA_ROOT/train.data.list") 个 parquet, $(wc -l < "$DATA_ROOT/train/wav.scp") 条"
  echo "  dev:   $(wc -l < "$DATA_ROOT/dev.data.list") 个 parquet, $(wc -l < "$DATA_ROOT/dev/wav.scp") 条"
fi

# ---------- Stage 5: 训练 llm / flow / hifigan ----------
if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
  echo "[Stage 5] 训练 CosyVoice3 (llm -> flow -> hifigan) ..."
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F "," '{print NF}')
  job_id=1986
  train_engine=torch_ddp
  conf_dir="$REPO_ROOT/examples/libritts/cosyvoice3/conf"
  config="$conf_dir/cosyvoice3.yaml"

  if [ ! -f "$config" ]; then
    echo "未找到配置: $config"
    exit 1
  fi

  for model in llm flow hifigan; do
    model_dir="$REPO_ROOT/exp/env_instruct_room/$model/$train_engine"
    echo "  --- $model ---"
    torchrun --nnodes=1 --nproc_per_node="$num_gpus" \
      --rdzv_id=$job_id --rdzv_backend=c10d --rdzv_endpoint=localhost:1234 \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config "$config" \
      --train_data "$DATA_ROOT/train.data.list" \
      --cv_data "$DATA_ROOT/dev.data.list" \
      --qwen_pretrain_path "$PRETRAINED_DIR/CosyVoice-BlankEN" \
      --onnx_path "$PRETRAINED_DIR" \
      --model $model \
      --checkpoint "$PRETRAINED_DIR/$model.pt" \
      --model_dir "$model_dir" \
      --tensorboard_dir "$REPO_ROOT/tensorboard/env_instruct_room/$model/$train_engine" \
      --ddp.dist_backend nccl \
      --num_workers 2 \
      --prefetch 100 \
      --pin_memory \
      --use_amp
  done

  echo "训练完成。模型保存在: $REPO_ROOT/exp/env_instruct_room/"
fi

echo "Done."
