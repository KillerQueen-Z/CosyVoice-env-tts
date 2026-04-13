#!/usr/bin/env bash
# setup_conda_and_data.sh
# 一键：创建 conda 环境 + 安装依赖 + 下载 CosyVoice3 预训练模型 + 生成 room-size 数据 + parquet

set -e

# ====== 配置项 ======
ENV_NAME="${ENV_NAME:-cosyvoice}"
PY_VERSION="${PY_VERSION:-3.10}"

PRETRAIN_DIR="${PRETRAIN_DIR:-pretrained_models/Fun-CosyVoice3-0.5B}"
BACKEND="${BACKEND:-modelscope}"          # modelscope 或 huggingface

DATA_ROOT="${DATA_ROOT:-env_instruct_pipeline/output/env_instruct_room100}"

# ====== 1. 创建并激活 conda 环境 ======
echo ">>> Step 1: Create and activate conda env '${ENV_NAME}' (python=${PY_VERSION})"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Please make sure Miniconda/Anaconda is installed and 'conda' is in PATH." >&2
  exit 1
fi

conda env list | grep -q "^${ENV_NAME} " || conda create -n "${ENV_NAME}" python="${PY_VERSION}" -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ====== 2. 安装依赖 ======
echo ">>> Step 2: Install Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

# ====== 3. 下载 CosyVoice3 预训练模型 ======
echo ">>> Step 3: Download Fun-CosyVoice3-0.5B pretrained model (backend=${BACKEND})"

if [ "${BACKEND}" = "modelscope" ]; then
  python env_instruct_pipeline/scripts/download_pretrained_cosyvoice3.py \
    --backend modelscope --out_dir "${PRETRAIN_DIR}"
elif [ "${BACKEND}" = "huggingface" ]; then
  python env_instruct_pipeline/scripts/download_pretrained_cosyvoice3.py \
    --backend huggingface --out_dir "${PRETRAIN_DIR}"
else
  echo "ERROR: BACKEND must be 'modelscope' or 'huggingface' (got '${BACKEND}')." >&2
  exit 1
fi

echo ">>> Pretrained model downloaded to: ${PRETRAIN_DIR}"

# ====== 4. 生成粗粒度 room-size 数据 ======
echo ">>> Step 4: Generate coarse room-size env-instruct data (env_instruct_room100)"
export OUT_DIR="${DATA_ROOT}"
bash env_instruct_pipeline/scripts/run_room_100.sh

echo ">>> Room-size data generated under: ${DATA_ROOT}"

# ====== 5. 生成 parquet + *.data.list ======
echo ">>> Step 5: Make parquet and data.list"

mkdir -p "${DATA_ROOT}/train/parquet" "${DATA_ROOT}/dev/parquet"

python tools/make_parquet_list.py \
  --num_utts_per_parquet 100 --num_processes 2 \
  --src_dir "${DATA_ROOT}/train" --des_dir "${DATA_ROOT}/train/parquet"

python tools/make_parquet_list.py \
  --num_utts_per_parquet 50 --num_processes 1 \
  --src_dir "${DATA_ROOT}/dev" --des_dir "${DATA_ROOT}/dev/parquet"

cat "${DATA_ROOT}/train/parquet/data.list" > "${DATA_ROOT}/train.data.list"
cat "${DATA_ROOT}/dev/parquet/data.list"   > "${DATA_ROOT}/dev.data.list"

echo ">>> Parquet and data.list ready:"
echo "    ${DATA_ROOT}/train.data.list"
echo "    ${DATA_ROOT}/dev.data.list"

echo ""
echo "All done."
echo "Conda env  : ${ENV_NAME}"
echo "Pretrain dir: ${PRETRAIN_DIR}"
echo "Data root   : ${DATA_ROOT}"
echo ""
echo "Next: to start training, run:"
echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\""
echo "  conda activate ${ENV_NAME}"
echo "  export PRETRAINED_DIR=${PRETRAIN_DIR}"
echo "  bash env_instruct_pipeline/scripts/run_train_room.sh"