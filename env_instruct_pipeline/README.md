# 环境 + Instruct 数据流水线

用于 CosyVoice「人声 + 环境音」instruct 微调：下载数据集、生成 Kaldi 格式、再转为 parquet 训练数据。

## 目录说明

```
env_instruct_pipeline/
├── datasets/           # 原始/下载的数据集
│   ├── env/            # 环境声音：OpenSLR 28 (RIRS_NOISES)
│   └── speech/         # 人声数据：LibriTTS (如 dev-clean)
├── scripts/            # 下载与处理脚本
│   ├── download_datasets.sh   # 下载 env + speech
│   ├── prepare_kaldi_libritts.py  # LibriTTS → wav.scp/text/utt2spk
│   └── run_build_env_instruct.sh  # 合成带环境的 train/dev + instruct
└── README.md
```

## 使用步骤

### 1. 下载数据集

在 **CosyVoice 仓库根目录** 执行：

```bash
bash env_instruct_pipeline/scripts/download_datasets.sh
```

- **环境数据**：会下载 OpenSLR 28 的 `rirs_noises.zip`（约 1.3G）到 `datasets/env/` 并解压出 `RIRS_NOISES/`。
- **人声数据**：会下载 LibriTTS 的 `dev-clean`（约 337MB）到 `datasets/speech/LibriTTS/`。

（若需 train 子集，可编辑 `download_datasets.sh` 中的 `for part in dev-clean` 增加 `train-clean-100` 等。）

### 2. 准备人声的 Kaldi 格式

LibriTTS 解压后是「说话人/章节/音频+文本」目录结构，需转成 `wav.scp`、`text`、`utt2spk`：

```bash
python env_instruct_pipeline/scripts/prepare_kaldi_libritts.py \
  --src_dir env_instruct_pipeline/datasets/speech/LibriTTS \
  --des_dir env_instruct_pipeline/datasets/speech/kaldi
```

会在 `datasets/speech/kaldi/` 下按子集生成 `dev-clean/` 等，内含 `wav.scp`、`text`、`utt2spk`。

### 3. 合成「带环境 + instruct」数据

```bash
bash env_instruct_pipeline/scripts/run_build_env_instruct.sh
```

脚本会读取 `datasets/speech/kaldi/` 的 train/dev，使用 `datasets/env/RIRS_NOISES` 做混响/加噪，并写入与每条音频对应的 instruct，输出到 `env_instruct_pipeline/output/env_instruct/train` 与 `dev`（或脚本内配置的路径）。

### 4. 转为 CosyVoice 训练用 parquet

在仓库根目录：

```bash
OUT=env_instruct_pipeline/output/env_instruct
mkdir -p $OUT/train/parquet $OUT/dev/parquet
python tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 4 \
  --src_dir $OUT/train --des_dir $OUT/train/parquet
python tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 4 \
  --src_dir $OUT/dev --des_dir $OUT/dev/parquet
cat $OUT/train/parquet/data.list > $OUT/train.data.list
cat $OUT/dev/parquet/data.list   > $OUT/dev.data.list
```

之后在 `cosyvoice/bin/train.py` 里用 `--train_data` / `--cv_data` 指向上述 `train.data.list` 和 `dev.data.list` 即可。

## 数据集说明

| 目录 | 内容 | 来源 |
|------|------|------|
| `datasets/env/` | 房间冲激响应(RIR) + 噪声 | [OpenSLR 28](https://www.openslr.org/28/) |
| `datasets/speech/` | 英文朗读语音 | [OpenSLR 60 LibriTTS](https://www.openslr.org/60/) |
