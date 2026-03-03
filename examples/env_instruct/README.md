# 环境 + Instruct 数据集（用于 CosyVoice 微调）

每条样本 = **一段带环境/混响的人声** + **对应文本** + **一条描述该环境的 instruct**。  
训练时 (audio, text, instruct) 一一对应，方便模型学到「根据 instruct 改变输出音色/环境感」。

## 1. 用英文数据快速跑通（LibriTTS）

### 1.1 准备英文干净语音（Kaldi 格式）

在 CosyVoice 仓库根目录下，先准备 LibriTTS 的 `wav.scp` / `text` / `utt2spk`（若已有可跳过）：

```bash
cd examples/libritts/cosyvoice2
# 仅做数据准备（需先配置 data_dir / 下载 LibriTTS）
stage=0 stop_stage=0 ./run.sh
```

完成后会有 `data/train-clean-100`、`data/dev-clean` 等目录。

### 1.2 合成「带环境音 + instruct」数据

回到仓库根目录，执行：

```bash
# 使用内置合成混响/噪声（无需下载 OpenSLR 28）
bash examples/env_instruct/run_build_env_instruct.sh
```

脚本会：

- 从 `examples/libritts/cosyvoice2/data` 读取 train/dev 的 Kaldi 目录；
- 为每条 utterance 随机分配一种环境类型：`clean` / `small_room` / `medium_room` / `large_room` / `street`；
- 对该条语音做对应处理（混响和/或噪声），并写入**与该环境严格对应**的 instruct 文本；
- 输出到 `data/env_instruct/train` 和 `data/env_instruct/dev`，内含 `wav.scp`、`text`、`utt2spk`、`instruct`、`spk2utt`。

环境与 instruct 的对应关系（由 `tools/build_env_instruct_dataset.py` 固定）：

| 环境类型    | Instruct（中文） |
|------------|------------------|
| clean      | 自然、干净、无环境音。 |
| small_room  | 模拟在小型房间内说话，有轻微混响。 |
| medium_room | 模拟在中等大小房间内，混响适中。 |
| large_room  | 模拟在较大空间内说话，混响较长，类似大厅或山洞。 |
| street      | 模拟在街道旁说话，有交通与环境背景噪声。 |

### 1.3 使用 OpenSLR 28 的 RIR + 噪声（可选）

若已下载并解压 OpenSLR 28，可指定 `RIR_NOISE_DIR`，用真实 RIR 与噪声替代内置合成：

```bash
# 下载并解压（约 1.3G）
wget https://www.openslr.org/resources/28/rirs_noises.zip
unzip rirs_noises.zip -d data/

# 指定 RIR 目录后再跑
RIR_NOISE_DIR=data/RIRS_NOISES bash examples/env_instruct/run_build_env_instruct.sh
```

### 1.4 转为 CosyVoice 训练用的 parquet 与 list

```bash
OUT=data/env_instruct
mkdir -p $OUT/train/parquet $OUT/dev/parquet

python tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 4 \
  --src_dir $OUT/train --des_dir $OUT/train/parquet
python tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 4 \
  --src_dir $OUT/dev --des_dir $OUT/dev/parquet

cat $OUT/train/parquet/data.list > $OUT/train.data.list
cat $OUT/dev/parquet/data.list   > $OUT/dev.data.list
```

之后在 `cosyvoice/bin/train.py` 里使用 `--train_data data/env_instruct/train.data.list` 和 `--cv_data data/env_instruct/dev.data.list` 即可。

## 2. 自定义输入目录

若你的英文数据已在其他路径（Kaldi 格式：`wav.scp`、`text`、`utt2spk`），可直接调脚本：

```bash
python tools/build_env_instruct_dataset.py \
  --clean_train_dirs /path/to/train1 /path/to/train2 \
  --clean_dev_dirs /path/to/dev \
  --out_dir data/my_env_instruct \
  --target_sr 24000
```

或单目录按比例划分 train/dev：

```bash
python tools/build_env_instruct_dataset.py \
  --clean_data_dir /path/to/english_kaldi \
  --dev_ratio 0.1 \
  --out_dir data/my_env_instruct
```

## 3. 环境权重

默认五种环境等权。若希望多些「干净」或「街道」等，可用 `--env_weights`（顺序：clean, small_room, medium_room, large_room, street），例如：

```bash
# 更多干净、更多街道
--env_weights 2.0,1.0,1.0,1.0,1.5
```

这样即可用英文数据合成「人声 + 环境音」并得到与每条音频严格对应的 instruct，直接用于 CosyVoice 的 instruct 微调。
