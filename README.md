# SoniSphere-LoRA（基于 CosyVoice 的环境感知 TTS）

本仓库是在 **CosyVoice3-0.5B** 上的课题向 fork，研究 **指令驱动的环境语音重建**：给定如「小房间」「大厅」等自然语言指令，合成带对应混响特征的语音（与「去混响、做干声」的传统 TTS 目标不同）。

**核心**：CosyVoice3-0.5B（LLM + flow + vocoder）→ **LoRA 微调** → 监督来自 **env-instruct**（干声 + RIR + 文本指令），并计划扩展真实场景语料与自蒸馏。

上游 CosyVoice 的完整安装、演示、评测与引用说明保留在 **`README_raw.md`**。

---

## 大盘清理后的完整重建

训练数据、预训练模型和逐 epoch checkpoint 不属于 Git 仓库。释放旧训练盘后，保留仓库代码、实验文档、少量评测输出，以及单独归档的最终模型；其余大文件按本节重新下载或生成。

本次 Jetstream 清理保留的最终 Phase 5 文件位于 `/media/volume/geo2/tts_final_artifacts/`：`neural_reverb.pt`、`reverb_classifier.pt`、`rt60_manifest.json` 及 `SHA256SUMS`。恢复前可在该目录运行 `sha256sum -c SHA256SUMS` 校验。

### 哪些内容可以删除后重建

| 内容 | 是否需要备份 | 重建方式 |
|------|--------------|----------|
| `pretrained_models/Fun-CosyVoice3-0.5B` | 否 | 由 ModelScope 或 Hugging Face 重新下载 |
| LibriTTS `train-clean-100` | 否 | 从 OpenSLR 60 重新下载并解压 |
| `RIRS_NOISES` | 否 | 从 OpenSLR 28 重新下载并解压 |
| Kaldi 列表、RT60 manifest | 否 | 使用仓库脚本重新生成 |
| `env_instruct_output` | 否 | 运行 `run_room_30k.sh` 重新生成 |
| `exp/**/epoch_*_whole.pt` | 仅备份最终选定模型 | 历史 epoch 不长期保存，需要时重新训练 |

### 1. 指定新的训练盘

不要在脚本里写死旧盘符。为新挂载盘指定一个统一根目录：

```bash
cd /path/to/CosyVoice-env-tts
export COSYVOICE_STORAGE_ROOT=/path/to/new-volume/cosyvoice
mkdir -p "$COSYVOICE_STORAGE_ROOT"/{pretrained_models,speech_data,datasets/env,env_instruct_output,exp}
```

完整重建和训练建议预留至少 **150 GiB**；若只保留最终 checkpoint，训练结束后可以删除中间 epoch。

### 2. 重建 Python 环境和 base model

```bash
conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice
pip install -r requirements.txt

export PRETRAINED_DIR="$COSYVOICE_STORAGE_ROOT/pretrained_models/Fun-CosyVoice3-0.5B"
python env_instruct_pipeline/scripts/download_pretrained_cosyvoice3.py \
  --backend modelscope \
  --out_dir "$PRETRAINED_DIR"
```

海外机器可将 `--backend modelscope` 改为 `--backend huggingface`。模型来源为 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`。

### 3. 重新下载训练数据

```bash
# OpenSLR 28: RIR + Noise
curl -L -o "$COSYVOICE_STORAGE_ROOT/datasets/env/rirs_noises.zip" \
  https://www.openslr.org/resources/28/rirs_noises.zip
unzip -q "$COSYVOICE_STORAGE_ROOT/datasets/env/rirs_noises.zip" \
  -d "$COSYVOICE_STORAGE_ROOT/datasets/env"

# OpenSLR 60: LibriTTS train-clean-100
curl -L -o "$COSYVOICE_STORAGE_ROOT/speech_data/train-clean-100.tar.gz" \
  https://www.openslr.org/resources/60/train-clean-100.tar.gz
mkdir -p "$COSYVOICE_STORAGE_ROOT/speech_data/LibriTTS"
tar -xzf "$COSYVOICE_STORAGE_ROOT/speech_data/train-clean-100.tar.gz" \
  -C "$COSYVOICE_STORAGE_ROOT/speech_data/LibriTTS"
```

生成 Kaldi 索引：

```bash
python env_instruct_pipeline/scripts/prepare_kaldi_libritts.py \
  --src_dir "$COSYVOICE_STORAGE_ROOT/speech_data/LibriTTS" \
  --des_dir "$COSYVOICE_STORAGE_ROOT/speech_data/kaldi"
```

### 4. 重新生成 RT60 manifest 和 30k env-instruct 数据

```bash
export LIBRITTS_ROOT="$COSYVOICE_STORAGE_ROOT/speech_data/LibriTTS"
export KALDI_ROOT="$COSYVOICE_STORAGE_ROOT/speech_data/kaldi"
export RIR_DIR="$COSYVOICE_STORAGE_ROOT/datasets/env/RIRS_NOISES"
export RIR_MANIFEST="$COSYVOICE_STORAGE_ROOT/rt60_manifest.json"
export OUT_DIR="$COSYVOICE_STORAGE_ROOT/env_instruct_output/env_instruct_room30k_both"

SPEECH_SAMPLE=$(find "$LIBRITTS_ROOT/train-clean-100" -type f -name '*.wav' | head -n 1)
python tools/rt60_filter_rirs.py \
  --rir_dir "$RIR_DIR" \
  --speech_sample "$SPEECH_SAMPLE" \
  --out_dir rt60_demo_out \
  --save_manifest "$RIR_MANIFEST"

bash env_instruct_pipeline/scripts/run_room_30k.sh
```

### 5. 重新训练

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export DATA_ROOT="$OUT_DIR"
export EXP_ROOT="$COSYVOICE_STORAGE_ROOT/exp/env_instruct_room_both"
export EXP_TAG=_both
export PRETRAINED_DIR="$COSYVOICE_STORAGE_ROOT/pretrained_models/Fun-CosyVoice3-0.5B"

bash env_instruct_pipeline/scripts/run_train_room.sh
```

训练结束后先完成评测并记录最终选定的 LLM/Flow checkpoint，再清理其余 `epoch_*_whole.pt`。不要把大模型或数据提交到 GitHub。

---

## 仓库结构（与本课题相关）

| 路径 | 说明 |
|------|------|
| `quick_init.sh` | **新环境一键初始化**：依赖安装、数据下载、Kaldi 列表、CosyVoice3 预训练下载 |
| `env_instruct_pipeline/` | 数据流水线：下载、合成 env-instruct、导出 parquet |
| `tools/build_env_instruct_dataset.py` | 带环境与 instruct 的数据合成 |
| `env_instruct_pipeline/scripts/download_datasets.sh` | OpenSLR28（RIR）+ LibriTTS |
| `env_instruct_pipeline/scripts/run_room_100.sh` | 小规模试跑（约 100 train / 20 dev） |
| `env_instruct_pipeline/scripts/run_room_5000.sh` | 较大训练集（默认 5000 train / 500 dev） |
| `env_instruct_pipeline/scripts/run_train_room.sh` | **GPU 训练入口**（可含 parquet 生成 + llm→flow→hifigan） |
| `env_instruct_pipeline/docs/DEPLOY_CLOUD.md` | 云上 Linux + NVIDIA 从零部署 |
| `report.tex` | 课题报告（论文体例） |

> **`env_instruct_pipeline/datasets/`**、**`env_instruct_pipeline/output/`**、**`pretrained_models/`**、**`exp/`** 等体积大，已 **gitignore**，换机器需重新下载或自行拷贝。

---

## 新环境快速启动

以下均在 **仓库根目录** 执行。训练需 **Linux + NVIDIA GPU**；仅做数据准备可在 macOS 上完成大部分步骤（合成较慢）。

### 0. 系统与工具

- **Python ≥ 3.10**（推荐 Conda）
- 训练机安装较新的 **NVIDIA 驱动**（与 `requirements.txt` 中 **CUDA 12.1** 版 PyTorch 匹配）
- 建议安装 **sox**（部分音频处理会用到）：

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y sox libsox-dev
```

### 1. 克隆与 Conda 环境

```bash
git clone <你的仓库 HTTPS 或 SSH 地址>
cd CosyVoice-env-tts

conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice
```

### 2. 一键初始化（推荐）

在已激活的 Conda 环境中执行：

```bash
bash quick_init.sh
```

脚本会依次完成：

1. `pip install -r requirements.txt`（可用国内镜像，见下）
2. `bash env_instruct_pipeline/scripts/download_datasets.sh`（RIR + LibriTTS `dev-clean`）
3. `prepare_kaldi_libritts.py`（自动识别 `LibriTTS/` 或 `LibriTTS/LibriTTS/` 解压结构）
4. `download_pretrained_cosyvoice3.py`（默认 **ModelScope**；海外见下）

**常用环境变量**（均为可选）：

```bash
# 国内 pip 镜像示例
export PIP_EXTRA_ARGS='-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com'

# 海外下载预训练改为 Hugging Face
export PRETRAINED_BACKEND=huggingface

# 预训练保存路径（默认仓库内 pretrained_models/Fun-CosyVoice3-0.5B）
export PRETRAINED_DIR=/你的大盘路径/pretrained_models/Fun-CosyVoice3-0.5B

# 若某步已做过，可跳过
export SKIP_PIP=1          # 跳过 pip
export SKIP_DATA=1       # 跳过数据下载
export SKIP_KALDI=1      # 跳过 Kaldi 准备
export SKIP_PRETRAINED=1 # 跳过预训练下载
```

### 3. 手动初始化（与一键等价，便于排查）

```bash
pip install -r requirements.txt

bash env_instruct_pipeline/scripts/download_datasets.sh

# LibriTTS 解压后多为 datasets/speech/LibriTTS/LibriTTS/，请按实际目录二选一：
python env_instruct_pipeline/scripts/prepare_kaldi_libritts.py \
  --src_dir env_instruct_pipeline/datasets/speech/LibriTTS/LibriTTS \
  --des_dir env_instruct_pipeline/datasets/speech/kaldi

python env_instruct_pipeline/scripts/download_pretrained_cosyvoice3.py \
  --backend modelscope \
  --out_dir pretrained_models/Fun-CosyVoice3-0.5B
```

更细说明见 **`env_instruct_pipeline/README.md`**。

### 4. 数据处理：生成 env-instruct 训练数据

在已有 **Kaldi 人声** 与 **RIRS_NOISES** 的前提下，从仓库根目录执行其一：

```bash
# 小规模试跑（输出约百级 utterances）
bash env_instruct_pipeline/scripts/run_room_100.sh

# 或较大规模（默认 5000 train / 500 dev）
bash env_instruct_pipeline/scripts/run_room_5000.sh
```

默认输出目录：

- 试跑：`env_instruct_pipeline/output/env_instruct_room100/`
- 5000 档：`env_instruct_pipeline/output/env_instruct_room5000/`

指令类别（粗粒度 room）：`clean`、`small_room`、`medium_room`、`large_room`。

**可选**：导出检查用 JSON（wav / 文本 / instruct 对照）：

```bash
python env_instruct_pipeline/scripts/make_instruct_audio_list.py \
  --src_dir env_instruct_pipeline/output/env_instruct_room100/train
```

### 5. 训练 CosyVoice3（基座 + LoRA 流程）

1. 设置环境变量（路径按你本机修改）：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export PRETRAINED_DIR="${PRETRAINED_DIR:-$(pwd)/pretrained_models/Fun-CosyVoice3-0.5B}"
```

2. **数据目录**：`run_train_room.sh` 默认使用 **`env_instruct_pipeline/output/env_instruct_room100`**。若你用的是 `run_room_5000.sh`，必须指定：

```bash
export DATA_ROOT=env_instruct_pipeline/output/env_instruct_room5000
```

3. 启动训练（内部顺序：**Stage 0** 生成 parquet 与 `*.data.list` → **Stage 5** 训练 llm / flow / hifigan）：

```bash
bash env_instruct_pipeline/scripts/run_train_room.sh
```

**仅重新跑训练、不重建 parquet** 时：

```bash
export stage=5
export stop_stage=5
bash env_instruct_pipeline/scripts/run_train_room.sh
```

**训练产物**（默认）：

- 检查点与日志：`exp/env_instruct_room/`
- TensorBoard：`tensorboard/env_instruct_room/`

**Git 远程**：若使用 SSH，请将 `origin` 设为 `git@github.com:用户名/仓库名.git`；首次连接需信任主机键（见 `ssh-keyscan github.com >> ~/.ssh/known_hosts`）。勿将 **miniconda 安装包、训练日志、本机预训练目录的符号链接** 提交到仓库（已写入 `.gitignore`）。

---

## 数据与产物路径速查

| 内容 | 路径 |
|------|------|
| 下载的 RIR / LibriTTS | `env_instruct_pipeline/datasets/` |
| Kaldi 格式人声 | `env_instruct_pipeline/datasets/speech/kaldi/` |
| 合成后的 env-instruct（wav、scp、instruct 等） | `env_instruct_pipeline/output/<你的 OUT 目录>/` |
| Parquet 与 `train.data.list` / `dev.data.list` | 同上目录下 `train/parquet/`、`dev/parquet/` 及根级 `*.data.list` |
| CosyVoice3 预训练 | `PRETRAINED_DIR` 指向的目录（需含 `llm.pt`、`flow.pt`、`hifigan.pt` 等） |

---

## 云端与延伸阅读

- 从零在云 GPU 上跑通：**`env_instruct_pipeline/docs/DEPLOY_CLOUD.md`**
- 流水线逐步说明（中文）：**`env_instruct_pipeline/README.md`**
- CosyVoice 原版 WebUI、vLLM、Docker、评测表等：**`README_raw.md`**

---

## 常见问题

- **换机器只有 `git clone` 不够**：需重装依赖、重新下载或拷贝 `datasets/`、`output/`、`pretrained_models/`，并重新设置 `PYTHONPATH`、`PRETRAINED_DIR`、`DATA_ROOT`。
- **推送 GitHub 失败且提示大文件**：单文件不能超过 100MB；不要将 `miniconda.sh`、整包数据或日志提交进 Git。
- **训练报 CUDA 错误**：确认在 Linux + NVIDIA 上安装的是 **CUDA 版 PyTorch**（与 `requirements.txt` 中索引一致）。

---

## 数据来源（课题三条线）

- **A（已实现）**：合成 room-size env-instruct（干声 + OpenSLR28 RIR + 指令）。
- **B（计划）**：真实场景语料（CHiME、VOiCES 等）映射为指令。
- **C（计划）**：自蒸馏与增强，稳定扩大规模。

---

## 上游致谢与引用

代码与模型能力大量来自 [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)。引用格式见 **`README_raw.md`** 文末 BibTeX。
