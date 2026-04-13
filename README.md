# SoniSphere-LoRA（基于 CosyVoice 的环境感知 TTS）

本仓库是在 **CosyVoice3-0.5B** 上的课题向 fork，研究 **指令驱动的环境语音重建**：给定如「小房间」「大厅」等自然语言指令，合成带对应混响特征的语音（与「去混响、做干声」的传统 TTS 目标不同）。

**核心**：CosyVoice3-0.5B（LLM + flow + vocoder）→ **LoRA 微调** → 监督来自 **env-instruct**（干声 + RIR + 文本指令），并计划扩展真实场景语料与自蒸馏。

上游 CosyVoice 的完整安装、演示、评测与引用说明保留在 **`README_raw.md`**。

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
