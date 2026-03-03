# 云上运行指南（env-instruct 数据 + CosyVoice3 训练）

在带 **NVIDIA GPU（CUDA）** 的云服务器上跑「房间混响 instruct」数据生成与训练时，按以下步骤操作即可。

---

## 一、机器与资源

- **系统**：Linux（推荐 Ubuntu 20.04/22.04）
- **GPU**：至少 1 张 NVIDIA，显存 ≥16GB（如 T4 16GB、V100、A10、RTX 3090/4090）
- **内存**：≥32GB
- **磁盘**：≥30GB 可用（数据集 + 预训练模型 + 训练产出）

---

## 二、环境准备

### 1. 克隆/上传代码

```bash
# 若从 GitHub 拉取
git clone https://github.com/你的用户名/你的仓库名.git CosyVoice
cd CosyVoice
```

或把本地仓库打包上传到云上后解压，再 `cd CosyVoice`。

### 2. Conda 环境

```bash
conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice
```

### 3. 安装依赖

```bash
# 在仓库根目录
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

海外机器可去掉 `-i ... --trusted-host`，直接：

```bash
pip install -r requirements.txt
```

**说明**：`requirements.txt` 已包含 env-instruct 流水线与训练所需依赖（含 `scipy`、`pandas`、`tqdm`、`HyperPyYAML` 等）。Linux 上会安装 `deepspeed` 和 `onnxruntime-gpu`。

### 4. 可选：sox（部分音频处理会用到）

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y sox libsox-dev
# CentOS
# sudo yum install -y sox sox-devel
```

---

## 三、数据与模型

### 方案 A：在云上重新下载（推荐，代码与数据分离）

**1）下载环境数据（RIR + 噪声）与人声数据**

```bash
cd /path/to/CosyVoice
bash env_instruct_pipeline/scripts/download_datasets.sh
```

会得到：
- `env_instruct_pipeline/datasets/env/RIRS_NOISES/`
- `env_instruct_pipeline/datasets/speech/LibriTTS/`（如 dev-clean）

**2）准备 Kaldi 格式人声**

```bash
python env_instruct_pipeline/scripts/prepare_kaldi_libritts.py \
  --src_dir env_instruct_pipeline/datasets/speech/LibriTTS/LibriTTS \
  --des_dir env_instruct_pipeline/datasets/speech/kaldi
```

（若 `download_datasets.sh` 把 LibriTTS 解压到 `LibriTTS/` 下的子目录，请按实际路径调整 `--src_dir`。）

**3）生成粗粒度 room 数据（small/medium/large room + clean）**

```bash
export OUT_DIR=env_instruct_pipeline/output/env_instruct_room100
export MAX_TRAIN=100
export MAX_DEV=20
export INSTRUCT_MODE=coarse
bash env_instruct_pipeline/scripts/run_room_100.sh
```

**4）下载 CosyVoice3 预训练模型**

```bash
# 国内
python env_instruct_pipeline/scripts/download_pretrained_cosyvoice3.py --backend modelscope

# 海外
python env_instruct_pipeline/scripts/download_pretrained_cosyvoice3.py --backend huggingface
```

默认会下载到 `pretrained_models/Fun-CosyVoice3-0.5B`。

---

### 方案 B：本地上传（已有数据和模型）

若本地已生成好 `output/env_instruct_room100` 和预训练模型，可在本机打包后传到云上（注意 `.gitignore` 已忽略 `output/` 和 `datasets/`，需自行拷贝）：

```bash
# 本机打包（在 CosyVoice 根目录）
tar -czvf env_instruct_room100.tar.gz env_instruct_pipeline/output/env_instruct_room100
tar -czvf pretrained_cosyvoice3.tar.gz pretrained_models/Fun-CosyVoice3-0.5B
# 用 scp / oss / 对象存储 等上传到云上
```

云上解压到仓库根目录对应路径，保证存在：
- `env_instruct_pipeline/output/env_instruct_room100/train/`（含 `wav.scp`, `text`, `utt2spk`, `instruct`）
- `env_instruct_pipeline/output/env_instruct_room100/dev/`
- `pretrained_models/Fun-CosyVoice3-0.5B/`（含 `llm.pt`, `flow.pt`, `hifigan.pt`, `campplus.onnx`, `speech_tokenizer_v3.onnx`, `CosyVoice-BlankEN/`）

---

## 四、训练

```bash
cd /path/to/CosyVoice
conda activate cosyvoice

export PRETRAINED_DIR=pretrained_models/Fun-CosyVoice3-0.5B
export DATA_ROOT=env_instruct_pipeline/output/env_instruct_room100
export CUDA_VISIBLE_DEVICES=0

bash env_instruct_pipeline/scripts/run_train_room.sh
```

- 脚本会先做 parquet + data.list（若尚未生成），再依次训练 llm → flow → hifigan。
- 多卡时设置 `CUDA_VISIBLE_DEVICES=0,1` 等即可。

训练结果：
- 模型：`exp/env_instruct_room/llm/`, `flow/`, `hifigan/`（下各有 `torch_ddp/`）
- 日志：`tensorboard/env_instruct_room/`

---

## 五、常见问题

| 问题 | 处理 |
|------|------|
| `ModuleNotFoundError: No module named 'hyperpyyaml'` | `pip install HyperPyYAML` 或重新 `pip install -r requirements.txt` |
| `torch.cuda.is_available()` 为 False | 确认驱动与 CUDA、且安装的是带 CUDA 的 PyTorch（见 requirements 中 cu121 源） |
| OOM（显存不足） | 在 `examples/libritts/cosyvoice3/conf/cosyvoice3.yaml` 里将 `max_frames_in_batch` 从 2000 调小（如 1200） |
| 数据路径不对 | 检查 `DATA_ROOT`、`PRETRAINED_DIR` 与当前目录是否一致；路径建议用绝对路径 |

---

## 六、一键脚本示例（从零到训练）

在云上新建一台 GPU 实例后，可按下面顺序执行（需先 `git clone` 或上传代码并 `cd CosyVoice`）：

```bash
conda create -n cosyvoice python=3.10 -y && conda activate cosyvoice
pip install -r requirements.txt
bash env_instruct_pipeline/scripts/download_datasets.sh
python env_instruct_pipeline/scripts/prepare_kaldi_libritts.py \
  --src_dir env_instruct_pipeline/datasets/speech/LibriTTS/LibriTTS \
  --des_dir env_instruct_pipeline/datasets/speech/kaldi
INSTRUCT_MODE=coarse MAX_TRAIN=100 MAX_DEV=20 bash env_instruct_pipeline/scripts/run_room_100.sh
python env_instruct_pipeline/scripts/download_pretrained_cosyvoice3.py --backend modelscope
export PRETRAINED_DIR=pretrained_models/Fun-CosyVoice3-0.5B
bash env_instruct_pipeline/scripts/run_train_room.sh
```

按需把 `--backend` 改为 `huggingface`、或调整 `MAX_TRAIN`/`MAX_DEV`。
