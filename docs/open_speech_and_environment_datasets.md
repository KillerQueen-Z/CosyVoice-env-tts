# 开源语音与环境音资源汇总

用于 CosyVoice「人声 + 环境音 / 混响」instruct 微调的数据构建：干净人声数据集、环境/噪声/RIR 数据集、以及融合方法（卷积混响 + 加噪）。

---

## 一、干净人声数据集（干声）

| 名称 | 语言 | 采样率 | 规模 | 许可 | 下载/说明 |
|-----|------|--------|------|------|-----------|
| **LibriSpeech** | 英文 | 16k | 960h+ | CC BY 4.0 | [OpenSLR 12](https://www.openslr.org/12/)；CosyVoice 示例已用 |
| **LibriTTS** | 英文 | 24k | 585h | CC BY 4.0 | [OpenSLR 60](https://www.openslr.org/60/)；带文本，适合 TTS |
| **VCTK** | 英文 | 48k | 44h, 110 说话人 | CC BY 4.0 | [OpenSLR 43](https://www.openslr.org/43/)；多说话人朗读 |
| **AISHELL-1** | 中文普通话 | 16k | 178h, 400 说话人 | 学术免费 | [OpenSLR 33](https://www.openslr.org/33/)；安静室内录制 |
| **AISHELL-2** | 中文普通话 | 16k | 1000h+ | 需申请 | [官网](https://www.aishelltech.com/aishell_2) |
| **MagicData-RAMC** | 中文普通话 | 16k | 755h | 需申请 | CosyVoice 已有示例 `examples/magicdata-read` |
| **DNS Challenge 干净语音** | 多语 | 16k/48k | 数百小时 | 非商用/研究 | [GitHub microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge)；含干净语音列表与合成脚本 |

---

## 二、环境音 / 噪声 / 混响（RIR）数据集

| 名称 | 内容 | 采样率 | 许可 | 下载/说明 |
|-----|------|--------|------|-----------|
| **OpenSLR 28 (RIR + Noise)** | 真实+仿真 RIR、各向同性噪声、MUSAN 点源噪声 | 16k | Apache 2.0 | [OpenSLR 28](https://www.openslr.org/28/) `rirs_noises.zip`（约 1.3G）；**最常用做混响+噪声增强** |
| **MUSAN** | 音乐、语音、噪声（Freesound 等） | 16k | CC BY 4.0 | [OpenSLR 17](https://www.openslr.org/17/)；常与 SLR28 一起用 |
| **RWCP 声场数据库 (OpenSLR 13)** | 非语音声、房间重建、麦克风阵列 RIR、背景噪声 | - | 研究用 | [OpenSLR 13](https://www.openslr.org/13/) |
| **BUT Reverb DB** | 真实房间 RIR、环境噪声、重放语音 | - | CC BY 4.0 | [BUT Reverb](https://speech.fit.vut.cz/software/but-speech-fit-reverb-database) |
| **DNS Challenge 噪声集** | 多种真实噪声（室内、街道、办公室等） | 16k/48k | 随 DNS 规则 | 见 [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge) |
| **ESC-50** | 50 类环境声（门、狗、雨、街道等） | 44.1k | CC BY 4.0 | [GitHub ESC-50](https://github.com/karoldvl/ESC-50) |

---

## 三、现成「带背景噪声的语音」数据集

| 名称 | 内容 | 说明 |
|-----|------|------|
| **CHiME-5 / CHiME-6** | 家庭聚餐场景真实录音，多人对话 + 室内噪声 | [chimechallenge.org](https://www.chimechallenge.org/)；CC BY-SA 4.0；16k |
| **CHiME-4** | WSJ0 + 模拟/真实噪声（公交、咖啡馆、街道等） | 4 种环境；适合鲁棒 ASR/TTS 研究 |
| **DNS Challenge 合成数据** | 用官方脚本把「干净语音 + 噪声」合成带噪语音 | 可控制 SNR、噪声类型；适合做「干净→带噪」配对 |
| **AISHELL-4** | 会议场景多说话人中文，含重叠与噪声 | [OpenSLR](https://www.openslr.org/111/)；真实会议环境 |

若你希望**直接拿到「说话 + 背景噪声」的成对数据**，可优先用 **DNS Challenge** 的合成流程或 **CHiME** 的真实录音。

---

## 四、融合方法：卷积混响 + 加性噪声

目标：**干声 `s(t)` → 带环境/混响的语音 `y(t)`**，用于构造 (text, instruct, wav) 三元组。

### 4.1 卷积混响（模拟房间/山洞等）

- **公式**：`y = s * h`（`*` 为卷积，`h` 为房间冲激响应 RIR）。
- **数据**：用 OpenSLR 28 里的 RIR 文件（不同房间/距离），选「山洞感」强的或长尾 RIR。
- **实现**：
  - **torchaudio**：`torch.nn.functional.conv1d` 或 `torchaudio.functional.fftconvolve`（若有）。
  - **SciPy**：`scipy.signal.fftconvolve(s, h)`。
  - **第三方**：`convolution-reverb`（PyPI），或 SpeechBrain 的环境失真教程 [Environmental Corruption](https://speechbrain.readthedocs.io/en/stable/tutorials/preprocessing/environmental-corruption.html)。

### 4.2 加性背景噪声

- **公式**：`y = s + α * n`，其中 `n` 为噪声片段，`α` 由目标 SNR（dB）推出。
- **数据**：OpenSLR 28 的 noise、MUSAN、或 DNS 噪声集；环境描述与 instruct 一致（如「街道」「山洞内」）。
- **实现**：对 `s` 和 `n` 按长度裁剪/重复并对齐，根据 `SNR = 10*log10(Ps/Pn)` 算 `α`，再混合。

### 4.3 组合流程（推荐）

1. **干声** → 可选：归一化、重采样到 16k/24k。  
2. **混响**：选一条 RIR，与干声卷积，得到 `s_rev`。  
3. **加噪**：取噪声片段，按目标 SNR 与 `s_rev` 混合，得到最终 `y`。  
4. **instruct**：根据使用的 RIR/噪声类型写自然语言，如「模拟在山洞里的声音」「模拟在街道旁，有车流声」。

这样即可用开源数据批量生成「人声 + 环境音」数据，并和 CosyVoice 的 `instruct` 字段对齐。

---

## 五、本仓库中的相关脚本

- **`tools/make_demo_env_instruct_dataset.py`**：生成最小 demo 数据集（合成音+简单回声/噪声），用于验证 Kaldi 格式与 parquet 管线。  
- **`tools/augment_speech_with_rir_noise.py`**（见下）：用 OpenSLR 28 的 RIR 与噪声，对已有干声做「混响+加噪」，并写出 `wav.scp/text/utt2spk/instruct`，便于接到 `make_parquet_list.py` 做 CosyVoice 微调。

---

## 六、推荐组合（快速起步）

1. **干声**：LibriTTS 或 AISHELL-1（按 CosyVoice 所需采样率重采样）。  
2. **RIR + 噪声**：OpenSLR 28 的 `rirs_noises.zip`（[直接下载](https://www.openslr.org/resources/28/rirs_noises.zip)）。  
3. **融合**：用本仓库脚本对干声做卷积 + 加噪，并生成带 instruct 的 Kaldi 目录：

```bash
# 1) 下载并解压 OpenSLR 28
wget https://www.openslr.org/resources/28/rirs_noises.zip
unzip rirs_noises.zip   # 得到 RIRS_NOISES/

# 2) 准备干净语音的 Kaldi 目录（例如 data/clean_train，内含 wav.scp, text, utt2spk）

# 3) 运行增强脚本（输出带 instruct 的 data/env_train）
python tools/augment_speech_with_rir_noise.py \
  --clean_data_dir data/clean_train \
  --rir_noise_dir /path/to/RIRS_NOISES \
  --out_dir data/env_train \
  --target_sr 24000 \
  --snr_db 15 \
  --noise_prob 0.6
```

4. **CosyVoice**：对 `data/env_train` 跑 `make_parquet_list.py` → 生成 `train.data.list` / `dev.data.list` → 微调。

这样全部使用开源数据，即可构建「人声 + 环境音」的 instruct 微调数据集。

---

## 七、链接速查

| 资源 | URL |
|------|-----|
| OpenSLR 28 (RIR+Noise) | https://www.openslr.org/28/ |
| OpenSLR 17 (MUSAN) | https://www.openslr.org/17/ |
| LibriTTS | https://www.openslr.org/60/ |
| AISHELL-1 | https://www.openslr.org/33/ |
| DNS Challenge (GitHub) | https://github.com/microsoft/DNS-Challenge |
| CHiME 数据 | https://www.chimechallenge.org/ |
| SpeechBrain 环境失真教程 | https://speechbrain.readthedocs.io/en/stable/tutorials/preprocessing/environmental-corruption.html |
