## Demo: 环境音/空间感 Instruct 数据集（最小样例）

这个目录用于演示 CosyVoice 的 `instruct` 条件输入如何进入训练管线：样本里同时提供 `text` 与 `instruct`，并配对一条对应风格/环境的语音。

### 你会得到什么

- `train/` 与 `dev/` 两个子集
- 每个子集包含 Kaldi 风格的元数据文件：
  - `wav.scp`：utt → wav 路径
  - `text`：utt → 文本
  - `utt2spk`：utt → 说话人
  - `instruct`：utt → 指令（例如“模拟在山洞里的声音…”）
  - `spk2utt`
- wav 文件在 `train/wavs/`、`dev/wavs/`

### 生成 demo 数据

在仓库根目录执行：

```bash
python tools/make_demo_env_instruct_dataset.py --out_dir demo_env_instruct_dataset --sr 24000
```

> 注意：脚本生成的是“合成的类人声波形 + 简单回声/噪声”，只是为了跑通数据格式与训练管线。真实训练请用你录制/采集的干净人声作为输入，再做环境增强。

### 下一步（进入 CosyVoice 训练）

1) （可选但推荐）提取 embedding / speech token（提高训练速度、或满足离线特征需求）

2) 生成 parquet + `data.list`

```bash
mkdir -p demo_env_instruct_dataset/train/parquet
mkdir -p demo_env_instruct_dataset/dev/parquet

python tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 4 \
  --src_dir demo_env_instruct_dataset/train --des_dir demo_env_instruct_dataset/train/parquet

python tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 4 \
  --src_dir demo_env_instruct_dataset/dev --des_dir demo_env_instruct_dataset/dev/parquet
```

3) 写 `train.data.list` / `dev.data.list`

```bash
cat demo_env_instruct_dataset/train/parquet/data.list > demo_env_instruct_dataset/train.data.list
cat demo_env_instruct_dataset/dev/parquet/data.list   > demo_env_instruct_dataset/dev.data.list
```

4) 用你拷贝修改过的 yaml（例如 `conf/my_finetune_instruct.yaml`）开始训练

