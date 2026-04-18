"""训练一个 4 类混响分类器(clean/small/medium/large),用来量化评估 TTS 模型输出。

用法:
    # 训练(从我们的数据集学):
    python tools/train_reverb_classifier.py train \
        --data_root /media/volume/geo3/env_instruct_output/env_instruct_room5000_both \
        --out /media/volume/geo3/reverb_classifier.pt

    # 评估一批音频(传目录,里面放对应类别命名的 wav):
    python tools/train_reverb_classifier.py predict \
        --ckpt /media/volume/geo3/reverb_classifier.pt \
        --audio_dir planb_test_out
"""
import argparse
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SR = 24000
NUM_CLASSES = 4
CLASSES = ['clean', 'small', 'medium', 'large']
CROP_SEC = 3.0  # 训练/评估统一裁剪到 3 秒


# ---------------- 标签解析:从 instruct 文本推类别 ----------------
ZH_KEYS = {
    'clean':  ['干声', '无混响', '录音棚', '无回声', '无空间反射', '清晰的无混响'],
    'small':  ['小房间', '小空间', '狭小'],
    'medium': ['中等', '普通房间'],
    'large':  ['大型厅堂', '大厅', '开阔空间', '大房间', '宽敞', '回荡'],
}
EN_KEYS = {
    'clean':  ['dry recording', 'dry vocal', 'studio-quality', 'no room echo', 'no reverberation', 'anechoic'],
    'small':  ['small room', 'small space', 'tiny room', 'compact room', 'enclosed space', 'short reverb'],
    'medium': ['medium-sized', 'medium space', 'ordinary room', 'average-sized', 'medium chamber', 'noticeable room reverb'],
    'large':  ['large hall', 'spacious hall', 'open space', 'vast room', 'cavernous', 'long reverberant', 'lengthy reverb'],
}


def label_of(text: str) -> int:
    t = text.lower()
    for i, cls in enumerate(CLASSES):
        for kw in ZH_KEYS[cls]:
            if kw in text:
                return i
        for kw in EN_KEYS[cls]:
            if kw in t:
                return i
    return -1


# ---------------- 特征提取 ----------------
class MelExtractor(nn.Module):
    def __init__(self, sr=SR, n_mels=64, n_fft=1024, hop=256):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0)

    def forward(self, wav):
        m = self.melspec(wav)
        m = torch.log(m + 1e-6)
        # per-sample 标准化 → 去掉音量影响
        mean = m.mean(dim=(-2, -1), keepdim=True)
        std = m.std(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
        return (m - mean) / std


# ---------------- 模型 ----------------
class ReverbClassifier(nn.Module):
    def __init__(self, n_mels=64, num_classes=NUM_CLASSES):
        super().__init__()
        self.extractor = MelExtractor(n_mels=n_mels)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, wav):
        m = self.extractor(wav)
        if m.dim() == 3:  # [B, F, T]
            m = m.unsqueeze(1)  # [B, 1, F, T]
        h = self.conv(m).flatten(1)
        return self.head(h)


# ---------------- 数据集 ----------------
class ReverbDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: Path, split: str, train: bool):
        self.train = train
        split_dir = data_root / split
        wav_scp = {}
        for line in (split_dir / 'wav.scp').read_text().splitlines():
            if not line.strip(): continue
            k, _, v = line.partition(' '); wav_scp[k] = v
        inst_map = {}
        for line in (split_dir / 'instruct').read_text().splitlines():
            if not line.strip(): continue
            k, _, v = line.partition(' '); inst_map[k] = v
        self.items = []
        skipped = 0
        for utt, ins in inst_map.items():
            lbl = label_of(ins)
            if lbl == -1:
                skipped += 1
                continue
            if utt not in wav_scp:
                continue
            self.items.append((wav_scp[utt], lbl))
        print(f'[{split}] {len(self.items)} 样本,跳过 {skipped} 无法归类')

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        wav_path, lbl = self.items[i]
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != SR: wav = torchaudio.transforms.Resample(sr, SR)(wav)
        # peak normalize
        wav = wav / (wav.abs().max() + 1e-9)
        # crop/pad to 3s
        n_need = int(CROP_SEC * SR)
        n_have = wav.shape[1]
        if n_have >= n_need:
            start = random.randint(0, n_have - n_need) if self.train else (n_have - n_need) // 2
            wav = wav[:, start:start + n_need]
        else:
            wav = F.pad(wav, (0, n_need - n_have))
        return wav.squeeze(0), lbl


# ---------------- 训练 ----------------
def do_train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)
    train_ds = ReverbDataset(data_root, 'train', train=True)
    dev_ds = ReverbDataset(data_root, 'dev', train=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    dev_loader = torch.utils.data.DataLoader(dev_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = ReverbClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for wav, lbl in train_loader:
            wav, lbl = wav.to(device), lbl.to(device)
            logit = model(wav)
            loss = F.cross_entropy(logit, lbl)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * lbl.size(0)
            correct += (logit.argmax(1) == lbl).sum().item()
            total += lbl.size(0)
        tr_acc = correct / total

        # dev
        model.eval()
        total, correct = 0, 0
        cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
        with torch.no_grad():
            for wav, lbl in dev_loader:
                wav, lbl = wav.to(device), lbl.to(device)
                pred = model(wav).argmax(1)
                correct += (pred == lbl).sum().item()
                total += lbl.size(0)
                for t, p in zip(lbl.tolist(), pred.tolist()):
                    cm[t, p] += 1
        dv_acc = correct / total
        print(f'E{epoch}  train_loss={loss_sum/len(train_ds):.4f} train_acc={tr_acc:.3f}  dev_acc={dv_acc:.3f}')
        if dv_acc > best_acc:
            best_acc = dv_acc
            torch.save({'state': model.state_dict(), 'acc': dv_acc, 'classes': CLASSES}, args.out)
            print(f'   ↑ best, saved → {args.out}')
            # 打印 confusion matrix
            print('   confusion matrix (行=真实,列=预测):')
            print('   ' + ' '.join(f'{c:>7s}' for c in CLASSES))
            for i, c in enumerate(CLASSES):
                row = ' '.join(f'{cm[i,j].item():>7d}' for j in range(NUM_CLASSES))
                print(f'   {c:>7s} ' + row)

    print(f'\n[done] best dev accuracy: {best_acc:.3f}  saved: {args.out}')


# ---------------- 推理 ----------------
@torch.no_grad()
def do_predict(args):
    ckpt = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReverbClassifier().to(device)
    model.load_state_dict(ckpt['state']); model.eval()
    classes = ckpt.get('classes', CLASSES)

    audio_dir = Path(args.audio_dir)
    wavs = sorted(audio_dir.glob('*.wav'))
    if not wavs:
        print(f'no wavs in {audio_dir}')
        return

    print(f'{"file":32s}  {"pred":>8s}  {" ".join(f"{c:>7s}" for c in classes)}')
    print('=' * 70)
    for wp in wavs:
        wav, sr = torchaudio.load(str(wp))
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != SR: wav = torchaudio.transforms.Resample(sr, SR)(wav)
        wav = wav / (wav.abs().max() + 1e-9)
        # center crop/pad
        n_need = int(CROP_SEC * SR)
        if wav.shape[1] >= n_need:
            start = (wav.shape[1] - n_need) // 2
            wav = wav[:, start:start + n_need]
        else:
            wav = F.pad(wav, (0, n_need - wav.shape[1]))
        wav = wav.to(device)
        logit = model(wav)
        prob = logit.softmax(-1).squeeze(0).cpu().tolist()
        pred = classes[int(max(range(len(prob)), key=lambda k: prob[k]))]
        prob_str = ' '.join(f'{p:>7.3f}' for p in prob)
        print(f'{wp.name:32s}  {pred:>8s}  {prob_str}')


# ---------------- CLI ----------------
def main():
    sub = argparse.ArgumentParser().add_subparsers(dest='cmd', required=True)
    ap = argparse.ArgumentParser()
    ss = ap.add_subparsers(dest='cmd', required=True)

    p1 = ss.add_parser('train')
    p1.add_argument('--data_root', required=True)
    p1.add_argument('--out', default='/media/volume/geo3/reverb_classifier.pt')
    p1.add_argument('--epochs', type=int, default=8)
    p1.add_argument('--batch', type=int, default=32)

    p2 = ss.add_parser('predict')
    p2.add_argument('--ckpt', default='/media/volume/geo3/reverb_classifier.pt')
    p2.add_argument('--audio_dir', required=True)

    args = ap.parse_args()
    if args.cmd == 'train':
        do_train(args)
    elif args.cmd == 'predict':
        do_predict(args)


if __name__ == '__main__':
    main()
