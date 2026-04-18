"""Neural Reverb:学 4 个 class-specific RIR 应用到任意 clean 波形。
本质是个非常简单的"可学习 IIR 滤波器组"。

用法:
    # 训练
    python tools/train_neural_reverb.py train \
        --data_root /media/volume/geo3/env_instruct_output/env_instruct_room5000_both \
        --out /media/volume/geo3/neural_reverb.pt --epochs 30

    # 推理(给一个 clean wav 加 4 种混响)
    python tools/train_neural_reverb.py predict \
        --ckpt /media/volume/geo3/neural_reverb.pt \
        --input clean_input.wav --out_dir nr_out
"""
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


SR = 24000
RIR_LENGTH = int(2.0 * SR)  # 2 秒 RIR(够覆盖 large RT60)
NUM_CLASSES = 4
CLASSES = ['clean', 'small', 'medium', 'large']
CROP_SEC = 4.0


# ---------------- 标签解析 ----------------
ZH_KEYS = {
    'clean': ['干声', '无混响', '录音棚', '无回声', '无空间反射'],
    'small': ['小房间', '小空间', '狭小'],
    'medium': ['中等', '普通房间'],
    'large': ['大型厅堂', '大厅', '开阔空间', '大房间', '宽敞', '回荡'],
}
EN_KEYS = {
    'clean': ['dry recording', 'dry vocal', 'studio-quality', 'no room echo', 'no reverberation', 'anechoic'],
    'small': ['small room', 'small space', 'tiny room', 'compact room', 'enclosed space', 'short reverb'],
    'medium': ['medium-sized', 'medium space', 'ordinary room', 'average-sized', 'medium chamber', 'noticeable room reverb'],
    'large': ['large hall', 'spacious hall', 'open space', 'vast room', 'cavernous', 'long reverberant', 'lengthy reverb'],
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


# ---------------- Model ----------------
class NeuralReverb(nn.Module):
    """4 个可学习的 class-specific RIR,通过 FFT 卷积应用到 clean 波形"""

    def __init__(self, num_classes=NUM_CLASSES, rir_length=RIR_LENGTH):
        super().__init__()
        self.rir_length = rir_length
        # 初始化:第一帧 = 1(delta = 不混响),其他 = 小随机
        rirs = torch.randn(num_classes, rir_length) * 0.001
        rirs[:, 0] = 1.0
        self.rirs = nn.Parameter(rirs)
        # 一个可学习的 wet/dry 混合系数(每类一个)
        self.dry_mix = nn.Parameter(torch.ones(num_classes) * 0.7)

    def get_rir(self, class_id):
        """返回归一化的 RIR(取消能量爆炸)"""
        rir = self.rirs[class_id]  # [B, L] 或 [L]
        # 用最大幅度归一化,保持类间相对能量
        peak = rir.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        return rir / peak

    def forward(self, clean_wav, class_id):
        """clean_wav: [B, T],class_id: [B]
        返回 [B, T] 带混响的波形"""
        B, T = clean_wav.shape
        rir = self.get_rir(class_id)  # [B, L]
        # FFT 卷积
        n_conv = T + self.rir_length - 1
        n_fft = 1 << (n_conv - 1).bit_length()
        clean_fft = torch.fft.rfft(clean_wav, n=n_fft)
        rir_fft = torch.fft.rfft(rir, n=n_fft)
        wet = torch.fft.irfft(clean_fft * rir_fft, n=n_fft)[:, :T]
        # wet/dry mix
        mix = torch.sigmoid(self.dry_mix[class_id]).unsqueeze(-1)  # [B, 1]
        out = mix * wet + (1 - mix) * clean_wav
        # 归一化避免溢出
        peak = out.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        return out / peak * 0.95


# ---------------- Loss ----------------
def stft_loss(pred, target, n_ffts=(512, 1024, 2048)):
    """Multi-resolution STFT loss"""
    loss = 0.0
    for n_fft in n_ffts:
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=pred.device)
        sp_pred = torch.stft(pred, n_fft=n_fft, hop_length=hop, window=win, return_complex=True).abs()
        sp_tgt = torch.stft(target, n_fft=n_fft, hop_length=hop, window=win, return_complex=True).abs()
        loss = loss + F.l1_loss(torch.log(sp_pred + 1e-7), torch.log(sp_tgt + 1e-7))
    return loss / len(n_ffts)


# ---------------- Dataset ----------------
class ReverbPairDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: Path, split: str, train: bool):
        self.train = train
        d = data_root / split
        wav_scp = {}
        for line in (d / 'wav.scp').read_text().splitlines():
            if not line.strip(): continue
            k, _, v = line.partition(' '); wav_scp[k] = v
        # clean.scp 优先用没改过的;改过的也试试 .disabled_for_llm_train
        clean_scp = d / 'wav.clean.scp'
        if not clean_scp.exists():
            cand = d / 'wav.clean.scp.disabled_for_llm_train'
            if cand.exists(): clean_scp = cand
        clean_map = {}
        for line in clean_scp.read_text().splitlines():
            if not line.strip(): continue
            k, _, v = line.partition(' '); clean_map[k] = v
        inst_map = {}
        for line in (d / 'instruct').read_text().splitlines():
            if not line.strip(): continue
            k, _, v = line.partition(' '); inst_map[k] = v
        self.items = []
        for utt, ins in inst_map.items():
            lbl = label_of(ins)
            if lbl < 0 or utt not in wav_scp or utt not in clean_map:
                continue
            self.items.append((clean_map[utt], wav_scp[utt], lbl))
        print(f'[{split}] {len(self.items)} pairs')

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        clean_p, reverb_p, lbl = self.items[i]
        cw, csr = torchaudio.load(clean_p)
        rw, rsr = torchaudio.load(reverb_p)
        if cw.shape[0] > 1: cw = cw.mean(0, keepdim=True)
        if rw.shape[0] > 1: rw = rw.mean(0, keepdim=True)
        if csr != SR: cw = torchaudio.transforms.Resample(csr, SR)(cw)
        if rsr != SR: rw = torchaudio.transforms.Resample(rsr, SR)(rw)
        # 对齐长度
        L = min(cw.shape[1], rw.shape[1])
        cw, rw = cw[:, :L], rw[:, :L]
        # 归一化
        cw = cw / (cw.abs().max() + 1e-9)
        rw = rw / (rw.abs().max() + 1e-9)
        # crop / pad
        n = int(CROP_SEC * SR)
        if L >= n:
            s = random.randint(0, L - n) if self.train else (L - n) // 2
            cw, rw = cw[:, s:s+n], rw[:, s:s+n]
        else:
            cw = F.pad(cw, (0, n - L))
            rw = F.pad(rw, (0, n - L))
        return cw.squeeze(0), rw.squeeze(0), lbl


# ---------------- Train ----------------
def do_train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)
    train_ds = ReverbPairDataset(data_root, 'train', train=True)
    dev_ds = ReverbPairDataset(data_root, 'dev', train=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    dev_loader = torch.utils.data.DataLoader(dev_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = NeuralReverb().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_dev = float('inf')

    for epoch in range(args.epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for cw, rw, lbl in train_loader:
            cw, rw, lbl = cw.to(device), rw.to(device), lbl.to(device)
            pred = model(cw, lbl)
            loss = F.l1_loss(pred, rw) + 2.0 * stft_loss(pred, rw)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * lbl.size(0); n += lbl.size(0)
        train_loss = loss_sum / n

        model.eval()
        dl_sum, dn = 0.0, 0
        with torch.no_grad():
            for cw, rw, lbl in dev_loader:
                cw, rw, lbl = cw.to(device), rw.to(device), lbl.to(device)
                pred = model(cw, lbl)
                loss = F.l1_loss(pred, rw) + 2.0 * stft_loss(pred, rw)
                dl_sum += loss.item() * lbl.size(0); dn += lbl.size(0)
        dev_loss = dl_sum / dn
        print(f'E{epoch:2d}  train={train_loss:.4f}  dev={dev_loss:.4f}', end='')
        if dev_loss < best_dev:
            best_dev = dev_loss
            torch.save({'state': model.state_dict(), 'classes': CLASSES, 'rir_length': RIR_LENGTH}, args.out)
            print(f'  ↑ best, saved')
        else:
            print('')


# ---------------- Predict ----------------
@torch.no_grad()
def do_predict(args):
    ckpt = torch.load(args.ckpt, map_location='cpu')
    classes = ckpt.get('classes', CLASSES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralReverb(num_classes=len(classes), rir_length=ckpt.get('rir_length', RIR_LENGTH)).to(device)
    model.load_state_dict(ckpt['state']); model.eval()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    cw, sr = torchaudio.load(args.input)
    if cw.shape[0] > 1: cw = cw.mean(0, keepdim=True)
    if sr != SR: cw = torchaudio.transforms.Resample(sr, SR)(cw)
    cw = cw / (cw.abs().max() + 1e-9)
    cw = cw.squeeze(0).to(device)

    # 对每类生成一个版本
    for i, cls in enumerate(classes):
        lbl = torch.tensor([i], device=device)
        out = model(cw.unsqueeze(0), lbl).squeeze(0).cpu()
        outp = out_dir / f'{Path(args.input).stem}_{cls}.wav'
        torchaudio.save(str(outp), out.unsqueeze(0), SR)
        print(f'  {cls:>7s} → {outp}')


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ss = ap.add_subparsers(dest='cmd', required=True)
    p1 = ss.add_parser('train')
    p1.add_argument('--data_root', required=True)
    p1.add_argument('--out', default='/media/volume/geo3/neural_reverb.pt')
    p1.add_argument('--epochs', type=int, default=30)
    p1.add_argument('--batch', type=int, default=4)
    p1.add_argument('--lr', type=float, default=1e-3)

    p2 = ss.add_parser('predict')
    p2.add_argument('--ckpt', default='/media/volume/geo3/neural_reverb.pt')
    p2.add_argument('--input', required=True, help='clean wav 路径')
    p2.add_argument('--out_dir', default='neural_reverb_out')

    args = ap.parse_args()
    if args.cmd == 'train': do_train(args)
    else: do_predict(args)


if __name__ == '__main__':
    main()
