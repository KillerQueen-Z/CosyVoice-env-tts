"""计算 RIRS_NOISES 里每个 RIR 的 RT60,按目标范围筛出核心 RIR,
并生成每类一段演示音频让你试听类间差异。

用法:
  python tools/rt60_filter_rirs.py \
    --rir_dir env_instruct_pipeline/datasets/env/RIRS_NOISES \
    --speech_sample /media/volume/geo3/speech_data/LibriTTS/train-clean-100/<选一个 wav> \
    --out_dir rt60_demo_out
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.signal import fftconvolve


def compute_rt60(rir_path: Path, sr_target: int = 24000) -> float:
    """Schroeder 积分法估算 RT60(用 RT20 外推到 -60dB)"""
    wav, sr = torchaudio.load(str(rir_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sr_target:
        wav = torchaudio.transforms.Resample(sr, sr_target)(wav)
    h = wav.squeeze(0).numpy()
    # 从最大能量点开始
    peak = np.argmax(np.abs(h))
    h = h[peak:]
    # Schroeder 反向能量积分
    edc = np.flipud(np.cumsum(np.flipud(h ** 2)))
    edc = edc / (edc[0] + 1e-12)
    edc_db = 10 * np.log10(edc + 1e-12)
    # 在 -5 到 -25 dB 之间拟合斜率(RT20)
    try:
        idx_start = int(np.argmax(edc_db <= -5))
        idx_end = int(np.argmax(edc_db <= -25))
        if idx_end <= idx_start:
            return float('nan')
        t = np.arange(idx_start, idx_end) / sr_target
        slope, _ = np.polyfit(t, edc_db[idx_start:idx_end], 1)
        if slope >= 0:
            return float('nan')
        rt60 = -60.0 / slope  # 秒
        return float(rt60)
    except Exception:
        return float('nan')


def make_demo(speech_path: Path, rir_paths: list, label: str, out_dir: Path, sr=24000):
    """用 RIR 卷积一段语音,保存为演示音频"""
    wav, orig_sr = torchaudio.load(str(speech_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.transforms.Resample(orig_sr, sr)(wav)
    speech = wav.squeeze(0).numpy()

    if label == 'clean' or not rir_paths or rir_paths[0] is None:
        out = speech
        rir_path = None
    else:
        # 用中位 RIR(不是最大/最小,代表该类中心)
        rir_path = rir_paths[len(rir_paths) // 2]
        rir_wav, rir_sr = torchaudio.load(str(rir_path))
        if rir_wav.shape[0] > 1:
            rir_wav = rir_wav.mean(dim=0, keepdim=True)
        if rir_sr != sr:
            rir_wav = torchaudio.transforms.Resample(rir_sr, sr)(rir_wav)
        rir = rir_wav.squeeze(0).numpy()
        peak = np.max(np.abs(rir)) + 1e-12
        rir = rir / peak
        out = fftconvolve(speech, rir, mode='full')[:len(speech)]
        out = out / (np.max(np.abs(out)) + 1e-9) * 0.95

    outp = out_dir / f'{label}_demo.wav'
    torchaudio.save(str(outp), torch.from_numpy(out.astype(np.float32)).unsqueeze(0), sr)
    return outp, rir_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rir_dir', required=True)
    ap.add_argument('--speech_sample', required=True, help='一段纯净语音,用于演示')
    ap.add_argument('--out_dir', default='rt60_demo_out')
    ap.add_argument('--save_manifest', default='rt60_manifest.json',
                    help='保存 RT60 分组到 JSON,给后续 build 脚本用')
    args = ap.parse_args()

    rir_root = Path(args.rir_dir)
    sim = rir_root / 'simulated_rirs'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有 RIR 并算 RT60
    folders = {'small': 'smallroom', 'medium': 'mediumroom', 'large': 'largeroom'}
    all_rt60 = defaultdict(list)  # cat → [(rir_path, rt60), ...]
    for cat, sub in folders.items():
        rirs = sorted((sim / sub).rglob('*.wav'))
        print(f'[info] {cat}: 扫描 {len(rirs)} 个 RIR ...')
        for rp in rirs:
            rt = compute_rt60(rp)
            if not np.isnan(rt) and rt > 0.01 and rt < 10.0:
                all_rt60[cat].append((str(rp), rt))
        print(f'  有效 RIR: {len(all_rt60[cat])}')

    # 打印直方图
    print('\n=== 原始 RT60 分布 ===')
    for cat in ['small', 'medium', 'large']:
        rts = [x[1] for x in all_rt60[cat]]
        if not rts: continue
        p = np.percentile(rts, [5, 25, 50, 75, 95])
        print(f'  {cat:7s}  n={len(rts):4d}  '
              f'p5={p[0]:.2f}s  p25={p[1]:.2f}s  median={p[2]:.2f}s  '
              f'p75={p[3]:.2f}s  p95={p[4]:.2f}s')

    # 核心区间筛选(目标:类间 gap 明显)
    targets = {
        'small':  (0.10, 0.30),   # 轻混响
        'medium': (0.55, 0.75),   # 中混响
        'large':  (1.20, 100.0),  # 长混响(上限宽松,取 RT60 越大越好)
    }
    print('\n=== 核心区间过滤 ===')
    filtered = {}
    for cat, (lo, hi) in targets.items():
        good = [p for p, rt in all_rt60[cat] if lo <= rt <= hi]
        # 如果某类没过滤到足够多,放宽
        if len(good) < 50:
            print(f'  [warn] {cat} 目标范围 [{lo},{hi}]s 只有 {len(good)} 条,放宽到本类 p75 以上')
            rts = sorted([x[1] for x in all_rt60[cat]])
            if cat == 'small':
                thresh = np.percentile(rts, 25)
                good = [p for p, rt in all_rt60[cat] if rt <= thresh]
            elif cat == 'large':
                thresh = np.percentile(rts, 75)
                good = [p for p, rt in all_rt60[cat] if rt >= thresh]
            else:  # medium
                lo2, hi2 = np.percentile(rts, [40, 60])
                good = [p for p, rt in all_rt60[cat] if lo2 <= rt <= hi2]
        # 重新算下过滤后的 RT60 分布
        good_rts = [rt for p, rt in all_rt60[cat] if p in set(good)]
        if good_rts:
            p = np.percentile(good_rts, [5, 50, 95])
            print(f'  {cat:7s}  筛出 {len(good)}/{len(all_rt60[cat])}  '
                  f'RT60 分布 p5={p[0]:.2f}s / median={p[1]:.2f}s / p95={p[2]:.2f}s')
        filtered[cat] = good

    # 保存 manifest 给 build 脚本读
    manifest = {
        'clean': [],  # 不用 RIR
        'small_room': filtered['small'],
        'medium_room': filtered['medium'],
        'large_room': filtered['large'],
    }
    manifest_path = Path(args.save_manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f'\n[ok] manifest 写入 {manifest_path}')

    # 生成演示音频(clean + 3 类各一段)
    print('\n=== 生成演示音频 ===')
    speech_path = Path(args.speech_sample)
    for label in ['clean', 'small', 'medium', 'large']:
        if label == 'clean':
            outp, _ = make_demo(speech_path, [None], 'clean', out_dir)
            print(f'  clean    → {outp}')
        else:
            rirs = filtered[label] if label != 'clean' else []
            if not rirs:
                print(f'  {label} 无可用 RIR,跳过')
                continue
            # 按 RT60 排序,取中位
            rir_rts = [(p, rt) for p, rt in all_rt60[label] if p in set(rirs)]
            rir_rts.sort(key=lambda x: x[1])
            sorted_rirs = [p for p, _ in rir_rts]
            outp, picked = make_demo(speech_path, sorted_rirs, label, out_dir)
            picked_rt = [rt for p, rt in all_rt60[label] if p == str(picked)][0]
            print(f'  {label:7s} (RT60={picked_rt:.2f}s) → {outp}')

    print(f'\n完成。去听 {out_dir}/*.wav,对比 clean/small/medium/large 的差异')


if __name__ == '__main__':
    main()
