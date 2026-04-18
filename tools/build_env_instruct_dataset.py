#!/usr/bin/env python3
# Build an env-instruct Kaldi-style data dir from a clean Kaldi dir
# by convolving with RIRs bucketed into clean/small/medium/large room.
#
# Inputs:
#   --src_dir : Kaldi dir with wav.scp/text/utt2spk
#   --rir_dir : path to RIRS_NOISES (or a parent containing it)
#   --out_dir : Kaldi output dir (writes wav.scp/text/utt2spk/spk2utt/instruct + wavs/)
#
# Outputs per utterance are resampled to --target_sr and written as 16-bit WAV.
# Category assignment is round-robin over [clean, small_room, medium_room, large_room].

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio


COARSE_INSTRUCTS = {
    "clean": [
        "干声录音，无混响。",
        "清晰的无混响人声。",
    ],
    "small_room": [
        "在小房间里说话，混响较轻。",
        "小空间内，有短促的混响。",
    ],
    "medium_room": [
        "在中等大小房间内说话，混响适中。",
        "中等空间内，有明显的房间混响。",
    ],
    "large_room": [
        "在大型厅堂内说话，混响较长。",
        "开阔空间内的长混响人声。",
    ],
}

# English counterparts — keep same length/slot order as Chinese so pairs align.
COARSE_INSTRUCTS_EN = {
    "clean": [
        "Dry recording, no reverberation.",
        "Clear dry vocal with no reverb.",
    ],
    "small_room": [
        "Speaking in a small room with light reverb.",
        "Small space with short reverb.",
    ],
    "medium_room": [
        "Speaking in a medium-sized room with moderate reverb.",
        "Medium space with noticeable room reverb.",
    ],
    "large_room": [
        "Speaking in a large hall with long reverb.",
        "Open space with long reverberant vocal.",
    ],
}

CATEGORIES = ["clean", "small_room", "medium_room", "large_room"]
RIR_SUBDIR = {
    "small_room": "smallroom",
    "medium_room": "mediumroom",
    "large_room": "largeroom",
}


def resolve_rirs_root(p: Path) -> Path:
    if p.name == "RIRS_NOISES":
        return p
    cand = p / "RIRS_NOISES"
    if cand.exists():
        return cand
    return p


def collect_rirs(rirs_root: Path, manifest_path: Path = None) -> Dict[str, List[Path]]:
    """收集 RIR。若给了 manifest(由 rt60_filter_rirs.py 生成),按 RT60 过滤后的列表;
    否则扫 simulated_rirs 下全部。"""
    if manifest_path is not None and manifest_path.exists():
        import json
        m = json.loads(manifest_path.read_text())
        out: Dict[str, List[Path]] = {
            "small_room": [Path(p) for p in m.get("small_room", [])],
            "medium_room": [Path(p) for p in m.get("medium_room", [])],
            "large_room": [Path(p) for p in m.get("large_room", [])],
        }
        for key, lst in out.items():
            if not lst:
                raise RuntimeError(f"manifest 里 {key} 是空的")
        return out
    out: Dict[str, List[Path]] = {k: [] for k in RIR_SUBDIR}
    sim = rirs_root / "simulated_rirs"
    for key, sub in RIR_SUBDIR.items():
        d = sim / sub
        if d.exists():
            out[key] = sorted(d.rglob("*.wav"))
    for key, lst in out.items():
        if not lst:
            raise RuntimeError(f"No RIR wavs under {sim / RIR_SUBDIR[key]}")
    return out


def load_wav(path: str, target_sr: int) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    return wav.squeeze(0).numpy()


def save_wav(path: Path, sr: int, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    torchaudio.save(str(path), torch.from_numpy(x).unsqueeze(0), sr)


def convolve(speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve
    # Trim RIR leading silence / normalize peak to 1 to keep level similar.
    peak = np.max(np.abs(rir)) + 1e-12
    rir = rir / peak
    out = fftconvolve(speech, rir, mode="full")
    return out[: speech.shape[0]]


def read_two_col(p: Path) -> List[Tuple[str, str]]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            k, _, v = line.partition(" ")
            rows.append((k, v))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--rir_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_sr", type=int, default=24000)
    ap.add_argument("--max_utts", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--instruct_lang", choices=["zh", "en", "both"], default="zh",
                    help="instruct 文本语言:zh=中文(默认), en=英文, both=中英随机 50/50")
    ap.add_argument("--rir_manifest", default="",
                    help="RT60 过滤后的 RIR manifest(rt60_filter_rirs.py 生成);不给则用全部")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    wav_scp = read_two_col(src / "wav.scp")
    text_map = dict(read_two_col(src / "text"))
    utt2spk_map = dict(read_two_col(src / "utt2spk"))
    if args.max_utts > 0:
        wav_scp = wav_scp[: args.max_utts]

    rirs_root = resolve_rirs_root(Path(args.rir_dir))
    manifest = Path(args.rir_manifest) if args.rir_manifest else None
    rirs = collect_rirs(rirs_root, manifest)
    if manifest is not None:
        print(f"[build] 使用 RT60 筛选后的 RIR: "
              f"small={len(rirs['small_room'])}, medium={len(rirs['medium_room'])}, large={len(rirs['large_room'])}")

    out_wav_dir = out / "wavs"       # reverb 音频(目标)
    out_clean_dir = out / "wavs_clean"  # clean 音频(token 源)
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    out_clean_dir.mkdir(parents=True, exist_ok=True)

    # entries: (utt_id, reverb_path, clean_path, text, spk, instruct)
    entries: List[Tuple[str, Path, Path, str, str, str]] = []
    for i, (utt_id, wav_path) in enumerate(wav_scp):
        cat = CATEGORIES[i % len(CATEGORIES)]
        text = text_map.get(utt_id, "")
        spk = utt2spk_map.get(utt_id, "spk")
        try:
            clean_speech = load_wav(wav_path, args.target_sr)
        except Exception as e:
            print(f"skip {utt_id}: {e}")
            continue

        # reverb 版本(训练目标 mel 来源)
        if cat != "clean":
            rir_path = random.choice(rirs[cat])
            rir = load_wav(str(rir_path), args.target_sr)
            reverb_speech = convolve(clean_speech, rir)
        else:
            reverb_speech = clean_speech.copy()

        # 归一化
        rpk = np.abs(reverb_speech).max()
        if rpk > 1e-6:
            reverb_speech = reverb_speech / rpk * 0.95
        cpk = np.abs(clean_speech).max()
        if cpk > 1e-6:
            clean_speech = clean_speech / cpk * 0.95

        if args.instruct_lang == "zh":
            pool = COARSE_INSTRUCTS[cat]
        elif args.instruct_lang == "en":
            pool = COARSE_INSTRUCTS_EN[cat]
        else:
            pool = COARSE_INSTRUCTS[cat] + COARSE_INSTRUCTS_EN[cat]
        instruct = random.choice(pool)

        reverb_path = out_wav_dir / f"{utt_id}.wav"
        clean_path = out_clean_dir / f"{utt_id}.wav"
        save_wav(reverb_path, args.target_sr, reverb_speech)
        save_wav(clean_path, args.target_sr, clean_speech)
        entries.append((utt_id, reverb_path.resolve(), clean_path.resolve(), text, spk, instruct))

    with (out / "wav.scp").open("w", encoding="utf-8") as fw, \
         (out / "wav.clean.scp").open("w", encoding="utf-8") as fwc, \
         (out / "text").open("w", encoding="utf-8") as ft, \
         (out / "utt2spk").open("w", encoding="utf-8") as fu, \
         (out / "instruct").open("w", encoding="utf-8") as fi:
        for utt_id, wav_path, clean_path, text, spk, instruct in entries:
            fw.write(f"{utt_id} {wav_path}\n")
            fwc.write(f"{utt_id} {clean_path}\n")
            ft.write(f"{utt_id} {text}\n")
            fu.write(f"{utt_id} {spk}\n")
            fi.write(f"{utt_id} {instruct}\n")

    spk2utt: Dict[str, List[str]] = {}
    for utt_id, _, _, _, spk, _ in entries:
        spk2utt.setdefault(spk, []).append(utt_id)
    with (out / "spk2utt").open("w", encoding="utf-8") as f:
        for spk in sorted(spk2utt):
            f.write(f"{spk} {' '.join(spk2utt[spk])}\n")

    counts: Dict[str, int] = {c: 0 for c in CATEGORIES}
    for i, _ in enumerate(entries):
        counts[CATEGORIES[i % len(CATEGORIES)]] += 1
    print(f"wrote {len(entries)} utts to {out}; per-category: {counts}")


if __name__ == "__main__":
    main()
