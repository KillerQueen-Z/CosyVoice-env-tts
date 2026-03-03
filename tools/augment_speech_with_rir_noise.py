#!/usr/bin/env python3
# Copyright (c) 2026
#
# Augment clean speech with open-source RIR and noise (e.g. OpenSLR 28).
# Reads a Kaldi-style data dir, convolves with random RIR, optionally adds
# noise at given SNR, and writes a new data dir with 'instruct' for CosyVoice.
#
# Prerequisite: download and extract OpenSLR 28, e.g.:
#   wget https://www.openslr.org/resources/28/rirs_noises.zip
#   unzip rirs_noises.zip   # -> RIRS_NOISES/
# Then point --rir_noise_dir to the path containing RIRS_NOISES (or to RIRS_NOISES itself).

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio


def _load_wav(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    return wav.squeeze(0).numpy(), target_sr


def _save_wav(path: Path, sr: int, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    torchaudio.save(str(path), torch.from_numpy(x).unsqueeze(0), sr)


def _convolve_rir(speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """FFT convolution: speech * rir. Both 1D float."""
    from scipy.signal import fftconvolve
    out = fftconvolve(speech, rir, mode="full")
    return out[: speech.shape[0]]


def _add_noise_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix noise with speech at given SNR (dB)."""
    n = speech.shape[0]
    if noise.shape[0] < n:
        noise = np.tile(noise, int(np.ceil(n / noise.shape[0])))[:n]
    else:
        noise = noise[:n]
    ps = np.mean(speech ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(ps / pn * 10 ** (-snr_db / 10.0))
    return speech + scale * noise


def _collect_rirs_and_noises(rir_noise_dir: Path) -> Tuple[List[Path], List[Path], List[str], List[str]]:
    """Returns (rir_paths, noise_paths, rir_instruct_templates, noise_instruct_templates)."""
    base = rir_noise_dir
    if base.name == "RIRS_NOISES":
        rirs_root = base
        noises_root = base
    else:
        rirs_root = base / "RIRS_NOISES"
        noises_root = base / "RIRS_NOISES"
    if not rirs_root.exists():
        raise FileNotFoundError(f"RIRS_NOISES not found under {rir_noise_dir}. Extract rirs_noises.zip first.")

    rir_paths: List[Path] = []
    real_dir = rirs_root / "real_rirs_isotropic_noises"
    if real_dir.exists():
        for f in real_dir.rglob("*.wav"):
            if "rir" in f.stem.lower():
                rir_paths.append(f)
    sim_dir = rirs_root / "simulated_rirs"
    if sim_dir.exists():
        for room in ("smallroom", "mediumroom", "largeroom"):
            d = sim_dir / room
            if d.exists():
                for f in d.rglob("*.wav"):
                    rir_paths.append(f)

    noise_paths: List[Path] = []
    pt_dir = noises_root / "pointsource_noises"
    if pt_dir.exists():
        for f in pt_dir.rglob("*.wav"):
            noise_paths.append(f)
    if real_dir.exists():
        for f in real_dir.rglob("*.wav"):
            if "noise" in f.stem.lower():
                noise_paths.append(f)

    rir_tpl = [
        "模拟在真实房间内的混响效果。",
        "模拟在小型房间内说话，有轻微混响。",
        "模拟在中等大小房间内，混响适中。",
        "模拟在较大空间内说话，混响较长。",
    ]
    noise_tpl = [
        "带背景环境噪声，模拟室内/室外杂声。",
        "模拟在嘈杂环境中说话，有环境噪声。",
    ]
    return rir_paths, noise_paths, rir_tpl, noise_tpl


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment clean speech with RIR + noise for CosyVoice instruct data.")
    parser.add_argument("--clean_data_dir", type=str, required=True, help="Kaldi-style dir: wav.scp, text, utt2spk")
    parser.add_argument("--rir_noise_dir", type=str, required=True, help="Path to RIRS_NOISES (or parent of it)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output Kaldi dir with wav.scp, text, utt2spk, instruct")
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--snr_db", type=float, default=15.0, help="SNR when adding noise (dB)")
    parser.add_argument("--noise_prob", type=float, default=0.6, help="Probability to add noise after reverb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_utts", type=int, default=-1, help="Max utterances to process (-1 = all)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    clean_dir = Path(args.clean_data_dir)
    out_dir = Path(args.out_dir)
    rir_paths, noise_paths, rir_tpl, noise_tpl = _collect_rirs_and_noises(Path(args.rir_noise_dir))

    if not rir_paths:
        raise RuntimeError("No RIR files found. Check --rir_noise_dir and RIRS_NOISES layout.")
    if not noise_paths:
        print("Warning: no noise files found; only reverb will be applied.")

    def read_scp(path: Path) -> List[Tuple[str, str]]:
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                out.append((parts[0], parts[1] if len(parts) > 1 else ""))
        return out

    wav_scp = read_scp(clean_dir / "wav.scp")
    text_map = dict(read_scp(clean_dir / "text"))
    utt2spk_map = dict(read_scp(clean_dir / "utt2spk"))
    if not wav_scp:
        raise FileNotFoundError(f"No entries in {clean_dir / 'wav.scp'}")

    if args.max_utts > 0:
        wav_scp = wav_scp[: args.max_utts]

    out_wav_dir = out_dir / "wavs"
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    entries: List[Tuple[str, Path, str, str, str]] = []

    for utt_id, wav_path in wav_scp:
        text = text_map.get(utt_id, "")
        spk = utt2spk_map.get(utt_id, "spk")
        try:
            speech, sr = _load_wav(wav_path, args.target_sr)
        except Exception as e:
            print(f"Skip {utt_id}: load failed: {e}")
            continue

        rir_path = random.choice(rir_paths)
        rir, _ = _load_wav(str(rir_path), args.target_sr)
        speech = _convolve_rir(speech, rir)
        peak = np.abs(speech).max()
        if peak > 1e-6:
            speech = speech / peak * 0.95
        instruct = random.choice(rir_tpl)

        if noise_paths and random.random() < args.noise_prob:
            noise_path = random.choice(noise_paths)
            noise, _ = _load_wav(str(noise_path), args.target_sr)
            speech = _add_noise_at_snr(speech, noise, args.snr_db)
            instruct = instruct + " " + random.choice(noise_tpl)

        out_wav_path = out_wav_dir / f"{utt_id}.wav"
        _save_wav(out_wav_path, args.target_sr, speech)
        entries.append((utt_id, out_wav_path.resolve(), text, spk, instruct))

    with (out_dir / "wav.scp").open("w", encoding="utf-8") as fw, \
         (out_dir / "text").open("w", encoding="utf-8") as ft, \
         (out_dir / "utt2spk").open("w", encoding="utf-8") as fu, \
         (out_dir / "instruct").open("w", encoding="utf-8") as fi:
        for utt_id, wav_path, text, spk, instruct in entries:
            fw.write(f"{utt_id} {wav_path}\n")
            ft.write(f"{utt_id} {text}\n")
            fu.write(f"{utt_id} {spk}\n")
            fi.write(f"{utt_id} {instruct}\n")

    spk2utt: dict = {}
    for utt_id, _, _, spk, _ in entries:
        spk2utt.setdefault(spk, []).append(utt_id)
    with (out_dir / "spk2utt").open("w", encoding="utf-8") as f:
        for spk in sorted(spk2utt):
            f.write(f"{spk} {' '.join(spk2utt[spk])}\n")

    print(f"Wrote {len(entries)} utterances to {out_dir}")


if __name__ == "__main__":
    main()
