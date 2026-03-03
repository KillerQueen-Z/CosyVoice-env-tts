#!/usr/bin/env python3
# Copyright (c) 2026
#
# Minimal demo dataset generator for CosyVoice instruct conditioning.
# It creates a Kaldi-style data dir (wav.scp/text/utt2spk/instruct/spk2utt)
# with a few synthetic "voice-like" waveforms and simple environment effects
# (cave/room reverb-ish echoes + noise). Replace the synthetic wavs with real
# clean speech to build a practical dataset.

from __future__ import annotations

import argparse
import math
import os
import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _write_wav_mono_16bit(path: Path, sr: int, x: List[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for s in x:
            s = int(round(_clamp(s, -1.0, 1.0) * 32767.0))
            frames += int(s).to_bytes(2, byteorder="little", signed=True)
        wf.writeframes(bytes(frames))


def _hann_env(n: int, fade: int) -> List[float]:
    fade = max(1, min(fade, n // 2))
    env = [1.0] * n
    for i in range(fade):
        a = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / (fade + 1))
        env[i] = a
        env[n - 1 - i] = a
    return env


def _synth_voice_like(sr: int, dur_s: float, f0: float, seed: int) -> List[float]:
    """
    A tiny "voice-like" harmonic stack with an amplitude envelope.
    This is NOT real speech; it's only to produce valid wav files for pipeline demos.
    """
    rnd = random.Random(seed)
    n = int(sr * dur_s)
    env = _hann_env(n, fade=int(0.12 * sr))

    # Add small vibrato so it doesn't sound like a pure tone.
    vib_hz = rnd.uniform(4.5, 6.5)
    vib_depth = rnd.uniform(0.002, 0.01)  # relative

    x: List[float] = [0.0] * n
    phase = 0.0
    for t in range(n):
        tt = t / sr
        f = f0 * (1.0 + vib_depth * math.sin(2.0 * math.pi * vib_hz * tt))
        phase += 2.0 * math.pi * f / sr
        s = 0.0
        # harmonic stack with gentle roll-off
        for k in range(1, 18):
            s += (1.0 / (k ** 1.25)) * math.sin(k * phase)
        # mild waveshaping to create "formant-ish" roughness
        s = math.tanh(1.6 * s)
        x[t] = 0.12 * env[t] * s
    return x


@dataclass
class EchoSpec:
    delay_ms: float
    gain: float


def _apply_multi_tap_echo(sr: int, x: List[float], taps: Iterable[EchoSpec], feedback: float = 0.0) -> List[float]:
    """
    Simple echo / reverb-ish effect using a few delay taps and optional feedback.
    O(N * num_taps).
    """
    n = len(x)
    y = x[:]
    d_samples = [max(1, int(sr * tap.delay_ms / 1000.0)) for tap in taps]
    gains = [tap.gain for tap in taps]
    if feedback != 0.0:
        feedback = _clamp(feedback, 0.0, 0.98)

    # feed-forward taps
    for i in range(n):
        acc = y[i]
        for d, g in zip(d_samples, gains):
            if i - d >= 0:
                acc += g * y[i - d]
        y[i] = acc

    # light feedback to extend tail
    if feedback > 0.0:
        for i in range(1, n):
            y[i] += feedback * y[i - 1]

    # normalize
    peak = max(1e-9, max(abs(s) for s in y))
    scale = 0.95 / peak
    return [s * scale for s in y]


def _add_colored_noise(x: List[float], noise_level: float, seed: int, hp: float = 0.0) -> List[float]:
    """
    Add a simple "colored" noise using 1-pole low-pass (and optional high-pass).
    """
    rnd = random.Random(seed)
    y: List[float] = []
    lp = 0.0
    hp_state = 0.0
    for s in x:
        w = rnd.uniform(-1.0, 1.0)
        # 1-pole lowpass -> brown-ish
        lp = 0.995 * lp + 0.005 * w
        nn = lp
        if hp > 0.0:
            # crude high-pass: remove slow drift
            hp_state = 0.999 * hp_state + 0.001 * nn
            nn = nn - hp * hp_state
        y.append(s + noise_level * nn)
    peak = max(1e-9, max(abs(s) for s in y))
    if peak > 0.99:
        y = [0.99 * s / peak for s in y]
    return y


def _render_env(sr: int, dry: List[float], env: str, seed: int) -> List[float]:
    env = env.lower().strip()
    if env == "clean":
        return dry
    if env == "cave":
        # Long-ish delays + feedback to simulate cave reflections.
        taps = [
            EchoSpec(27.0, 0.42),
            EchoSpec(61.0, 0.31),
            EchoSpec(113.0, 0.22),
            EchoSpec(177.0, 0.16),
        ]
        wet = _apply_multi_tap_echo(sr, dry, taps=taps, feedback=0.25)
        wet = _add_colored_noise(wet, noise_level=0.008, seed=seed + 11)
        return wet
    if env == "room":
        # Shorter early reflections.
        taps = [
            EchoSpec(9.0, 0.28),
            EchoSpec(17.0, 0.19),
            EchoSpec(31.0, 0.13),
        ]
        wet = _apply_multi_tap_echo(sr, dry, taps=taps, feedback=0.10)
        wet = _add_colored_noise(wet, noise_level=0.004, seed=seed + 13)
        return wet
    if env == "street":
        # Mostly noise + slight slapback.
        taps = [EchoSpec(45.0, 0.10)]
        wet = _apply_multi_tap_echo(sr, dry, taps=taps, feedback=0.0)
        wet = _add_colored_noise(wet, noise_level=0.03, seed=seed + 17, hp=0.8)
        return wet
    raise ValueError(f"Unknown env: {env}")


def _write_kaldi_files(
    out_dir: Path,
    entries: List[Tuple[str, Path, str, str, str]],
) -> None:
    """
    entries: (utt, wav_path, text, spk, instruct)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_scp = out_dir / "wav.scp"
    text_f = out_dir / "text"
    utt2spk = out_dir / "utt2spk"
    instruct_f = out_dir / "instruct"
    spk2utt = out_dir / "spk2utt"

    spk_map = {}
    with wav_scp.open("w", encoding="utf-8") as fwav, \
            text_f.open("w", encoding="utf-8") as ftxt, \
            utt2spk.open("w", encoding="utf-8") as f_u2s, \
            instruct_f.open("w", encoding="utf-8") as finst:
        for utt, wav_path, text, spk, instruct in entries:
            fwav.write(f"{utt} {wav_path}\n")
            ftxt.write(f"{utt} {text}\n")
            f_u2s.write(f"{utt} {spk}\n")
            finst.write(f"{utt} {instruct}\n")
            spk_map.setdefault(spk, []).append(utt)

    with spk2utt.open("w", encoding="utf-8") as f:
        for spk, utts in sorted(spk_map.items()):
            f.write(spk + " " + " ".join(utts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="demo_env_instruct_dataset")
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=519)
    args = parser.parse_args()

    sr = args.sr
    root = Path(args.out_dir).resolve()
    rnd = random.Random(args.seed)

    # A tiny, Chinese-focused instruct set. In practice you want many paraphrases.
    samples = [
        ("clean", "请用自然的语气说：今天我们去公园散步。", "自然、干净、无环境音。"),
        ("cave", "请用自然的语气说：我在这里听到了回声。", "模拟在山洞里的声音，有明显回声和空间感。"),
        ("room", "请用自然的语气说：请把门轻轻关上。", "模拟在小房间里说话，轻微混响。"),
        ("street", "请用自然的语气说：前方车辆较多，请注意安全。", "模拟在街道旁说话，有交通环境噪声。"),
        ("cave", "请用低声说：我们不要吵醒它。", "模拟在山洞里低声说话，回声更长。"),
        ("street", "请用稍快的语速说：我们在路口集合。", "模拟在嘈杂街道，背景噪声更明显。"),
    ]

    spk = "spk_demo"
    # split
    rnd.shuffle(samples)
    train = samples[:4]
    dev = samples[4:]

    def build_split(split_name: str, split_samples: List[Tuple[str, str, str]]) -> None:
        entries = []
        wav_dir = root / split_name / "wavs"
        for idx, (env, text, instruct) in enumerate(split_samples):
            utt = f"{split_name}_{idx:04d}_{env}"
            dur = rnd.uniform(2.2, 3.2)
            f0 = rnd.uniform(105.0, 165.0)
            dry = _synth_voice_like(sr, dur_s=dur, f0=f0, seed=args.seed * 1000 + idx)
            wet = _render_env(sr, dry, env=env, seed=args.seed * 1000 + 97 + idx)
            wav_path = wav_dir / f"{utt}.wav"
            _write_wav_mono_16bit(wav_path, sr, wet)
            entries.append((utt, wav_path, text, spk, instruct))
        _write_kaldi_files(root / split_name, entries)

    build_split("train", train)
    build_split("dev", dev)

    print(f"Demo dataset created at: {root}")
    print("Next steps:")
    print(f"  - (Optional) extract embeddings/tokens into {root}/train and {root}/dev")
    print("  - make parquet via tools/make_parquet_list.py")
    print("  - create train.data.list/dev.data.list and run cosyvoice/bin/train.py")


if __name__ == "__main__":
    # Make sure relative paths are stable if executed from repo root.
    os.chdir(Path(__file__).resolve().parents[1])
    main()

