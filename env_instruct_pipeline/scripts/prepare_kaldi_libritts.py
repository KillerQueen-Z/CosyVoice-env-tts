#!/usr/bin/env python3
"""将 LibriTTS 解压目录转为 Kaldi 格式 (wav.scp, text, utt2spk)。
LibriTTS 结构: LibriTTS/<partition>/<speaker_id>/<chapter_id>/*.wav 与 *.normalized.txt
"""
import argparse
import glob
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True, help="LibriTTS 根目录，内含 dev-clean, train-clean-100 等")
    parser.add_argument("--des_dir", type=str, required=True, help="Kaldi 输出根目录，将生成 des_dir/dev-clean 等")
    args = parser.parse_args()
    src = Path(args.src_dir)
    des_root = Path(args.des_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"Not a directory: {src}")
    partitions = [p.name for p in src.iterdir() if p.is_dir()]
    if not partitions:
        raise FileNotFoundError(f"No subdirs (e.g. dev-clean) under {src}")
    for part in sorted(partitions):
        part_src = src / part
        wavs = list(part_src.rglob("*.wav"))
        if not wavs:
            continue
        utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
        for wav in wavs:
            txt = wav.with_suffix(".normalized.txt")
            if not txt.exists():
                continue
            with open(txt, "r", encoding="utf-8") as f:
                content = f.readline().replace("\n", "").strip()
            utt = wav.stem
            spk = utt.split("_")[0]
            utt2wav[utt] = str(wav.resolve())
            utt2text[utt] = content
            utt2spk[utt] = spk
            spk2utt.setdefault(spk, []).append(utt)
        part_des = des_root / part
        part_des.mkdir(parents=True, exist_ok=True)
        with (part_des / "wav.scp").open("w", encoding="utf-8") as f:
            for k, v in sorted(utt2wav.items()):
                f.write(f"{k} {v}\n")
        with (part_des / "text").open("w", encoding="utf-8") as f:
            for k, v in sorted(utt2text.items()):
                f.write(f"{k} {v}\n")
        with (part_des / "utt2spk").open("w", encoding="utf-8") as f:
            for k, v in sorted(utt2spk.items()):
                f.write(f"{k} {v}\n")
        with (part_des / "spk2utt").open("w", encoding="utf-8") as f:
            for spk in sorted(spk2utt):
                f.write(f"{spk} {' '.join(spk2utt[spk])}\n")
        print(f"  {part}: {len(utt2wav)} utts -> {part_des}")


if __name__ == "__main__":
    main()
