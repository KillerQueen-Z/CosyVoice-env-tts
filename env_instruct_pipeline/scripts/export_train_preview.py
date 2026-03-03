#!/usr/bin/env python3
"""导出 train 前 50 条为 JSON 预览。"""
import json
import sys
from pathlib import Path


def load_map(path):
    m = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split(maxsplit=1)
            m[k] = v
    return m


def main():
    root = Path(sys.argv[1]) / "train"
    wav_scp = root / "wav.scp"
    text_f = root / "text"
    inst_f = root / "instruct"
    utt2spk_f = root / "utt2spk"

    utt2wav = load_map(wav_scp)
    utt2text = load_map(text_f)
    utt2inst = load_map(inst_f)
    utt2spk = load_map(utt2spk_f)

    out_path = root / "train_preview.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for i, utt in enumerate(sorted(utt2wav.keys())):
            if i >= 50:
                break
            rec = {
                "utt": utt,
                "wav": utt2wav[utt],
                "text": utt2text.get(utt, ""),
                "instruct": utt2inst.get(utt, ""),
                "spk": utt2spk.get(utt, ""),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("JSON 预览写入:", out_path.resolve())


if __name__ == "__main__":
    main()
