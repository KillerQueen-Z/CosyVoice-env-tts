#!/usr/bin/env python3
"""从 env_instruct 的 train/dev 目录生成「instruct–语音」对应 JSON，便于人工检查。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="生成 instruct 与语音对应的 JSON 列表")
    parser.add_argument("--src_dir", type=str, required=True, help="train 或 dev 目录（含 wav.scp, text, instruct）")
    parser.add_argument("--out", type=str, default="", help="输出 JSON 路径，默认 src_dir/instruct_audio_list.json")
    args = parser.parse_args()

    src = Path(args.src_dir)
    out_path = Path(args.out) if args.out else src / "instruct_audio_list.json"

    wav_scp = src / "wav.scp"
    text_f = src / "text"
    instruct_f = src / "instruct"

    for f in (wav_scp, text_f, instruct_f):
        if not f.exists():
            raise FileNotFoundError(f"缺少文件: {f}")

    # utt_id -> wav path (取相对路径便于可读)
    utt2wav = {}
    with wav_scp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                utt2wav[parts[0]] = parts[1].strip()

    utt2text = {}
    with text_f.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                utt2text[parts[0]] = parts[1].strip()

    utt2instruct = {}
    with instruct_f.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                utt2instruct[parts[0]] = parts[1].strip()

    utt_ids = sorted(utt2wav.keys())
    rows = []
    for utt_id in utt_ids:
        wav = utt2wav.get(utt_id, "")
        # 相对路径便于在不同机器上查看
        try:
            wav_rel = str(Path(wav).relative_to(src.resolve().parent))
        except ValueError:
            wav_rel = wav
        rows.append({
            "utt_id": utt_id,
            "wav": wav_rel,
            "text": utt2text.get(utt_id, ""),
            "instruct": utt2instruct.get(utt_id, ""),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"已写入 {len(rows)} 条 -> {out_path}")


if __name__ == "__main__":
    main()
