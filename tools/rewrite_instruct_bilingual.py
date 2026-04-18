#!/usr/bin/env python3
"""将已生成的 env_instruct 数据集的 `instruct` 文件原地改写成中英混合。

不重新卷积音频,只替换文本 — 因为 instruct 只是条件文本,
和音频无关,所以直接改 instruct 文件就能扩充训练样本覆盖到英文。

用法:
    python tools/rewrite_instruct_bilingual.py \
        --data_root env_instruct_pipeline/output/env_instruct_room5000 \
        --mode both --en_ratio 0.5 --seed 42

之后要重做 parquet + 重启训练(Stage 0,1,5)。
"""
import argparse
import random
import shutil
from pathlib import Path

ZH = {
    "clean": [
        "干声录音，无混响。",
        "清晰的无混响人声。",
        "录音棚级别的干净人声。",
        "没有任何空间回声的干声。",
        "完全无混响的录音。",
    ],
    "small_room": [
        "在小房间里说话，混响较轻。",
        "小空间内，有短促的混响。",
        "在一个小房间里的录音。",
        "狭小空间里有轻微回声的人声。",
        "在小房间录制，能听到一点点回声。",
    ],
    "medium_room": [
        "在中等大小房间内说话，混响适中。",
        "中等空间内，有明显的房间混响。",
        "在一个普通房间里录音，带清楚的回声。",
        "中等房间的混响人声。",
        "在一个中等大小的屋子里说话，回声感明显。",
    ],
    "large_room": [
        "在大型厅堂内说话，混响较长。",
        "开阔空间内的长混响人声。",
        "在一个大厅里录音，有很强的回声。",
        "大房间里回荡的人声，混响拖尾很长。",
        "置身于宽敞大厅中，回声绵延不断。",
    ],
}
EN = {
    "clean": [
        "Dry recording, no reverberation.",
        "Clear dry vocal with no reverb.",
        "Studio-quality clean vocal.",
        "Completely dry speech, no room echo.",
        "A perfectly anechoic recording.",
    ],
    "small_room": [
        "Speaking in a small room with light reverb.",
        "Small space with short reverb.",
        "Recorded inside a tiny room, slight echo.",
        "A compact room with a faint room reflection.",
        "Speech in a small enclosed space, brief decay tail.",
    ],
    "medium_room": [
        "Speaking in a medium-sized room with moderate reverb.",
        "Medium space with noticeable room reverb.",
        "Recorded in an ordinary room, clear echo present.",
        "An average-sized room with obvious reverberation.",
        "Speech in a medium chamber, moderate echo tail.",
    ],
    "large_room": [
        "Speaking in a large hall with long reverb.",
        "Open space with long reverberant vocal.",
        "Recorded in a spacious hall with strong echo.",
        "Voice ringing out in a vast room, lengthy reverb tail.",
        "In a cavernous hall, the echo lingers endlessly.",
    ],
}

# Reverse lookup: Chinese句 → category
ZH2CAT = {s: cat for cat, lst in ZH.items() for s in lst}

CANONICAL_PREFIX = "You are a helpful assistant. "
CANONICAL_SUFFIX = "<|endofprompt|>"


def _canonicalize(text: str, enable: bool) -> str:
    """把核心指令包装成 'You are a helpful assistant. {text}<|endofprompt|>'"""
    if not enable:
        return text
    t = text.strip()
    if t.startswith(CANONICAL_PREFIX):
        t = t[len(CANONICAL_PREFIX):]
    if t.endswith(CANONICAL_SUFFIX):
        t = t[:-len(CANONICAL_SUFFIX)]
    return f"{CANONICAL_PREFIX}{t.strip()}{CANONICAL_SUFFIX}"


def _backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists() and path.exists():
        shutil.copyfile(path, bak)
        print(f"[backup] {bak}")


def _read_two_col(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        k, _, v = line.partition(" ")
        rows.append((k, v))
    return rows


def rewrite_split(split_dir: Path, mode: str, en_ratio: float, duplicate: bool, canonical: bool, rng: random.Random):
    inst_path = split_dir / "instruct"
    if not inst_path.exists():
        print(f"[skip] {inst_path} 不存在")
        return

    for fn in ("instruct", "wav.scp", "wav.clean.scp", "text", "utt2spk", "spk2utt"):
        p = split_dir / fn
        if p.exists():
            _backup(p)

    inst_rows = _read_two_col(inst_path)
    wav_rows = _read_two_col(split_dir / "wav.scp") if (split_dir / "wav.scp").exists() else []
    clean_rows = _read_two_col(split_dir / "wav.clean.scp") if (split_dir / "wav.clean.scp").exists() else []
    txt_rows = _read_two_col(split_dir / "text") if (split_dir / "text").exists() else []
    u2s_rows = _read_two_col(split_dir / "utt2spk") if (split_dir / "utt2spk").exists() else []
    wav_map = dict(wav_rows); clean_map = dict(clean_rows); txt_map = dict(txt_rows); u2s_map = dict(u2s_rows)

    if duplicate:
        # 真翻倍:每条音频配一个中文 instruct + 一个英文 instruct,utt_id 加 _zh / _en 后缀
        new_inst, new_wav, new_clean, new_txt, new_u2s = [], [], [], [], []
        stats = {"zh": 0, "en": 0, "unknown": 0}
        for utt, text in inst_rows:
            cat = ZH2CAT.get(text.strip())
            if cat is None:
                stats["unknown"] += 1
                new_inst.append((utt, text))
                continue
            zh_utt = f"{utt}_zh"
            en_utt = f"{utt}_en"
            zh_text = _canonicalize(rng.choice(ZH[cat]), canonical)
            en_text = _canonicalize(rng.choice(EN[cat]), canonical)
            new_inst.append((zh_utt, zh_text)); stats["zh"] += 1
            new_inst.append((en_utt, en_text)); stats["en"] += 1
            for new_utt in (zh_utt, en_utt):
                if utt in wav_map: new_wav.append((new_utt, wav_map[utt]))
                if utt in clean_map: new_clean.append((new_utt, clean_map[utt]))
                if utt in txt_map: new_txt.append((new_utt, txt_map[utt]))
                if utt in u2s_map: new_u2s.append((new_utt, u2s_map[utt]))

        _write_two_col(split_dir / "instruct", new_inst)
        if new_wav: _write_two_col(split_dir / "wav.scp", new_wav)
        if new_clean: _write_two_col(split_dir / "wav.clean.scp", new_clean)
        if new_txt: _write_two_col(split_dir / "text", new_txt)
        if new_u2s:
            _write_two_col(split_dir / "utt2spk", new_u2s)
            # 重建 spk2utt
            from collections import defaultdict
            s2u = defaultdict(list)
            for utt, spk in new_u2s:
                s2u[spk].append(utt)
            with (split_dir / "spk2utt").open("w", encoding="utf-8") as f:
                for spk in sorted(s2u):
                    f.write(f"{spk} {' '.join(s2u[spk])}\n")
        print(f"[done-DUP] {split_dir} — 原 {len(inst_rows)} → 新 {len(new_inst)};zh={stats['zh']} en={stats['en']} unknown={stats['unknown']}")
        return

    # 原地模式:只换一半 instruct 文本,不翻倍
    lines_out = []
    stats = {"zh": 0, "en": 0, "unknown": 0}
    for utt, text in inst_rows:
        cat = ZH2CAT.get(text.strip())
        if cat is None:
            lines_out.append((utt, text)); stats["unknown"] += 1; continue
        if mode == "en":
            new = rng.choice(EN[cat]); stats["en"] += 1
        elif mode == "zh":
            new = rng.choice(ZH[cat]); stats["zh"] += 1
        else:
            if rng.random() < en_ratio:
                new = rng.choice(EN[cat]); stats["en"] += 1
            else:
                new = rng.choice(ZH[cat]); stats["zh"] += 1
        new = _canonicalize(new, canonical)
        lines_out.append((utt, new))
    _write_two_col(inst_path, lines_out)
    print(f"[done] {inst_path} — zh={stats['zh']} en={stats['en']} unknown={stats['unknown']}")


def _write_two_col(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for k, v in rows:
            f.write(f"{k} {v}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="例如 env_instruct_pipeline/output/env_instruct_room5000")
    ap.add_argument("--mode", choices=["zh", "en", "both"], default="both")
    ap.add_argument("--en_ratio", type=float, default=0.5,
                    help="mode=both 时英文占比,默认 0.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--duplicate", action="store_true",
                    help="真翻倍:每条音频生成 _zh/_en 两条 utt,样本总数 ×2")
    ap.add_argument("--no_canonical", dest="canonical", action="store_false",
                    help="关闭 canonical 包装(默认开启:使用 'You are a helpful assistant. XXX<|endofprompt|>' 格式,与官方 base 模型训练格式一致)")
    ap.set_defaults(canonical=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.data_root)
    for split in args.splits:
        rewrite_split(root / split, args.mode, args.en_ratio, args.duplicate, args.canonical, rng)

    print("\n下一步:")
    print(f"  # 1. 重新生成 parquet(instruct 变了所以 parquet 得刷)")
    print(f"  export DATA_ROOT={root}")
    print(f"  export PRETRAINED_DIR=/media/volume/geo3/pretrained_models/Fun-CosyVoice3-0.5B")
    print(f"  stage=0 stop_stage=0 bash env_instruct_pipeline/scripts/run_train_room.sh")
    print(f"  # 2. 继续训练")
    print(f"  stage=5 stop_stage=5 bash env_instruct_pipeline/scripts/run_train_room.sh")


if __name__ == "__main__":
    main()
