#!/usr/bin/env python3
"""下载 CosyVoice3 预训练模型，用于 env-instruct 微调训练。

训练 room/instruct 需要: llm.pt, flow.pt, hifigan.pt, campplus.onnx, speech_tokenizer_v3.onnx, CosyVoice-BlankEN/
这些均包含在 Fun-CosyVoice3-0.5B-2512 中。

用法（在 CosyVoice 仓库根目录）:
  # 国内推荐：Modelscope
  python env_instruct_pipeline/scripts/download_pretrained_cosyvoice3.py --backend modelscope

  # 海外：Huggingface
  python env_instruct_pipeline/scripts/download_pretrained_cosyvoice3.py --backend huggingface

  # 指定保存目录
  python env_instruct_pipeline/scripts/download_pretrained_cosyvoice3.py --out_dir /path/to/pretrained_models/Fun-CosyVoice3-0.5B
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="下载 Fun-CosyVoice3-0.5B 预训练模型")
    parser.add_argument(
        "--backend",
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="modelscope 适合国内；huggingface 适合海外",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="pretrained_models/Fun-CosyVoice3-0.5B",
        help="保存目录，默认 pretrained_models/Fun-CosyVoice3-0.5B",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.backend == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError:
            print("请先安装 modelscope: pip install modelscope")
            sys.exit(1)
        print("使用 Modelscope 下载 Fun-CosyVoice3-0.5B-2512 ...")
        snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=out_dir)
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("请先安装 huggingface_hub: pip install huggingface_hub")
            sys.exit(1)
        print("使用 Huggingface 下载 Fun-CosyVoice3-0.5B-2512 ...")
        snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=out_dir)

    print(f"已下载到: {out_dir}")
    print("训练时设置: export PRETRAINED_DIR=%s" % out_dir)


if __name__ == "__main__":
    main()
