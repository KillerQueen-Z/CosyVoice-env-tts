"""诊断:混响信号能不能通过 speech_tokenizer → Flow → HifiGAN 这条链?

做法:
  1. 从训练集挑 4 条音频(clean/small/medium/large 各 1 条,都是真实混响数据)
  2. 把每条音频提 speech_token (绕过 LLM)
  3. 用我们训好的 Flow + base HifiGAN 重建
  4. 保存重建音频 vs 原始混响音频

判断:
  - 如果重建音频有明显混响 → tokens 保留了混响信息 → LLM 是瓶颈,方案 B 值得做
  - 如果重建是干声 → tokenizer 压平了混响 → 只能方案 A(后处理 RIR)

输出: diagnostic_out/{label}_original.wav  和  diagnostic_out/{label}_reconstructed.wav
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'third_party' / 'Matcha-TTS'))

import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice3
from cosyvoice.utils.file_utils import load_wav

BASE_MODEL = '/media/volume/geo3/pretrained_models/Fun-CosyVoice3-0.5B'
FLOW_CKPT = '/media/volume/geo3/exp/env_instruct_room_both/flow/torch_ddp/epoch_8_whole.pt'
DATA_ROOT = Path('env_instruct_pipeline/output/env_instruct_room5000_both/train')
OUT_DIR = Path('diagnostic_out')
OUT_DIR.mkdir(exist_ok=True)


def find_one_per_category():
    wav_scp = {}
    for line in (DATA_ROOT / 'wav.scp').read_text().splitlines():
        if not line.strip():
            continue
        k, _, v = line.partition(' ')
        wav_scp[k] = v
    instruct_map = {}
    for line in (DATA_ROOT / 'instruct').read_text().splitlines():
        if not line.strip():
            continue
        k, _, v = line.partition(' ')
        instruct_map[k] = v

    # 关键词 → 类别
    def cat_of(ins: str) -> str:
        if '干声' in ins or '清晰的无混响' in ins or 'Dry' in ins or 'Clear dry' in ins:
            return 'clean'
        if '小房间' in ins or '小空间' in ins or 'small room' in ins or 'Small space' in ins:
            return 'small'
        if '中等' in ins or 'medium' in ins or 'Medium' in ins:
            return 'medium'
        if '大型厅堂' in ins or '开阔空间' in ins or 'large hall' in ins or 'Open space' in ins:
            return 'large'
        return None

    samples = {}
    for utt, ins in instruct_map.items():
        cat = cat_of(ins)
        if cat and cat not in samples:
            samples[cat] = (utt, wav_scp[utt], ins.strip())
        if len(samples) == 4:
            break
    return samples


def main():
    print('[info] loading CosyVoice3 base ...')
    cv = CosyVoice3(BASE_MODEL, load_trt=False, load_vllm=False, fp16=False)

    # swap fine-tuned flow
    print(f'[info] swapping Flow → {FLOW_CKPT}')
    sd = torch.load(FLOW_CKPT, map_location=cv.model.device, weights_only=True)
    sd = {k: v for k, v in sd.items() if k not in ('epoch', 'step')}
    cv.model.flow.load_state_dict(sd, strict=True)
    cv.model.flow.to(cv.model.device).eval()
    print('[ok] Flow = fine-tuned (_both/flow/epoch_8); HifiGAN = base')

    samples = find_one_per_category()
    print(f'[info] 找到 {len(samples)} 条样本:')
    for cat, (utt, wp, ins) in samples.items():
        print(f'  {cat:7s} | {utt} | {ins}')

    for cat, (utt, wav_path, ins) in samples.items():
        print(f'\n[test] {cat}')
        # 原始混响音频
        orig, orig_sr = torchaudio.load(wav_path)
        orig_out = OUT_DIR / f'{cat}_original.wav'
        torchaudio.save(str(orig_out), orig, orig_sr)
        print(f'  原始: {orig_out}')

        # inference_vc: 把这条音频自己喂给自己做 voice conversion
        # = 提取 token → flow → hifigan → 波形
        # 跳过 LLM,直接检测 token→mel→wav 路径能不能保留混响
        # NOTE: frontend 内部会再 load_wav,所以两个参数都传 filepath
        out_chunks = []
        for result in cv.inference_vc(wav_path, wav_path, stream=False):
            out_chunks.append(result['tts_speech'])
        if out_chunks:
            audio = torch.cat(out_chunks, dim=1)
            recon_out = OUT_DIR / f'{cat}_reconstructed.wav'
            torchaudio.save(str(recon_out), audio, cv.sample_rate)
            print(f'  重建: {recon_out}  ({audio.shape[1] / cv.sample_rate:.2f}s)')
        else:
            print('  [err] no output chunks')

    print('\n完成。比较 diagnostic_out/ 里的 original vs reconstructed:')
    print('  混响保留 → token 能编码混响,瓶颈在 LLM')
    print('  变干声  → token 丢了混响,需换策略')


if __name__ == '__main__':
    main()
