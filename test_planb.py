"""快速测试 Plan B Flow:同一 prompt + tts_text,换 3 种 instruct,听混响差异。
输出到 planb_test_out/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'third_party' / 'Matcha-TTS'))

import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice3

BASE = '/media/volume/geo3/pretrained_models/Fun-CosyVoice3-0.5B'
LLM_CKPT = '/media/volume/geo3/exp/env_instruct_room_both/llm/torch_ddp/epoch_7_whole.pt'
FLOW_CKPT = "/media/volume/geo3/exp/env_instruct_room_both/flow/torch_ddp/epoch_7_whole.pt"
PROMPT_WAV = '/home/exouser/CosyVoice-env-tts/asset/zero_shot_prompt.wav'
OUT_DIR = Path('planb_v4c_e7_out')
OUT_DIR.mkdir(exist_ok=True)

CANON = 'You are a helpful assistant. {}<|endofprompt|>'
TESTS = [
    ('clean',  '干声录音，无混响。'),
    ('small',  '小空间内，有短促的混响。'),
    ('medium', '在中等大小房间内说话，混响适中。'),
    ('large',  '在大型厅堂内说话，混响较长。'),
]
TTS_TEXT = '今天天气真不错，适合出去散步。'

def _strip(sd):
    return {k: v for k, v in sd.items() if k not in ('epoch', 'step')}

def main():
    print('[info] loading CosyVoice3 (base)...')
    cv = CosyVoice3(BASE, load_trt=False, load_vllm=False, fp16=False)
    print('[ok] base loaded')

    print(f'[info] swap LLM → {LLM_CKPT}')
    sd = torch.load(LLM_CKPT, map_location=cv.model.device, weights_only=True)
    cv.model.llm.load_state_dict(_strip(sd), strict=True)
    cv.model.llm.to(cv.model.device).eval()

    print(f'[info] swap Flow → {FLOW_CKPT}(Plan B)')
    sd = torch.load(FLOW_CKPT, map_location=cv.model.device, weights_only=True)
    cv.model.flow.load_state_dict(_strip(sd), strict=True)
    cv.model.flow.to(cv.model.device).eval()
    print('[ok] Plan B Flow loaded')

    for cat, ins in TESTS:
        instruct = CANON.format(ins)
        print(f'\n[{cat}] instruct = {ins}')
        chunks = []
        for out in cv.inference_instruct2(TTS_TEXT, instruct, PROMPT_WAV, stream=False):
            chunks.append(out['tts_speech'])
        if not chunks:
            print(f'  [err] no output')
            continue
        audio = torch.cat(chunks, dim=1)
        outp = OUT_DIR / f'{cat}.wav'
        torchaudio.save(str(outp), audio, cv.sample_rate)
        # 尾部能量比(混响指标)
        tail = audio[:, -int(0.3 * cv.sample_rate):]
        ratio = tail.pow(2).mean().sqrt().item() / (audio.pow(2).mean().sqrt().item() + 1e-9)
        print(f'  saved: {outp}  {audio.shape[1]/cv.sample_rate:.2f}s  tail/whole={ratio:.3f}')

    print('\n对比听:clean/small/medium/large 是否有递增的混响')

if __name__ == '__main__':
    main()
