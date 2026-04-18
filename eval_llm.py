"""评估 LLM 是否学会按 instruct 生成"含混响信号"的 speech token。

对每条样本:
  路径 A(LLM 预测): text + instruct → trained LLM → tokens_A → Flow → HifiGAN → audio_A
  路径 B(GT 真值):  reverb_audio → speech_tokenizer_v3 → tokens_B → Flow → HifiGAN → audio_B
  指标:
    - reverb_classifier(audio_A) vs 期望类别 → "instruct 准确率"
    - reverb_classifier(audio_B) vs 期望类别 → "上限"(诊断证明 ~70%)
    - audio_A 和 audio_B 类别一致率 → "LLM 模仿 GT token 的程度"

用法:
  python eval_llm.py --llm_ckpt /media/volume/geo3/exp/env_instruct_room_both/llm/torch_ddp/epoch_X_whole.pt
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'third_party' / 'Matcha-TTS'))

import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice3

BASE = '/media/volume/geo3/pretrained_models/Fun-CosyVoice3-0.5B'
DEFAULT_LLM = '/media/volume/geo3/exp/env_instruct_room_both/llm/torch_ddp/epoch_24_whole.pt'
PROMPT_WAV = '/home/exouser/CosyVoice-env-tts/asset/zero_shot_prompt.wav'
CLF_CKPT = '/media/volume/geo3/reverb_classifier.pt'

CANON = 'You are a helpful assistant. {}<|endofprompt|>'
TESTS = [
    # (label, text, instruct, ground_truth_reverb_audio_path)
    ('clean',  'It had no ornamentation, being exceedingly plain in appearance.',
        '干声录音，无混响。',
        '/home/exouser/CosyVoice-env-tts/diagnostic_out/clean_original.wav'),
    ('small',  'My own regiment was in the advance.',
        '小空间内，有短促的混响。',
        '/home/exouser/CosyVoice-env-tts/diagnostic_out/small_original.wav'),
    ('medium', 'I followed with the spade over my shoulder, dragging my snake.',
        '在中等大小房间内说话，混响适中。',
        '/home/exouser/CosyVoice-env-tts/diagnostic_out/medium_original.wav'),
    ('large',  'Before the day closed, the gardener came whistling from his farm work, to look over his pretty charges.',
        '在大型厅堂内说话，混响较长。',
        '/home/exouser/CosyVoice-env-tts/diagnostic_out/large_original.wav'),
]


def _strip(sd):
    return {k: v for k, v in sd.items() if k not in ('epoch', 'step')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--llm_ckpt', default=DEFAULT_LLM)
    ap.add_argument('--out_dir', default='eval_llm_out')
    ap.add_argument('--clf_ckpt', default=CLF_CKPT)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f'[info] loading CosyVoice3 base...')
    cv = CosyVoice3(BASE, load_trt=False, load_vllm=False, fp16=False)
    if args.llm_ckpt and Path(args.llm_ckpt).exists():
        print(f'[info] swap LLM → {args.llm_ckpt}')
        sd = torch.load(args.llm_ckpt, map_location=cv.model.device, weights_only=True)
        cv.model.llm.load_state_dict(_strip(sd), strict=True)
        cv.model.llm.to(cv.model.device).eval()

    # 路径 A:LLM 生成 → Flow → HifiGAN
    print('\n=== Path A: text + instruct → LLM → audio (TTS 全链)===')
    a_files = []
    for label, text, ins, _gt in TESTS:
        chunks = []
        for out in cv.inference_instruct2(text, CANON.format(ins), PROMPT_WAV, stream=False):
            chunks.append(out['tts_speech'])
        audio = torch.cat(chunks, dim=1) if chunks else torch.zeros(1, 1)
        outp = out_dir / f'A_{label}.wav'
        torchaudio.save(str(outp), audio, cv.sample_rate)
        a_files.append((label, outp))
        print(f'  saved A: {outp}')

    # 路径 B:reverb_audio → tokenizer → Flow → HifiGAN(VC 重建,跳过 LLM)
    print('\n=== Path B: reverb_audio → token → Flow → audio(GT 重建,跳过 LLM)===')
    b_files = []
    for label, _text, _ins, gt_audio in TESTS:
        if not Path(gt_audio).exists():
            print(f'  [skip] {label}: gt audio missing')
            continue
        chunks = []
        for out in cv.inference_vc(gt_audio, gt_audio, stream=False):
            chunks.append(out['tts_speech'])
        audio = torch.cat(chunks, dim=1) if chunks else torch.zeros(1, 1)
        outp = out_dir / f'B_{label}.wav'
        torchaudio.save(str(outp), audio, cv.sample_rate)
        b_files.append((label, outp))
        print(f'  saved B: {outp}')

    # 用分类器评分两个路径
    print('\n=== 分类器评分 ===')
    import importlib.util
    spec = importlib.util.spec_from_file_location("rc", "tools/train_reverb_classifier.py")
    rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)

    ckpt = torch.load(args.clf_ckpt, map_location='cpu')
    classes = ckpt.get('classes', rc.CLASSES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = rc.ReverbClassifier().to(device)
    clf.load_state_dict(ckpt['state']); clf.eval()

    def predict(wav_path):
        wav, sr = torchaudio.load(str(wav_path))
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        if sr != rc.SR:
            wav = torchaudio.transforms.Resample(sr, rc.SR)(wav)
        wav = wav / (wav.abs().max() + 1e-9)
        n = int(rc.CROP_SEC * rc.SR)
        if wav.shape[1] >= n:
            s = (wav.shape[1] - n) // 2
            wav = wav[:, s:s+n]
        else:
            wav = torch.nn.functional.pad(wav, (0, n - wav.shape[1]))
        with torch.no_grad():
            logit = clf(wav.to(device))
            pred_idx = int(logit.argmax(-1).item())
            probs = logit.softmax(-1).squeeze(0).tolist()
        return classes[pred_idx], probs

    a_results, b_results = [], []
    print(f'\n{"label":>8s} | {"path A (LLM)":<22s} | {"path B (GT VC)":<22s} | A==expected? | B==expected? | A==B?')
    print('-' * 100)
    for (label_a, fa), (label_b, fb) in zip(a_files, b_files):
        assert label_a == label_b
        pa, _ = predict(fa); pb, _ = predict(fb)
        ok_a = (pa == label_a); ok_b = (pb == label_b); same = (pa == pb)
        a_results.append(ok_a); b_results.append(ok_b)
        print(f'{label_a:>8s} | {pa:<22s} | {pb:<22s} | {"✓" if ok_a else "✗":<12s} | {"✓" if ok_b else "✗":<12s} | {"✓" if same else "✗"}')

    print('\n=== 总结 ===')
    print(f'  Path A 准确率(LLM 生成 vs 期望): {sum(a_results)}/{len(a_results)} = {100*sum(a_results)/len(a_results):.0f}%')
    print(f'  Path B 准确率(GT 重建 vs 期望): {sum(b_results)}/{len(b_results)} = {100*sum(b_results)/len(b_results):.0f}%(诊断上限)')


if __name__ == '__main__':
    main()
