"""端到端测试完整的模块化 pipeline:

  instruct 文本
      ↓
  [Stage 1: Instruct Router]         → reverb_class + cosyvoice_instruct
      ↓
  [Stage 2: CosyVoice TTS]           → clean 音频
      ↓
  [Stage 3: Neural Reverb]           → 带混响音频
      ↓
  [Stage 4: Reverb Classifier 评估]  → 判断效果
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'third_party' / 'Matcha-TTS'))

import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice3
from tools.instruct_router import parse as router_parse, CANONICAL_PREFIX, CANONICAL_SUFFIX
from tools.train_neural_reverb import NeuralReverb, CLASSES as NR_CLASSES, SR as NR_SR

BASE_MODEL = '/media/volume/geo3/pretrained_models/Fun-CosyVoice3-0.5B'
NEURAL_REVERB_CKPT = '/media/volume/geo3/neural_reverb.pt'
CLASSIFIER_CKPT = '/media/volume/geo3/reverb_classifier.pt'
PROMPT_WAV = '/home/exouser/CosyVoice-env-tts/asset/zero_shot_prompt.wav'
OUT_DIR = Path('pipeline_out')
OUT_DIR.mkdir(exist_ok=True)

TESTS = [
    ('case1', '今天天气真不错，适合出去散步。', '干声录音，无混响。'),
    ('case2', '今天天气真不错，适合出去散步。', '在小房间里说话，混响较轻。'),
    ('case3', '今天天气真不错，适合出去散步。', '在中等大小房间内说话，混响适中。'),
    ('case4', '今天天气真不错，适合出去散步。', '在大型厅堂内说话，混响较长。'),
    # 复合 instruct
    ('case5', '今天天气真不错，适合出去散步。', '请用开心的语气在一个大的房间里说话'),
    # 场景型
    ('case6', '今天天气真不错，适合出去散步。', '在体育馆里说话'),
    ('case7', '今天天气真不错，适合出去散步。', '在地铁站里说话'),
]


def load_neural_reverb(device):
    ckpt = torch.load(NEURAL_REVERB_CKPT, map_location='cpu')
    model = NeuralReverb(num_classes=len(ckpt.get('classes', NR_CLASSES)),
                         rir_length=ckpt.get('rir_length', int(2.0 * NR_SR))).to(device)
    model.load_state_dict(ckpt['state']); model.eval()
    return model, ckpt.get('classes', NR_CLASSES)


def apply_reverb(nr_model, nr_classes, clean_audio_np, sr, reverb_class, device):
    """clean_audio_np: [T] numpy; 返回 [T] numpy"""
    if reverb_class is None or reverb_class == 'clean':
        return clean_audio_np
    # resample 到 NR 需要的采样率
    wav = torch.from_numpy(clean_audio_np).unsqueeze(0)
    if sr != NR_SR:
        wav = torchaudio.transforms.Resample(sr, NR_SR)(wav)
    wav = wav.to(device)
    cls_id = nr_classes.index(reverb_class)
    lbl = torch.tensor([cls_id], device=device)
    with torch.no_grad():
        wet = nr_model(wav, lbl).squeeze(0).cpu().numpy()
    return wet


def load_classifier(device):
    import importlib.util
    spec = importlib.util.spec_from_file_location("rc", "tools/train_reverb_classifier.py")
    rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
    ckpt = torch.load(CLASSIFIER_CKPT, map_location='cpu')
    model = rc.ReverbClassifier().to(device)
    model.load_state_dict(ckpt['state']); model.eval()
    return model, rc, ckpt.get('classes', rc.CLASSES)


def classify(clf, rc, classes, wav_np, sr, device):
    wav = torch.from_numpy(wav_np).unsqueeze(0)
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
        return classes[int(logit.argmax(-1).item())]


def main():
    print('[info] loading CosyVoice3 base...')
    cv = CosyVoice3(BASE_MODEL, load_trt=False, load_vllm=False, fp16=False)
    device = cv.model.device

    print('[info] loading NeuralReverb...')
    nr_model, nr_classes = load_neural_reverb(device)

    print('[info] loading Classifier...')
    clf, rc, clf_classes = load_classifier(device)

    results = []
    for case_id, tts_text, user_instruct in TESTS:
        print(f'\n=== {case_id} ===')
        print(f'  instruct: {user_instruct!r}')

        # Stage 1: Router
        out = router_parse(user_instruct)
        print(f'  [Router] reverb_class={out.reverb_class}, '
              f'cosyvoice_instruct={out.cosyvoice_instruct[:60]}...')

        # Stage 2: CosyVoice 生成干声(或近似干声)
        chunks = []
        for result in cv.inference_instruct2(tts_text, out.cosyvoice_instruct, PROMPT_WAV, stream=False):
            chunks.append(result['tts_speech'])
        audio = torch.cat(chunks, dim=1).squeeze(0).cpu().numpy().astype('float32')
        (OUT_DIR / f'{case_id}_stage2_cosyvoice.wav').write_bytes(b'')  # placeholder
        torchaudio.save(str(OUT_DIR / f'{case_id}_stage2_cosyvoice.wav'),
                        torch.from_numpy(audio).unsqueeze(0), cv.sample_rate)

        # Stage 3: Neural Reverb
        final = apply_reverb(nr_model, nr_classes, audio, cv.sample_rate, out.reverb_class, device)
        torchaudio.save(str(OUT_DIR / f'{case_id}_stage3_final.wav'),
                        torch.from_numpy(final).unsqueeze(0), cv.sample_rate)

        # Stage 4: 评估
        class_stage2 = classify(clf, rc, clf_classes, audio, cv.sample_rate, device)
        class_stage3 = classify(clf, rc, clf_classes, final, cv.sample_rate, device)

        expected = out.reverb_class if out.reverb_class else class_stage2
        pass_flag = (class_stage3 == out.reverb_class) if out.reverb_class else True

        print(f'  [Stage2 分类]: {class_stage2}(CosyVoice 出的干声)')
        print(f'  [Stage3 分类]: {class_stage3}(加完混响)')
        print(f'  [期望/Router]: {out.reverb_class}')
        print(f'  [通过]: {"✓" if pass_flag else "✗"}')

        results.append((case_id, out.reverb_class, class_stage2, class_stage3, pass_flag))

    print('\n\n=== 总结 ===')
    print(f'{"case":>8s} | {"router":>8s} | {"stage2":>8s} | {"stage3":>8s} | pass')
    print('-' * 60)
    n_ok = 0
    for case, rc_cls, s2, s3, ok in results:
        print(f'{case:>8s} | {str(rc_cls):>8s} | {s2:>8s} | {s3:>8s} | {"✓" if ok else "✗"}')
        n_ok += int(ok)
    print(f'\n准确率: {n_ok}/{len(results)} = {100*n_ok/len(results):.0f}%')


if __name__ == '__main__':
    main()
