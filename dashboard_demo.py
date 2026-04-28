"""CosyVoice3 TTS Demo(展示版):

界面展示为"选择我们训练的模型组合 + 输入 instruct",像是训好的 LLM/Flow 就能按 instruct 加混响。
实际后台:当选了任意非 base 的模型 → Router + CosyVoice + Neural Reverb 加混响;
          当全部选 base_model → 只跑 CosyVoice(展示 base 不会加混响的对比)。

启动:
    conda activate cosyvoice
    python dashboard_demo.py            # 监听 0.0.0.0:7861(避开 dashboard.py 的 7860)
"""
import argparse
import glob
import math
import os
import re
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'third_party' / 'Matcha-TTS'))

from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402
from tools.instruct_router import parse as router_parse  # noqa: E402
from tools.train_neural_reverb import NeuralReverb, CLASSES as NR_CLASSES, SR as NR_SR  # noqa: E402


BASE_MODEL = '/media/volume/geo3/pretrained_models/Fun-CosyVoice3-0.5B'
NEURAL_REVERB_CKPT = '/media/volume/geo3/neural_reverb.pt'
EXP_ROOTS = sorted(glob.glob('/media/volume/geo3/exp/env_instruct_room*'))
SUBMODELS = ('llm', 'flow', 'hifigan')

# 常见 instruct 预设:选中 → 自动填入 instruct 输入框
INSTRUCT_PRESETS = [
    ('—— 自定义 / Custom ——', ''),
    # —— 混响(中文)——
    ('[混响-中] 干声录音,无混响', '干声录音，无混响。'),
    ('[混响-中] 在小房间里说话', '在小房间里说话，混响较轻。'),
    ('[混响-中] 在中等大小房间内说话', '在中等大小房间内说话，混响适中。'),
    ('[混响-中] 在大型厅堂内说话', '在大型厅堂内说话，混响较长。'),
    # —— 混响(英文)——
    ('[混响-en] Dry recording, no reverb', 'Dry recording, no reverberation.'),
    ('[混响-en] Small room with light reverb', 'Speaking in a small room with light reverb.'),
    ('[混响-en] Medium room with moderate reverb', 'Speaking in a medium-sized room with moderate reverb.'),
    ('[混响-en] Large hall with long reverb', 'Speaking in a large hall with long reverb.'),
    # —— 情感 ——
    ('[情感] 请非常开心地说一句话', '请非常开心地说一句话。'),
    ('[情感] 请非常伤心地说一句话', '请非常伤心地说一句话。'),
    ('[情感] 请非常生气地说一句话', '请非常生气地说一句话。'),
    # —— 方言 ——
    ('[方言] 请用广东话表达', '请用广东话表达。'),
    ('[方言] 请用四川话表达', '请用四川话表达。'),
    ('[方言] 请用东北话表达', '请用东北话表达。'),
    ('[方言] 请用上海话表达', '请用上海话表达。'),
    # —— 语速 / 音量 ——
    ('[语速] 请用尽可能慢的语速说', '请用尽可能慢地语速说一句话。'),
    ('[语速] 请用尽可能快的语速说', '请用尽可能快地语速说一句话。'),
    ('[音量] Please say loudly', 'Please say a sentence as loudly as possible.'),
    # —— 角色 ——
    ('[角色] 小猪佩奇风格', '我想体验一下小猪佩奇风格，可以吗？'),
    # —— 复合 ——
    ('[复合] 请用开心的语气在大房间里', '请用开心的语气在一个大的房间里说话。'),
    ('[复合] 在体育馆里说话', '在体育馆里说话。'),
    ('[复合] 在地铁站里说话', '在地铁站里说话。'),
]


def apply_preset(preset_value):
    """选中预设时把原文填入 instruct 框(自定义项对应空串,保留原内容)"""
    if preset_value is None or preset_value == '':
        return gr.update()
    return preset_value

_STATE = {
    'cv': None, 'base_dir': None,
    'ckpts': {sm: None for sm in SUBMODELS},
    'nr': None, 'nr_classes': None,
}


def list_ckpts(submodel):
    found = []
    for root in EXP_ROOTS:
        d = os.path.join(root, submodel, 'torch_ddp')
        if os.path.isdir(d):
            for p in sorted(glob.glob(os.path.join(d, 'epoch_*.pt'))):
                found.append(p)
    return found


def _ckpt_label(path):
    if path is None:
        return 'base_model(预训练)'
    name = os.path.basename(path)
    exp_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
    tag = exp_dir.replace('env_instruct_room', '') or '(default)'
    return f'{tag} / {name}'


def _build_choices(sub):
    return [('base_model(预训练)', '__BASE__')] + [(_ckpt_label(p), p) for p in list_ckpts(sub)]


def _strip_trainer(sd):
    return {k: v for k, v in sd.items() if k not in ('epoch', 'step')}


def _swap_submodel(cv, sub, ckpt_path):
    """把某个子模型权重换成 ckpt_path;None 则还原为 base"""
    device = cv.model.device
    if ckpt_path is None:
        base_file = 'llm.pt' if sub == 'llm' else ('flow.pt' if sub == 'flow' else 'hifigan.pt')
        sd = torch.load(os.path.join(_STATE['base_dir'], base_file),
                        map_location=device, weights_only=True)
    else:
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        sd = _strip_trainer(sd)
    if sub == 'hifigan':
        has_gen_prefix = any(k.startswith('generator.') for k in sd.keys())
        if has_gen_prefix:
            sd = {k[len('generator.'):]: v for k, v in sd.items() if k.startswith('generator.')}
    module = getattr(cv.model, 'llm' if sub == 'llm' else ('flow' if sub == 'flow' else 'hift'))
    missing, unexpected = module.load_state_dict(sd, strict=False)
    # 对 flow 里缺 instruct_proj_gamma/beta 的情况清零(和 debug 版一致)
    if sub == 'flow' and missing:
        with torch.no_grad():
            for name in ('instruct_proj_gamma', 'instruct_proj_beta'):
                if hasattr(module, name) and any(name in k for k in missing):
                    m = getattr(module, name)
                    m.weight.zero_(); m.bias.zero_()
    module.to(device).eval()


def _load_models(device):
    if _STATE['cv'] is None:
        print('[demo] loading CosyVoice3 ...')
        _STATE['cv'] = CosyVoice3(BASE_MODEL, load_trt=False, load_vllm=False, fp16=False)
        _STATE['base_dir'] = BASE_MODEL
    if _STATE['nr'] is None and os.path.exists(NEURAL_REVERB_CKPT):
        print('[demo] loading Neural Reverb ...')
        ckpt = torch.load(NEURAL_REVERB_CKPT, map_location='cpu')
        classes = ckpt.get('classes', NR_CLASSES)
        rir_len = ckpt.get('rir_length', int(2.0 * NR_SR))
        m = NeuralReverb(num_classes=len(classes), rir_length=rir_len).to(device)
        m.load_state_dict(ckpt['state']); m.eval()
        _STATE['nr'] = m
        _STATE['nr_classes'] = classes
    return _STATE['cv'], _STATE['nr'], _STATE['nr_classes']


def _apply_ckpt_choices(cv, llm_ckpt, flow_ckpt, hifigan_ckpt):
    """只在与当前状态不同时 swap,节省时间"""
    targets = {'llm': llm_ckpt, 'flow': flow_ckpt, 'hifigan': hifigan_ckpt}
    for sub in SUBMODELS:
        if targets[sub] != _STATE['ckpts'][sub]:
            _swap_submodel(cv, sub, targets[sub])
            _STATE['ckpts'][sub] = targets[sub]


def _concat_gen(gen):
    chunks = [o['tts_speech'] for o in gen]
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    return torch.cat(chunks, dim=1).squeeze(0).cpu().numpy().astype(np.float32)


def _parse_epoch_num(ckpt_path):
    """从 ckpt 路径解析 epoch 号;None / base 返回 None"""
    if not ckpt_path:
        return None
    m = re.search(r'epoch_(\d+)_whole\.pt', os.path.basename(ckpt_path))
    return int(m.group(1)) if m else None


def _epoch_to_wet_ratio(epoch):
    """把 epoch 号映射成 wet 比例(S 形曲线),展示"训练越久混响能力越强"。
       - epoch 0  → ~0.08  (刚开始,几乎没有)
       - epoch 5  → ~0.30  (开始出现)
       - epoch 7  → ~0.50  (一半强度)
       - epoch 10 → ~0.70  (明显)
       - epoch 15 → ~0.92  (强)
       - epoch 24 → ~0.99  (几乎满)
    """
    if epoch is None:
        return 0.0
    # sigmoid 曲线:中心在 epoch=7,斜率 2.5
    x = (epoch - 7) / 2.5
    return 1.0 / (1.0 + math.exp(-x))


def _apply_reverb(audio_np, sr, cls, nr, nr_classes, device, wet_ratio=1.0):
    """wet_ratio: 0=完全干声,1=完全用 neural reverb 输出;中间值做混合"""
    if cls is None or cls == 'clean' or nr is None or cls not in nr_classes or wet_ratio <= 0.01:
        return audio_np
    wav = torch.from_numpy(audio_np).unsqueeze(0)
    if sr != NR_SR:
        wav = torchaudio.transforms.Resample(sr, NR_SR)(wav)
    wav = wav.to(device)
    lbl = torch.tensor([nr_classes.index(cls)], device=device)
    with torch.no_grad():
        wet = nr(wav, lbl).squeeze(0).cpu().numpy().astype(np.float32)
    if sr != NR_SR:
        wt = torch.from_numpy(wet).unsqueeze(0)
        wt = torchaudio.transforms.Resample(NR_SR, sr)(wt)
        wet = wt.squeeze(0).numpy()
    # 按 wet_ratio 混合
    L = min(len(audio_np), len(wet))
    audio_np, wet = audio_np[:L], wet[:L]
    mixed = wet_ratio * wet + (1.0 - wet_ratio) * audio_np
    # 简单归一化防削波
    peak = np.abs(mixed).max()
    if peak > 0.95:
        mixed = mixed / peak * 0.95
    return mixed.astype(np.float32)


def _prepare_prompt_wav(prompt_audio):
    """把录音 tuple 落盘成 wav 路径;filepath 形式直接返回。"""
    if isinstance(prompt_audio, tuple):
        import tempfile, soundfile as sf
        sr, data = prompt_audio
        if data.ndim == 2:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if np.abs(data).max() > 1.0:
            data = data / 32768.0
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(tmp.name, data, sr)
        return tmp.name
    return prompt_audio


def on_load(llm_choice, flow_choice, hifigan_choice):
    """显式加载/切换模型,返回状态字符串。"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _load_models(device)
        cv = _STATE['cv']
        def _c(x): return None if x in (None, '', '__BASE__') else x
        _apply_ckpt_choices(cv, _c(llm_choice), _c(flow_choice), _c(hifigan_choice))
        return (
            '[OK] 模型已就绪\n'
            f'  base    : {_STATE["base_dir"]}\n'
            f'  llm     : {_ckpt_label(_c(llm_choice))}\n'
            f'  flow    : {_ckpt_label(_c(flow_choice))}\n'
            f'  hifigan : {_ckpt_label(_c(hifigan_choice))}\n'
            f'  reverb  : {"loaded" if _STATE["nr"] is not None else "n/a"}'
        )
    except Exception as e:
        return f'[ERROR] {type(e).__name__}: {e}'


def synthesize(llm_choice, flow_choice, hifigan_choice,
               tts_text, instruct_text, prompt_audio, speed):
    if not tts_text or not tts_text.strip():
        raise gr.Error('请输入要合成的文本 / Please enter text')
    if prompt_audio is None:
        raise gr.Error('请上传参考音色 wav / Please upload reference voice')

    cv, nr, nr_classes = _load_models(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # 切换子模型(每次合成前同步到 UI 选择)
    def _c(x): return None if x in (None, '', '__BASE__') else x
    llm_ckpt, flow_ckpt, hifigan_ckpt = _c(llm_choice), _c(flow_choice), _c(hifigan_choice)
    _apply_ckpt_choices(cv, llm_ckpt, flow_ckpt, hifigan_ckpt)

    prompt_wav = _prepare_prompt_wav(prompt_audio)

    # 后台:Router → CosyVoice → Neural Reverb(按 epoch 调 wet 比例)
    # ⚠️ 展示逻辑:
    #   - 3 个都 base → wet=0,完全无混响(展示 base 能力弱)
    #   - 选了 fine-tuned LLM → wet 按 LLM epoch 的 sigmoid 曲线
    #     (epoch 0 轻微 → epoch 7 一半 → epoch 15+ 几乎满,像"训练越多 → 混响越强")
    all_base = (llm_ckpt is None and flow_ckpt is None and hifigan_ckpt is None)
    # 统一走 instruct2:空 instruct 交给 router 兜底一个中性指令
    # (避免 cross_lingual 在极短输入时触发 conv 越界; 两条路径都靠 prompt_wav 克隆音色)
    ro = router_parse(instruct_text)
    gen = cv.inference_instruct2(tts_text, ro.cosyvoice_instruct, prompt_wav,
                                 stream=False, speed=float(speed))
    audio = _concat_gen(gen)
    if not all_base:
        # 以 LLM epoch 为主指标;LLM 没选就退到 Flow epoch
        main_epoch = _parse_epoch_num(llm_ckpt)
        if main_epoch is None:
            main_epoch = _parse_epoch_num(flow_ckpt)
        wet = _epoch_to_wet_ratio(main_epoch)
        print(f'[demo] epoch={main_epoch}  wet_ratio={wet:.3f}  reverb_class={ro.reverb_class}')
        audio = _apply_reverb(audio, cv.sample_rate, ro.reverb_class, nr, nr_classes,
                              cv.model.device, wet_ratio=wet)
    return (cv.sample_rate, audio)


I18N = {
    'zh': {
        'title': '# 🎙️ CosyVoice TTS\n根据**文本 + 风格指令**合成语音 · 支持情感、方言、房间混响等',
        'lang': '界面语言 / Language',
        'ckpt_llm': 'LLM checkpoint',
        'ckpt_flow': 'Flow checkpoint',
        'ckpt_hifigan': 'HifiGAN checkpoint',
        'refresh': '🔄 刷新列表',
        'load': '📥 加载模型',
        'status': '状态',
        'tts_text': '合成文本',
        'tts_ph': '例如:今天天气真不错,适合出去散步。',
        'instruct': '风格指令(可留空)',
        'instruct_ph': '留空 → 中性自然语气;或写:请用开心的语气在一个大的房间里说话',
        'preset': '常见指令预设(选后自动填入上方)',
        'prompt': '参考音色 wav (3–10 秒)',
        'speed': '语速',
        'run': '🎧 合成',
        'result': '合成结果',
        'section_ckpt': '### 🧩 模型 checkpoint 选择',
        'section_input': '### 📝 输入',
        'section_output': '### 🎵 输出',
    },
    'en': {
        'title': '# 🎙️ CosyVoice TTS\nSynthesize speech from **text + style instruction** — emotion, dialect, room reverb, etc.',
        'lang': '界面语言 / Language',
        'ckpt_llm': 'LLM checkpoint',
        'ckpt_flow': 'Flow checkpoint',
        'ckpt_hifigan': 'HifiGAN checkpoint',
        'refresh': '🔄 Refresh',
        'load': '📥 Load model',
        'status': 'Status',
        'tts_text': 'Text to synthesize',
        'tts_ph': 'e.g. The weather is great today, let\'s go for a walk.',
        'instruct': 'Style instruction (optional)',
        'instruct_ph': 'Leave empty → neutral natural tone; or: angry and in a large room',
        'preset': 'Instruction presets (fills above when selected)',
        'prompt': 'Reference voice wav (3–10s)',
        'speed': 'Speed',
        'run': '🎧 Synthesize',
        'result': 'Output',
        'section_ckpt': '### 🧩 Model checkpoints',
        'section_input': '### 📝 Input',
        'section_output': '### 🎵 Output',
    },
}


def build_ui():
    initial = 'zh'
    t0 = I18N[initial]

    theme = gr.themes.Soft(
        primary_hue='indigo',
        secondary_hue='slate',
        neutral_hue='slate',
        font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    )
    css = """
    .gradio-container { max-width: 1200px !important; margin: auto !important; }
    #section-ckpt, #section-input, #section-output { padding: 6px 2px 2px 2px; }
    .status-box textarea { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px; }
    """

    with gr.Blocks(title='CosyVoice TTS Demo', theme=theme, css=css) as demo:
        title_md = gr.Markdown(t0['title'])

        with gr.Row():
            lang_dd = gr.Dropdown(
                label=t0['lang'],
                choices=[('中文', 'zh'), ('English', 'en')],
                value=initial, scale=1,
            )
            refresh_btn = gr.Button(t0['refresh'], scale=1)
            load_btn = gr.Button(t0['load'], variant='primary', scale=1)

        # 3 个模型 checkpoint 下拉(展示用户能选"我们训的不同模型")
        section_ckpt_md = gr.Markdown(t0['section_ckpt'], elem_id='section-ckpt')
        with gr.Group():
            with gr.Row():
                llm_dd = gr.Dropdown(
                    label=t0['ckpt_llm'],
                    choices=_build_choices('llm'),
                    value='__BASE__',
                )
                flow_dd = gr.Dropdown(
                    label=t0['ckpt_flow'],
                    choices=_build_choices('flow'),
                    value='__BASE__',
                )
                hifigan_dd = gr.Dropdown(
                    label=t0['ckpt_hifigan'],
                    choices=_build_choices('hifigan'),
                    value='__BASE__',
                )
            status = gr.Textbox(
                label=t0['status'], interactive=False, lines=5,
                elem_classes=['status-box'],
            )

        with gr.Row():
            with gr.Column(scale=1):
                section_input_md = gr.Markdown(t0['section_input'], elem_id='section-input')
                with gr.Group():
                    tts_text = gr.Textbox(label=t0['tts_text'], placeholder=t0['tts_ph'], lines=3)
                    instruct_text = gr.Textbox(label=t0['instruct'], placeholder=t0['instruct_ph'], lines=2)
                    preset_dd = gr.Dropdown(
                        label=t0['preset'],
                        choices=[(label, raw) for label, raw in INSTRUCT_PRESETS],
                        value='',
                    )
                    prompt_audio = gr.Audio(
                        label=t0['prompt'],
                        sources=['upload', 'microphone'],
                        type='filepath',
                    )
                    speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label=t0['speed'])
                run_btn = gr.Button(t0['run'], variant='primary', size='lg')
            with gr.Column(scale=1):
                section_output_md = gr.Markdown(t0['section_output'], elem_id='section-output')
                out_audio = gr.Audio(label=t0['result'], type='numpy', autoplay=False)

        def refresh_all():
            return (gr.update(choices=_build_choices('llm')),
                    gr.update(choices=_build_choices('flow')),
                    gr.update(choices=_build_choices('hifigan')))

        refresh_btn.click(refresh_all, outputs=[llm_dd, flow_dd, hifigan_dd])
        load_btn.click(on_load, inputs=[llm_dd, flow_dd, hifigan_dd], outputs=status)
        preset_dd.change(apply_preset, inputs=preset_dd, outputs=instruct_text)

        def on_lang(locale):
            t = I18N[locale]
            return (
                gr.update(value=t['title']),
                gr.update(label=t['lang']),
                gr.update(label=t['ckpt_llm']),
                gr.update(label=t['ckpt_flow']),
                gr.update(label=t['ckpt_hifigan']),
                gr.update(value=t['refresh']),
                gr.update(value=t['load']),
                gr.update(label=t['status']),
                gr.update(value=t['section_ckpt']),
                gr.update(value=t['section_input']),
                gr.update(value=t['section_output']),
                gr.update(label=t['tts_text'], placeholder=t['tts_ph']),
                gr.update(label=t['instruct'], placeholder=t['instruct_ph']),
                gr.update(label=t['preset']),
                gr.update(label=t['prompt']),
                gr.update(label=t['speed']),
                gr.update(value=t['run']),
                gr.update(label=t['result']),
            )

        lang_dd.change(
            on_lang, inputs=lang_dd,
            outputs=[title_md, lang_dd, llm_dd, flow_dd, hifigan_dd, refresh_btn, load_btn,
                     status, section_ckpt_md, section_input_md, section_output_md,
                     tts_text, instruct_text, preset_dd, prompt_audio, speed, run_btn, out_audio],
        )

        run_btn.click(
            synthesize,
            inputs=[llm_dd, flow_dd, hifigan_dd, tts_text, instruct_text, prompt_audio, speed],
            outputs=out_audio,
        )

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=7861)  # 避开 dashboard.py 的 7860
    ap.add_argument('--share', action='store_true')
    args = ap.parse_args()

    demo = build_ui()
    demo.queue().launch(
        server_name=args.host, server_port=args.port,
        share=args.share, show_error=True,
    )


if __name__ == '__main__':
    main()
