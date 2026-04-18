"""CosyVoice3 可视化 Dashboard
支持在 base_model 与训练产出的 llm.pt checkpoint 之间切换,兼容 instruct2 用法
(文本 + instruct 提示 + 参考音色 wav)。

启动:
    conda activate cosyvoice
    python dashboard.py                    # 监听 0.0.0.0:7860
    python dashboard.py --port 7870 --share
"""
import argparse
import glob
import os
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
from cosyvoice.utils.file_utils import load_wav  # noqa: E402
from tools.instruct_router import parse as router_parse  # noqa: E402
from tools.train_neural_reverb import NeuralReverb, CLASSES as NR_CLASSES, SR as NR_SR  # noqa: E402

DEFAULT_BASE_MODEL = '/media/volume/geo3/pretrained_models/Fun-CosyVoice3-0.5B'
DEFAULT_NEURAL_REVERB_CKPT = '/media/volume/geo3/neural_reverb.pt'
EXP_ROOTS = sorted(glob.glob('/media/volume/geo3/exp/env_instruct_room*')) + [
    str(REPO_ROOT / 'exp/env_instruct_room'),
]
SUBMODELS = ('llm', 'flow', 'hifigan')
PROMPT_SR = 16000  # load_wav target sr for prompt (frontend 会再处理)

_STATE = {
    'cosyvoice': None,
    'base_model_dir': None,
    'ckpts': {sm: None for sm in SUBMODELS},  # None 表示使用 base
    'neural_reverb': None,
    'nr_classes': None,
}


def load_neural_reverb(ckpt_path: str, device):
    """(懒加载)加载 Neural Reverb 模型"""
    if _STATE['neural_reverb'] is not None:
        return _STATE['neural_reverb'], _STATE['nr_classes']
    if not os.path.exists(ckpt_path):
        return None, None
    ckpt = torch.load(ckpt_path, map_location='cpu')
    classes = ckpt.get('classes', NR_CLASSES)
    rir_length = ckpt.get('rir_length', int(2.0 * NR_SR))
    m = NeuralReverb(num_classes=len(classes), rir_length=rir_length).to(device)
    m.load_state_dict(ckpt['state'])
    m.eval()
    _STATE['neural_reverb'] = m
    _STATE['nr_classes'] = classes
    return m, classes


def apply_neural_reverb(clean_audio_np, sr, reverb_class, device, ckpt_path):
    """clean_audio_np: np [T] → 加指定类别混响的 np [T]"""
    if reverb_class is None or reverb_class == 'clean':
        return clean_audio_np
    m, classes = load_neural_reverb(ckpt_path, device)
    if m is None or reverb_class not in classes:
        return clean_audio_np
    wav = torch.from_numpy(clean_audio_np).unsqueeze(0)
    if sr != NR_SR:
        wav = torchaudio.transforms.Resample(sr, NR_SR)(wav)
    wav = wav.to(device)
    lbl = torch.tensor([classes.index(reverb_class)], device=device)
    with torch.no_grad():
        wet = m(wav, lbl).squeeze(0).cpu().numpy().astype(np.float32)
    # 如果采样率不匹配,需要再 resample 回原采样率
    if sr != NR_SR:
        wet_t = torch.from_numpy(wet).unsqueeze(0)
        wet_t = torchaudio.transforms.Resample(NR_SR, sr)(wet_t)
        wet = wet_t.squeeze(0).numpy()
    return wet


def list_ckpts(submodel):
    """列出某个子模型(llm/flow/hifigan)所有可用的 fine-tuned checkpoint"""
    found = []
    for root in EXP_ROOTS:
        d = os.path.join(root, submodel, 'torch_ddp')
        if os.path.isdir(d):
            for p in sorted(glob.glob(os.path.join(d, 'epoch_*.pt'))):
                found.append(p)
    return found


def _strip_trainer_keys(sd):
    """训练保存的 state_dict 里混入了 epoch/step 元数据,需要剔除"""
    return {k: v for k, v in sd.items() if k not in ('epoch', 'step')}


def _load_submodel_weights(cv, submodel, ckpt_path, base_model_dir):
    """把某个子模型(llm/flow/hifigan)的权重换成 ckpt_path 指定的,或还原为 base"""
    device = cv.model.device
    if ckpt_path is None:
        # 还原为 base;hifigan 对应文件是 hifigan.pt(软链到 hift.pt)
        base_file = 'llm.pt' if submodel == 'llm' else ('flow.pt' if submodel == 'flow' else 'hifigan.pt')
        sd = torch.load(os.path.join(base_model_dir, base_file), map_location=device, weights_only=True)
    else:
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        sd = _strip_trainer_keys(sd)

    # hifigan 权重只保留 generator 部分:
    # - 原版预训练:key 带 'generator.' 前缀,直接剥离即可
    # - 训练产物:包含整个 HiFiGan 模块(generator + discriminator + mel_spec_transform),
    #            需要过滤出 'generator.*' 这一部分,丢掉判别器权重
    if submodel == 'hifigan':
        has_gen_prefix = any(k.startswith('generator.') for k in sd.keys())
        if has_gen_prefix:
            sd = {k[len('generator.'):]: v for k, v in sd.items() if k.startswith('generator.')}

    module = getattr(cv.model, 'llm' if submodel == 'llm' else ('flow' if submodel == 'flow' else 'hift'))
    # Plan B 引入 Flow 的 instruct_embedding / instruct_proj 新参数,
    # base flow.pt 不含这些 key → 用 strict=False 容忍 missing;
    # 同时把缺失的 instruct_proj 清零,让 instruct bias = 0(等价于禁用 Plan B)
    missing, unexpected = module.load_state_dict(sd, strict=False)
    if submodel == 'flow' and missing:
        with torch.no_grad():
            for name in ('instruct_proj_gamma', 'instruct_proj_beta'):
                if hasattr(module, name) and any(name in k for k in missing):
                    m = getattr(module, name)
                    m.weight.zero_(); m.bias.zero_()
    if unexpected:
        raise RuntimeError(f'unexpected keys loading {submodel}: {unexpected[:3]}...')
    module.to(device).eval()


def load_model(base_model_dir, llm_ckpt, flow_ckpt, hifigan_ckpt):
    """加载(或切换)模型。每个子模型 ckpt 为 None 表示用 base 权重。"""
    need_reinit = (
        _STATE['cosyvoice'] is None
        or _STATE['base_model_dir'] != base_model_dir
    )
    if need_reinit:
        cv = CosyVoice3(base_model_dir, load_trt=False, load_vllm=False, fp16=False)
        _STATE['cosyvoice'] = cv
        _STATE['base_model_dir'] = base_model_dir
        _STATE['ckpts'] = {sm: None for sm in SUBMODELS}

    cv = _STATE['cosyvoice']
    targets = {'llm': llm_ckpt or None, 'flow': flow_ckpt or None, 'hifigan': hifigan_ckpt or None}
    for sm in SUBMODELS:
        if targets[sm] != _STATE['ckpts'][sm]:
            _load_submodel_weights(cv, sm, targets[sm], base_model_dir)
            _STATE['ckpts'][sm] = targets[sm]
    return cv


def _ckpt_label(path):
    if path is None:
        return 'base_model (预训练)'
    name = os.path.basename(path)
    # exp 目录名例如 env_instruct_room_both → 取 _both 作为标签
    exp_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
    tag = exp_dir.replace('env_instruct_room', '') or '(default)'
    return f'{tag} / {name}'


def refresh_ckpt_choices_for(submodel):
    ckpts = list_ckpts(submodel)
    choices = [('base_model (预训练)', '__BASE__')] + [
        (_ckpt_label(p), p) for p in ckpts
    ]
    return gr.update(choices=choices, value='__BASE__')


def on_load(base_dir, llm_choice, flow_choice, hifigan_choice, progress=gr.Progress(track_tqdm=False)):
    try:
        progress(0.1, desc='加载模型中…')
        def _c(x): return None if x in (None, '', '__BASE__') else x
        load_model(base_dir, _c(llm_choice), _c(flow_choice), _c(hifigan_choice))
        progress(1.0, desc='完成')
        status = (f'[OK] 已加载\n'
                  f'  base: {base_dir}\n'
                  f'  llm     : {_ckpt_label(_c(llm_choice))}\n'
                  f'  flow    : {_ckpt_label(_c(flow_choice))}\n'
                  f'  hifigan : {_ckpt_label(_c(hifigan_choice))}')
    except Exception as e:
        status = f'[ERROR] {type(e).__name__}: {e}'
    return status


def _concat_generator(gen):
    """收集 generator 里所有 chunk,拼成完整 numpy 波形"""
    chunks = []
    for out in gen:
        wav = out['tts_speech']  # shape (1, T) CPU tensor
        chunks.append(wav)
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    audio = torch.cat(chunks, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    return audio


CANONICAL_PREFIX = 'You are a helpful assistant. '
CANONICAL_SUFFIX = '<|endofprompt|>'

# 预设 instruct 菜单:每项是 (显示名, 原文). 原文不带前后缀,由格式模式决定如何包装。
INSTRUCT_PRESETS = [
    ('—— 自定义 / Custom ——', ''),
    # ---- 训练用的 8 种混响指令(我们 fine-tuned 模型对它们敏感)----
    ('[训练-中] 干声录音，无混响', '干声录音，无混响。'),
    ('[训练-中] 清晰的无混响人声', '清晰的无混响人声。'),
    ('[训练-中] 在小房间里说话，混响较轻', '在小房间里说话，混响较轻。'),
    ('[训练-中] 小空间内，有短促的混响', '小空间内，有短促的混响。'),
    ('[训练-中] 在中等大小房间内说话，混响适中', '在中等大小房间内说话，混响适中。'),
    ('[训练-中] 中等空间内，有明显的房间混响', '中等空间内，有明显的房间混响。'),
    ('[训练-中] 在大型厅堂内说话，混响较长', '在大型厅堂内说话，混响较长。'),
    ('[训练-中] 开阔空间内的长混响人声', '开阔空间内的长混响人声。'),
    # ---- 训练用的 8 种英文混响指令 ----
    ('[训练-en] Dry recording, no reverberation', 'Dry recording, no reverberation.'),
    ('[训练-en] Clear dry vocal with no reverb', 'Clear dry vocal with no reverb.'),
    ('[训练-en] Speaking in a small room with light reverb', 'Speaking in a small room with light reverb.'),
    ('[训练-en] Small space with short reverb', 'Small space with short reverb.'),
    ('[训练-en] Speaking in a medium-sized room with moderate reverb', 'Speaking in a medium-sized room with moderate reverb.'),
    ('[训练-en] Medium space with noticeable room reverb', 'Medium space with noticeable room reverb.'),
    ('[训练-en] Speaking in a large hall with long reverb', 'Speaking in a large hall with long reverb.'),
    ('[训练-en] Open space with long reverberant vocal', 'Open space with long reverberant vocal.'),
    # ---- base 模型已知敏感的方言/情感/语速指令(共 26 条,精选 10 条)----
    ('[Base-方言] 请用广东话表达', '请用广东话表达。'),
    ('[Base-方言] 请用四川话表达', '请用四川话表达。'),
    ('[Base-方言] 请用东北话表达', '请用东北话表达。'),
    ('[Base-方言] 请用上海话表达', '请用上海话表达。'),
    ('[Base-情感] 请非常开心地说一句话', '请非常开心地说一句话。'),
    ('[Base-情感] 请非常伤心地说一句话', '请非常伤心地说一句话。'),
    ('[Base-情感] 请非常生气地说一句话', '请非常生气地说一句话。'),
    ('[Base-语速] 请用尽可能慢地语速说一句话', '请用尽可能慢地语速说一句话。'),
    ('[Base-语速] 请用尽可能快地语速说一句话', '请用尽可能快地语速说一句话。'),
    ('[Base-音量] Please say a sentence as loudly as possible', 'Please say a sentence as loudly as possible.'),
    ('[Base-角色] 我想体验一下小猪佩奇风格，可以吗？', '我想体验一下小猪佩奇风格，可以吗？'),
]


def _wrap_instruct(raw: str, mode: str) -> str:
    """按 mode 包装 instruct 文本。
    - canonical:  'You are a helpful assistant. {raw}<|endofprompt|>' — 用于 base 模型
    - raw_suffix: '{raw}<|endofprompt|>' — 用于我们 fine-tune 的模型(训练时没前缀)
    - asis:       原样返回,由用户自己写全
    """
    s = (raw or '').strip()
    if not s:
        s = '干声录音，无混响。'  # 兜底
    if mode == 'asis':
        return s
    # 去掉可能重复的前后缀,再统一加
    if s.startswith(CANONICAL_PREFIX):
        s = s[len(CANONICAL_PREFIX):]
    if s.endswith(CANONICAL_SUFFIX):
        s = s[:-len(CANONICAL_SUFFIX)]
    s = s.strip()
    if mode == 'canonical':
        return f'{CANONICAL_PREFIX}{s}{CANONICAL_SUFFIX}'
    else:  # raw_suffix
        return f'{s}{CANONICAL_SUFFIX}'


def apply_preset(preset_value):
    """选择预设时,把原文填回 instruct 输入框"""
    return preset_value if preset_value else gr.update()


def synthesize(base_dir, llm_choice, flow_choice, hifigan_choice,
               tts_text, instruct_text, instruct_mode, prompt_audio, speed,
               use_neural_reverb, reverb_class_override):
    if not tts_text or not tts_text.strip():
        raise gr.Error('请输入要合成的文本 (tts_text)')
    if prompt_audio is None:
        raise gr.Error('请上传参考音色 wav (prompt audio) — CosyVoice3 的 instruct2 需要参考音色做 zero-shot')

    def _c(x): return None if x in (None, '', '__BASE__') else x
    cv = load_model(base_dir, _c(llm_choice), _c(flow_choice), _c(hifigan_choice))

    # 关键:如果 3 个子模型都是 base_model,强制禁用 Neural Reverb
    # 这样"纯 base"模式下只有 CosyVoice 原本的能力,公平展示它不会混响的缺陷
    all_base = all(_c(x) is None for x in (llm_choice, flow_choice, hifigan_choice))
    if all_base and use_neural_reverb:
        use_neural_reverb = False
        print('[info] 全 base_model — Neural Reverb 已强制禁用(展示纯 CosyVoice 能力)')

    # frontend 内部会自己调 load_wav 读 24k,所以这里必须传文件路径
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
        prompt_wav = tmp.name
    else:
        prompt_wav = prompt_audio

    # Stage 1: Instruct Router —— 如果启用模块化混响,用 router 决定 cosyvoice 的 instruct 和混响类
    if use_neural_reverb:
        ro = router_parse(instruct_text)
        # 手动覆盖优先级:auto 用 router 结果,其他直接用用户选择
        if reverb_class_override and reverb_class_override != 'auto':
            reverb_cls = None if reverb_class_override == 'none' else reverb_class_override
        else:
            reverb_cls = ro.reverb_class
        final_instruct = ro.cosyvoice_instruct
        debug_info = f'[Router] reverb={reverb_cls} / {ro.detected}'
    else:
        # 传统路径:用户的原文 + mode 包装
        final_instruct = _wrap_instruct(instruct_text, instruct_mode)
        reverb_cls = None
        debug_info = f'[Legacy] instruct={final_instruct!r}'

    print(debug_info)

    # Stage 2: CosyVoice TTS
    gen = cv.inference_instruct2(tts_text, final_instruct, prompt_wav, stream=False, speed=float(speed))
    audio = _concat_generator(gen)

    # Stage 3: Neural Reverb(如果启用)
    if use_neural_reverb and reverb_cls is not None:
        audio = apply_neural_reverb(audio, cv.sample_rate, reverb_cls, cv.model.device, DEFAULT_NEURAL_REVERB_CKPT)

    return (cv.sample_rate, audio)


I18N = {
    'zh': {
        'title': '## CosyVoice3 TTS Dashboard\n在 **base_model** 与训练产出的 **llm.pt** 间切换,支持 instruct2 (文本 + 风格指令 + 参考音色)。',
        'lang': '界面语言 / Language',
        'base_dir': 'Base model 目录',
        'ckpt_llm': 'LLM checkpoint',
        'ckpt_flow': 'Flow checkpoint',
        'ckpt_hifigan': 'HifiGAN checkpoint',
        'refresh': '刷新列表',
        'load': '加载模型',
        'status': '状态',
        'tts_text': '合成文本 (tts_text)',
        'tts_ph': '例如:今天天气真不错，适合出去散步。',
        'instruct': '风格指令原文 (instruct_text) — 只写核心指令,格式由下方"格式模式"决定',
        'instruct_ph': '例如:小空间内，有短促的混响。 / 请非常开心地说一句话。',
        'preset': '预设指令(选后自动填入上方)',
        'mode': '格式模式',
        'mode_canonical': 'Canonical — 加 "You are a helpful assistant. " 前缀 + <|endofprompt|> (Base 模型用)',
        'mode_raw': 'Raw — 只加 <|endofprompt|> (Fine-tuned 模型用)',
        'mode_asis': 'As-is — 原样提交,不做任何包装',
        'prompt': '参考音色 wav (3–10 秒,zero-shot 必需)',
        'speed': '语速',
        'run': '合成',
        'result': '合成结果',
        'base_label': 'base_model (预训练)',
    },
    'en': {
        'title': '## CosyVoice3 TTS Dashboard\nSwitch between the **base model** and fine-tuned **llm.pt** checkpoints. Supports instruct2 (text + style instruction + reference voice).',
        'lang': '界面语言 / Language',
        'base_dir': 'Base model directory',
        'ckpt_llm': 'LLM checkpoint',
        'ckpt_flow': 'Flow checkpoint',
        'ckpt_hifigan': 'HifiGAN checkpoint',
        'refresh': 'Refresh',
        'load': 'Load model',
        'status': 'Status',
        'tts_text': 'Text to synthesize (tts_text)',
        'tts_ph': 'e.g. The weather is great today, let\'s go for a walk.',
        'instruct': 'Style instruction (core text only) — wrapping is decided by Format mode below',
        'instruct_ph': 'e.g. Small space with short reverb. / Please speak happily.',
        'preset': 'Instruction preset (fills above)',
        'mode': 'Format mode',
        'mode_canonical': 'Canonical — add "You are a helpful assistant. " prefix + <|endofprompt|> (for Base)',
        'mode_raw': 'Raw — append only <|endofprompt|> (for fine-tuned)',
        'mode_asis': 'As-is — submit verbatim, no wrapping',
        'prompt': 'Reference voice wav (3–10s, required for zero-shot)',
        'speed': 'Speed',
        'run': 'Synthesize',
        'result': 'Output',
        'base_label': 'base_model (pretrained)',
    },
}


def _build_choices(locale, submodel):
    t = I18N[locale]
    ckpts = list_ckpts(submodel)
    return [(t['base_label'], '__BASE__')] + [(_ckpt_label(p), p) for p in ckpts]


def build_ui(default_base):
    initial_locale = 'zh'
    t0 = I18N[initial_locale]

    with gr.Blocks(title='CosyVoice3 TTS Dashboard') as demo:
        title_md = gr.Markdown(t0['title'])

        with gr.Row():
            lang_dd = gr.Dropdown(
                label=t0['lang'],
                choices=[('中文', 'zh'), ('English', 'en')],
                value=initial_locale,
                scale=1,
            )
            base_dir = gr.Textbox(label=t0['base_dir'], value=default_base, scale=3)
            refresh_btn = gr.Button(t0['refresh'], scale=1)
            load_btn = gr.Button(t0['load'], variant='primary', scale=1)

        with gr.Row():
            llm_dd = gr.Dropdown(
                label=t0['ckpt_llm'],
                choices=_build_choices(initial_locale, 'llm'),
                value='__BASE__',
            )
            flow_dd = gr.Dropdown(
                label=t0['ckpt_flow'],
                choices=_build_choices(initial_locale, 'flow'),
                value='__BASE__',
            )
            hifigan_dd = gr.Dropdown(
                label=t0['ckpt_hifigan'],
                choices=_build_choices(initial_locale, 'hifigan'),
                value='__BASE__',
            )

        status = gr.Textbox(label=t0['status'], interactive=False, lines=4)

        with gr.Row():
            with gr.Column(scale=1):
                tts_text = gr.Textbox(label=t0['tts_text'], placeholder=t0['tts_ph'], lines=4)
                instruct_text = gr.Textbox(label=t0['instruct'], placeholder=t0['instruct_ph'], lines=2)
                preset_dd = gr.Dropdown(
                    label=t0['preset'],
                    choices=[(label, raw) for label, raw in INSTRUCT_PRESETS],
                    value='',
                )
                mode_dd = gr.Radio(
                    label=t0['mode'],
                    choices=[
                        (t0['mode_canonical'], 'canonical'),
                        (t0['mode_raw'], 'raw_suffix'),
                        (t0['mode_asis'], 'asis'),
                    ],
                    value='canonical',
                )
                prompt_audio = gr.Audio(
                    label=t0['prompt'],
                    sources=['upload', 'microphone'],
                    type='filepath',
                )
                speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label=t0['speed'])
                # —— 模块化混响 pipeline ——
                use_reverb = gr.Checkbox(
                    label='启用 Neural Reverb 后处理(模块化 pipeline)',
                    value=True,
                )
                reverb_override = gr.Dropdown(
                    label='混响类(auto=由 Router 自动识别;可手动覆盖)',
                    choices=[
                        ('auto(自动识别)', 'auto'),
                        ('none(不加混响)', 'none'),
                        ('clean', 'clean'),
                        ('small(小房间)', 'small'),
                        ('medium(中房间)', 'medium'),
                        ('large(大厅)', 'large'),
                    ],
                    value='auto',
                )
                router_preview = gr.Textbox(
                    label='Router 预览(实时显示识别结果)',
                    interactive=False, lines=2,
                )
                run_btn = gr.Button(t0['run'], variant='primary')
            with gr.Column(scale=1):
                out_audio = gr.Audio(label=t0['result'], type='numpy', autoplay=False)

        # 预设选中 → 填入 instruct 文本框
        preset_dd.change(apply_preset, inputs=preset_dd, outputs=instruct_text)

        # Router 实时预览:instruct 变化时更新
        def _router_preview(text):
            if not text or not text.strip():
                return '(输入 instruct 后此处会显示 Router 识别结果)'
            ro = router_parse(text)
            return (f'reverb_class: {ro.reverb_class or "(无)"}\n'
                    f'cosyvoice_instruct: {ro.cosyvoice_instruct}\n'
                    f'source: {ro.detected.get("source", "?")}')
        instruct_text.change(_router_preview, inputs=instruct_text, outputs=router_preview)

        # 3 个 checkpoint 下拉变化时,检查是否全是 base_model → disable Neural Reverb
        def _reverb_state(llm_v, flow_v, hifigan_v):
            all_base = all(v in (None, '', '__BASE__') for v in (llm_v, flow_v, hifigan_v))
            if all_base:
                return (gr.update(value=False, interactive=False,
                                  label='启用 Neural Reverb 后处理 — 全 base_model 时不可用'),
                        gr.update(interactive=False))
            return (gr.update(interactive=True,
                              label='启用 Neural Reverb 后处理(模块化 pipeline)'),
                    gr.update(interactive=True))

        for dd in (llm_dd, flow_dd, hifigan_dd):
            dd.change(_reverb_state, inputs=[llm_dd, flow_dd, hifigan_dd],
                      outputs=[use_reverb, reverb_override])

        def on_lang_change(locale):
            t = I18N[locale]
            return (
                gr.update(value=t['title']),
                gr.update(label=t['lang']),
                gr.update(label=t['base_dir']),
                gr.update(label=t['ckpt_llm'], choices=_build_choices(locale, 'llm')),
                gr.update(label=t['ckpt_flow'], choices=_build_choices(locale, 'flow')),
                gr.update(label=t['ckpt_hifigan'], choices=_build_choices(locale, 'hifigan')),
                gr.update(value=t['refresh']),
                gr.update(value=t['load']),
                gr.update(label=t['status']),
                gr.update(label=t['tts_text'], placeholder=t['tts_ph']),
                gr.update(label=t['instruct'], placeholder=t['instruct_ph']),
                gr.update(label=t['preset']),
                gr.update(label=t['mode'], choices=[
                    (t['mode_canonical'], 'canonical'),
                    (t['mode_raw'], 'raw_suffix'),
                    (t['mode_asis'], 'asis'),
                ]),
                gr.update(label=t['prompt']),
                gr.update(label=t['speed']),
                gr.update(value=t['run']),
                gr.update(label=t['result']),
            )

        lang_dd.change(
            on_lang_change,
            inputs=lang_dd,
            outputs=[title_md, lang_dd, base_dir, llm_dd, flow_dd, hifigan_dd,
                     refresh_btn, load_btn, status, tts_text, instruct_text,
                     preset_dd, mode_dd, prompt_audio,
                     speed, run_btn, out_audio],
        )

        def refresh_all(locale):
            return (
                gr.update(choices=_build_choices(locale, 'llm'), value='__BASE__'),
                gr.update(choices=_build_choices(locale, 'flow'), value='__BASE__'),
                gr.update(choices=_build_choices(locale, 'hifigan'), value='__BASE__'),
            )

        refresh_btn.click(refresh_all, inputs=lang_dd, outputs=[llm_dd, flow_dd, hifigan_dd])
        load_btn.click(on_load, inputs=[base_dir, llm_dd, flow_dd, hifigan_dd], outputs=status)
        run_btn.click(
            synthesize,
            inputs=[base_dir, llm_dd, flow_dd, hifigan_dd,
                    tts_text, instruct_text, mode_dd, prompt_audio, speed,
                    use_reverb, reverb_override],
            outputs=out_audio,
        )
        demo.load(refresh_all, inputs=lang_dd, outputs=[llm_dd, flow_dd, hifigan_dd])

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_model', default=DEFAULT_BASE_MODEL)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=7860)
    ap.add_argument('--share', action='store_true')
    args = ap.parse_args()

    demo = build_ui(args.base_model)
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == '__main__':
    main()
