"""Instruct Router:解析自然语言 instruct,识别两类信号:

1. CosyVoice base 模型已学过的 instruct(方言/情感/语速/音量/角色)
   → 保留并 canonicalize,让 CosyVoice 处理
2. 混响相关 instruct(干声/小房间/中房间/大厅)
   → 提取出来,交给后处理 Neural Reverb 加混响

返回:
  {
      'reverb_class': 'clean' | 'small' | 'medium' | 'large' | None,
      'cosyvoice_instruct': str,   # 给 CosyVoice 的最终 instruct(canonical 格式)
      'detected': dict,             # debug 信息
  }
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

CANONICAL_PREFIX = 'You are a helpful assistant. '
CANONICAL_SUFFIX = '<|endofprompt|>'


# ----------------------- 混响关键词(覆盖训练分布 + 自然语言)-----------------------
REVERB_KEYWORDS = {
    'clean': [
        # 训练用的中文
        '干声', '无混响', '无回声', '无空间反射', '录音棚', '清晰的无混响',
        # 训练用的英文
        'dry recording', 'dry vocal', 'studio-quality', 'studio quality',
        'no room echo', 'no reverberation', 'anechoic', 'no reverb',
        # 自然语言扩展
        '没有回声', '近距离', '清亮', '清晰录音',
        'studio', 'no echo', 'completely dry',
    ],
    'small': [
        '小房间', '小空间', '狭小', '小屋',
        'small room', 'small space', 'tiny room', 'compact room',
        'enclosed space', 'short reverb', 'little room',
        # 真实场景延展
        '卧室', '浴室', '更衣室', '小车里', '电梯',
        'bedroom', 'bathroom', 'closet', 'elevator', 'tiny',
    ],
    'medium': [
        '中等大小房间', '中等空间', '普通房间', '中等房间', '中等大小',
        'medium-sized', 'medium room', 'medium space', 'ordinary room', 'average-sized',
        'medium chamber', 'noticeable room reverb', 'normal room',
        # 真实场景
        '客厅', '办公室', '教室', '会议室', '餐厅',
        'living room', 'office', 'classroom', 'meeting room', 'dining',
    ],
    'large': [
        '大型厅堂', '大厅', '开阔空间', '大房间', '大的房间', '宽敞', '回荡',
        'large hall', 'spacious hall', 'open space', 'vast room',
        'cavernous', 'long reverberant', 'lengthy reverb',
        # 英文自然语言扩展(之前漏了很多)
        'large room', 'big room', 'big hall', 'huge room', 'huge hall',
        'spacious room', 'large space', 'big space', 'wide room',
        # 真实场景
        '体育馆', '礼堂', '走廊', '山洞', '教堂', '车库', '剧院', '广场',
        '地铁站', '车站', '机场', '商场', '展厅', '大楼', '大空间',
        'gym', 'gymnasium', 'cathedral', 'church', 'corridor', 'cave',
        'auditorium', 'theater', 'theatre', 'arena', 'square',
        'subway station', 'train station', 'airport', 'mall', 'lobby',
        'warehouse', 'tunnel', 'stadium',
    ],
}

# 通用 fallback:中文/英文"空间形容词 + 场所"组合
SPATIAL_PATTERNS = [
    # 中文
    (r'(大|宽敞|空旷|开阔)\s*(的)?\s*(房间|屋|厅|空间|地方|场所)', 'large'),
    (r'(中等|普通|一般)\s*(大小)?\s*(的)?\s*(房间|屋|空间|地方)', 'medium'),
    (r'(小|窄|狭|紧凑)\s*(的)?\s*(房间|屋|空间|地方|场所)', 'small'),
    # 英文(大小写不敏感通过 .lower() 保证)
    (r'\b(large|big|huge|spacious|vast|wide|enormous)\s+(room|hall|space|place|area|chamber)\b', 'large'),
    (r'\b(medium|normal|regular|ordinary|average)\s+(room|hall|space|sized|chamber)\b', 'medium'),
    (r'\b(small|tiny|compact|little|narrow|cramped)\s+(room|hall|space|place|chamber)\b', 'small'),
]


# ----------------------- CosyVoice base 模型已知的 instruct ------------------
# 用户写自然语言 → 映射到 CosyVoice 训练时见过的标准格式
# 来源:cosyvoice/utils/common.py 的 instruct_list
COSYVOICE_KNOWN = [
    # ---- 情感 ----
    (['开心', '快乐', '高兴', '愉快', 'happy', 'cheerful', 'joyful'],
     '请非常开心地说一句话。'),
    (['伤心', '难过', '悲伤', 'sad', 'sorrow'],
     '请非常伤心地说一句话。'),
    (['生气', '愤怒', '气愤', 'angry', 'mad'],
     '请非常生气地说一句话。'),
    # ---- 语速 ----
    (['很慢', '极慢', '慢一点', '慢慢地说', 'slowly', 'very slow'],
     '请用尽可能慢地语速说一句话。'),
    (['很快', '快一点', '快速', 'quickly', 'fast', 'rapidly'],
     '请用尽可能快地语速说一句话。'),
    # ---- 音量 ----
    (['大声', '响亮', '吼', 'loud', 'loudly', 'shout'],
     'Please say a sentence as loudly as possible.'),
    (['小声', '轻声', '低声', '安静地', 'soft', 'softly', 'whisper', 'quiet'],
     'Please say a sentence in a very soft voice.'),
    # ---- 方言(挑常见的几种)----
    (['广东话', '粤语', 'cantonese'], '请用广东话表达。'),
    (['东北话', '东北腔'], '请用东北话表达。'),
    (['四川话'], '请用四川话表达。'),
    (['上海话'], '请用上海话表达。'),
    (['河南话'], '请用河南话表达。'),
    (['湖南话'], '请用湖南话表达。'),
    (['陕西话'], '请用陕西话表达。'),
    (['山东话'], '请用山东话表达。'),
    (['闽南话', '台语'], '请用闽南话表达。'),
    # ---- 角色 ----
    (['小猪佩奇', 'peppa pig'], '我想体验一下小猪佩奇风格，可以吗？'),
    (['机器人', 'robot', 'robotic'], '你可以尝试用机器人的方式解答吗？'),
]


# ----------------------- Router -----------------------
@dataclass
class RouterOutput:
    reverb_class: Optional[str]
    cosyvoice_instruct: str
    detected: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"RouterOutput(reverb_class={self.reverb_class!r}, "
                f"cosyvoice_instruct={self.cosyvoice_instruct!r}, detected={self.detected})")


def detect_reverb_class(text: str) -> Optional[str]:
    """关键词匹配:按关键词长度从长到短优先匹配,解决"中等大小房间"误中"小房间"的歧义"""
    t = text.lower()
    # 1) 所有关键词按长度降序(最长的先匹配,最具体的类胜出)
    all_kws = []
    for cls, kws in REVERB_KEYWORDS.items():
        for kw in kws:
            all_kws.append((len(kw), kw, cls))
    all_kws.sort(reverse=True)  # 按长度降序
    for _, kw, cls in all_kws:
        if kw in text or kw in t:
            return cls
    # 2) 兜底正则:"大/中/小 + 房间/空间"(中英正则,大小写不敏感)
    for pat, cls in SPATIAL_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return cls
    return None


def detect_cosyvoice_instruct(text: str) -> Optional[str]:
    """识别 CosyVoice 已知的 instruct,返回标准模板;无匹配返回 None"""
    t = text.lower()
    for keywords, canonical in COSYVOICE_KNOWN:
        for kw in keywords:
            if kw in text or kw in t:
                return canonical
    return None


DEFAULT_NEUTRAL = '请用自然的语气说一句话。'

def parse(text: str, *, force_canonical: bool = True) -> RouterOutput:
    """主入口:解析 instruct 文本。空输入也返回中性指令,避免 CosyVoice 拿到空串崩溃。"""
    text = (text or '').strip()
    if not text:
        # 空 instruct 兜底:给 CosyVoice 一个中性指令,不加任何混响
        core = DEFAULT_NEUTRAL
        if force_canonical:
            cosy = f'{CANONICAL_PREFIX}{core}{CANONICAL_SUFFIX}'
        else:
            cosy = core
        return RouterOutput(
            reverb_class=None,
            cosyvoice_instruct=cosy,
            detected={'source': 'empty_input_default'},
        )

    reverb_cls = detect_reverb_class(text)
    cosy_match = detect_cosyvoice_instruct(text)

    # 决定 cosyvoice_instruct 内容
    if cosy_match is not None:
        # 有 CosyVoice 已知指令 → 用标准模板,让 CosyVoice 发挥
        core = cosy_match
        source = 'cosyvoice_known'
    elif reverb_cls is not None:
        # 只识别到混响 → 给 CosyVoice 一个中性指令(不影响发音),混响后处理来加
        core = '请用自然的语气说一句话。'
        source = 'reverb_only_neutral'
    else:
        # 完全不认识 → 原样传(把 endofprompt 加上)
        # CosyVoice 见到不认识的 instruct,大概率正常发音
        core = text
        source = 'passthrough'

    if force_canonical:
        # 去掉可能已有的前后缀
        if core.startswith(CANONICAL_PREFIX):
            core = core[len(CANONICAL_PREFIX):]
        if core.endswith(CANONICAL_SUFFIX):
            core = core[:-len(CANONICAL_SUFFIX)]
        cosyvoice_instruct = f'{CANONICAL_PREFIX}{core.strip()}{CANONICAL_SUFFIX}'
    else:
        cosyvoice_instruct = core

    return RouterOutput(
        reverb_class=reverb_cls,
        cosyvoice_instruct=cosyvoice_instruct,
        detected={
            'reverb': reverb_cls,
            'cosyvoice_match': cosy_match,
            'source': source,
        },
    )


# ----------------------- CLI 测试 -----------------------
if __name__ == '__main__':
    cases = [
        '在大型厅堂内说话，混响较长。',                # 训练里见过
        '请用开心的语气在一个大的房间里说话',          # 复合
        '小猪佩奇风格,在小房间里',                    # 复合(角色 + 混响)
        '请非常生气地说一句话',                        # 纯 base 已知
        '在体育馆里大声说话',                          # 复合(音量 + 混响场景)
        '在地铁站里说话',                              # 不认识但暗示大空间
        '今天天气真好',                                # 完全不认识
        '请用四川话说话',                              # 纯 base 方言
        '',                                            # 空
    ]
    for c in cases:
        out = parse(c)
        print(f'\n输入: {c!r}')
        print(f'  混响类: {out.reverb_class}')
        print(f'  → CosyVoice: {out.cosyvoice_instruct}')
        print(f'  检测: {out.detected}')
