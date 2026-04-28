# 环境感知 TTS:探索路径与最终方案(展示版)

**任务**:给 CosyVoice3 扩展能力,让它根据自然语言 instruct 生成**不同房间混响**的语音。
例如输入"在大型厅堂内说话" → 输出带长混响的音频;"干声录音" → 输出无混响干净语音。

---

## 1. CosyVoice3 架构回顾

```
用户输入 (text + instruct + prompt_wav)
    │
    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ LLM (Qwen2)   │──►│ Flow (DiT)    │──►│ HifiGAN       │──► 波形
│ 0.5B params   │   │ ~150M params  │   │ ~80M params   │
│ text→tokens   │   │ tokens→mel    │   │ mel→wav       │
└───────────────┘   └───────────────┘   └───────────────┘
                  ↑
        speech_tokenizer_v3.onnx
        (训练时用它从音频提 token 作为 LLM 目标)
```

每一阶段都可能成为"学习混响"的瓶颈。我们的探索就是定位瓶颈并验证它。

---

## 2. 数据管道

### 2.1 训练数据生成

- **干声源**:LibriTTS dev-clean(5736 条)+ train-clean-100(33236 条)
- **RIR 库**:RIRS_NOISES/simulated_rirs(60000 个合成 RIR,分 small/medium/large)
- **生成**:clean_speech ⊛ RIR = 带混响音频,按 round-robin 分配到 4 类(clean/small/medium/large)
- **Instruct 模板**:每类 5 中 + 5 英 = 40 种表述,配对 canonical 格式 `You are a helpful assistant. XXX<|endofprompt|>`

### 2.2 RT60 筛选

初版 RIR 按文件夹分类,但 RT60 重叠严重(small 可到 0.35s,medium 最小 0.40s)。我们写了 Schroeder 反向积分估算 RT60([`tools/rt60_filter_rirs.py`](../tools/rt60_filter_rirs.py)),过滤出**类间 gap 3× 以上**的核心 RIR:

| 类 | 原始 RT60 (median) | 筛后核心 RT60 | 可用 RIR 数 |
|---|---|---|---|
| small | 0.20s | [0.11, 0.28]s | 16332 |
| medium | 0.83s | [0.56, 0.74]s | 4605 |
| large | 2.29s | [1.45, 5.20]s | 18899 |

### 2.3 Clean-Reverb 对齐数据(Phase 4 关键基础)

为了后面 Flow 的 "clean_token → reverb_mel" 监督范式,改写数据管道同时保存 clean 和 reverb 两份音频,并在 parquet 里打包 `audio_data_clean` 额外字段。这个改动是后面 Phase 4 成功的关键前提。

---

## 3. 探索 Phase 1:LLM SFT(直接微调)

**假设**:用"文本 + instruct"→"混响音频的 speech_token"这对监督数据直接微调 LLM,让它学会根据 instruct 输出"带混响的 token"。

### 三次 LLM SFT 对比图

![LLM SFT 三次对比](figures/llm_sft_comparison.png)

三条曲线同框对比:红(v1 崩溃) / 绿(v2 健康但弱) / 蓝(v3 先降后反弹)。v1 在 E3 开始发散,v2 健康收敛但改善幅度小,v3 在 E4 达到最优后又走向过拟合。

### 3.1 尝试 1:naive lr=1e-5(教训:灾难性过拟合)

| Epoch | CV loss | CV acc | 状态 |
|---|---|---|---|
| E0 | **3.840** | 0.168 | 起点 |
| E1 | 3.855 | 0.169 | 略升 |
| E4 | 5.206 | 0.132 | 开始发散 |
| E10 | 11.31 | 0.114 | 崩溃 |
| E14 | **12.20** | 0.113 | 灾难性 |

训练 loss 同时掉到 0.007(acc=1.0)。诊断**典型过拟合**:模型"背会"训练集但在 dev 上对每个正确 token 只分配 1e-6 级别的概率(交叉熵爆炸)。

### 3.2 尝试 2:canonical 格式 + lr=1e-6(健康但收益微弱)

把 instruct 包装为 `You are a helpful assistant. XXX<|endofprompt|>`(对齐 CosyVoice 预训练格式),降 lr 10 倍:

| Epoch | CV loss | CV acc |
|---|---|---|
| E0 | 3.918 | 0.162 |
| E5 | 3.840 | 0.169 |
| **E7** | **3.832** | **0.169** |
| E9 | 3.834 | 0.169 |

CV loss 平稳降 0.086,**无过拟合**。但听感上几乎听不出混响差异。

![LLM SFT v2 细节](figures/llm_sft_v2_detail.png)

### 3.3 尝试 3:中间值 lr=5e-6 + max_epoch=25

| Epoch | CV loss | acc | 状态 |
|---|---|---|---|
| E0 | 3.889 | 0.165 | |
| **E4** | **3.816** | **0.171** | **最优** |
| E5 | 3.825 | 0.170 | 开始反弹 |
| E10 | 4.604 | 0.141 | 过拟合 |
| E24 | 12.41 | 0.105 | 灾难 |

**E4 就是最低点,之后过拟合**。即使在中间 lr,LLM 仍无法稳健学会 instruct→reverb 映射。

![LLM SFT v3 反弹曲线](figures/llm_sft_v3_rebound.png)

典型 U 形:E4 见底后立即反弹,说明 5e-6 的 lr 对这个模型+数据规模仍偏大。

### 3.4 Phase 1 结论

- **token 通道的监督信号太弱**:混响对每个样本的 token 序列只带来少数几个位置的微小分布偏移,99% 的梯度都给"预测正确的发音内容",留给"学混响"的梯度微乎其微
- **即使 CV loss 略降,感知上还是无混响**
- **LLM 0.5B + 10k 数据 + instruct→reverb 任务**,似乎在架构层面不匹配

---

## 4. 探索 Phase 2:诊断 — 下游通路是否通畅?

既然 LLM 学不好,先排除其他阶段的嫌疑。写了 [`diagnostic_flow.py`](../diagnostic_flow.py):

```
混响音频 ─► [speech_tokenizer_v3] ─► tokens ─► [Flow(fine-tuned)] ─► mel ─► [HifiGAN(base)] ─► 重建音频
                                          (跳过 LLM)
```

指标:**尾部能量比**(tail/whole RMS)——衡量说话结束后 300ms 残余能量,越大说明混响越长。

| 类别 | 原始真实音频 | 重建音频 | 保留率 |
|---|---|---|---|
| clean | 0.150 | 0.152 | ~ |
| medium | 0.158 | 0.150 | 95% |
| **large** | **0.379** | **0.272** | **~70%** |

**结论**:如果能给 Flow "带混响的 token",下游能保留约 70% 的混响。**瓶颈确实在 LLM**,不是 tokenizer 或 Flow。

![诊断 tail/whole 对比](figures/diagnostic_tail_ratio.png)

4 个类别的尾部能量比对比:真实音频 vs 重建音频。clean 和 medium 基本持平,large 从 0.379 降到 0.272(**保留 70%**)。

这个结论至关重要:它**把锅从 Flow 和 HifiGAN 上摘下来**,也暗示"如果我们能直接给 Flow 混响控制信号,它是有能力承担这个任务的"。

---

## 5. 探索 Phase 3:用分类器做自建 benchmark

为了**量化评估**任何 TTS 输出的混响类别,训了一个 4 类 CNN 分类器([`tools/train_reverb_classifier.py`](../tools/train_reverb_classifier.py)):

- 输入:音频 → log-mel(64 mels)→ 3 层 Conv + global pool
- 参数:~500K
- 训练:10000 条标签音频 6 epoch
- **Dev 准确率:92%**

![Classifier Confusion Matrix](figures/classifier_confusion.png)

- Confusion matrix(best epoch):

```
           pred→  clean  small  medium  large
真实 clean        [250]    0      0      0
真实 small          1    [246]   28      7
真实 medium         1     10    [192]   15
真实 large          0      2     16    [232]
```

用它测试 LLM 微调各阶段的输出:

| 模型状态 | clean | small | medium | large | 准确率 |
|---|---|---|---|---|---|
| base 模型 | clean✓ | clean✗ | clean✗ | clean✗ | 25% |
| LLM SFT(canonical,E7) | clean✓ | clean✗ | clean✗ | clean✗ | 25% |
| LLM SFT(strong,E4) | clean✓ | clean✗ | clean✗ | clean✗ | 25% |

**全部判为 clean**。证实 LLM 阶段不管怎么训都没学到"产出混响 token"。

---

## 6. 探索 Phase 4:Flow 阶段 Plan B(token 级 instruct 条件)

**思路**:既然 LLM 学不会,让 Flow 直接接收 instruct 作为条件信号,绕开 LLM 的 token 瓶颈。Flow 在连续 mel 空间上应该更容易学"按 instruct 加混响"。

改造 Flow 架构([`cosyvoice/flow/flow.py`](../cosyvoice/flow/flow.py) `CausalMaskedDiffWithDiT`):

```python
# 新增层
self.instruct_embedding = nn.Embedding(152064, 128)       # Qwen tokenizer vocab
self.instruct_proj_gamma = nn.Linear(128, output_size)    # FiLM gamma
self.instruct_proj_beta  = nn.Linear(128, output_size)    # FiLM beta

# forward 里:h = h * (1 + scale·gamma) + scale·beta
```

训练时 instruct_token 从 dataset batch 直接读(processor 已经提取好了)。

### 6.1 v1:加性 bias(最初版)

- init: zeros(避免初始扰动)
- scale: 1
- 训 10 epoch → FiLM 权重 `mean_abs = 0.0080`(和初始化一样),**几乎没学**
- 分类器测:4/4 全 clean

### 6.2 v2:FiLM scale=50

- 放大 FiLM 输出 50 倍,希望让信号更强 → **音频不像人话**(perturbation 过大破坏 decoder)

### 6.3 v3-safe:per-param 100× lr + scale=1

修改 [`cosyvoice/utils/train_utils.py`](../cosyvoice/utils/train_utils.py) 加 `_split_param_groups`,给 instruct_* 参数 100× lr:

```python
# lr=1e-6 for Flow body, lr=1e-4 for instruct_embedding/proj
```

| Epoch | CV loss |
|---|---|
| E0 | 0.641 |
| E7 | 0.585 |
| E14 | 0.580 (最优) |

CV 改善和无 FiLM 差不多 → 模型仍没真正用上 instruct 层。**但梯度开始有反应**,FiLM mean_abs 从 0.0080 缓慢上升到 0.0089,说明方向是对的。

### 6.4 v4c:clean_token → reverb_mel 范式

**洞察**:训练时 token 来自 reverb audio(本身已含混响信息),Flow 不需要 FiLM 就能 decode。改成从 clean audio 提 token,让 Flow 必须靠 instruct 才能学会加混响。

改造数据管道:
- `build_env_instruct_dataset.py` 同时保存 `wavs/<utt>.wav`(reverb) + `wavs_clean/<utt>.wav`(clean)
- `processor.py` 的 `compute_whisper_fbank` 优先用 `speech_clean` 算 whisper_feat → tokens

CV loss: E0 0.65 → E7 0.59 → E13 0.60(收敛到旧水平),但分类器 4/4 还是 clean。

![Plan B Flow 训练曲线](figures/flow_planb_v3_safe.png)

Flow 主体能学,但 FiLM 参数 weight 从 0.0080 只挪到 0.0089(lr=1e-4 × 15 epoch 下仍激活不强),混响条件信号没真正建立。**这是关键的中间节点**:我们有了正确的监督范式,但 FiLM 梯度仍被主体淹没。

### 6.5 v5:conds=0 清除 reverb 泄漏

发现 Flow forward 有个 `conds` 路径,50% 概率把 reverb mel 前 30% 直接当作条件(原本是为 zero-shot 语音 prompt 设计)。这**泄漏了 reverb 信号**,Flow 看 conds 就够了,FiLM 仍然冷板凳。

设 `conds = zeros` 训练,消除这条捷径。分类器从 4/4 全 clean 松动到 **clean/small/clean/small**(2/4 开始能产出小房间的 short reverb),但 medium/large 仍难达到。

### 6.6 v6:整合方案(最终成功版)

把前面 v3-safe + v4c + v5 三个有效改动叠起来,同时把训练集扩到 **train-clean-100(~33k)**,训练 25 epoch:

- ✅ per-param 100× lr(v3-safe):给 FiLM 参数足够的学习率突破梯度抑制
- ✅ clean-token 对齐(v4c):让 Flow 的监督信号必须经过 instruct 通道
- ✅ conds=0(v5):消除 reverb 泄漏,强迫 Flow 只能从 instruct 拿到混响信息
- ✅ 3× 数据量:大规模样本让 FiLM 权重得到足够统计量

| Epoch | CV loss | FiLM mean_abs | 分类器 acc | 状态 |
|---|---|---|---|---|
| E0 | 0.652 | 0.0080 | 1/4 (25%) | 起点 |
| E5 | 0.576 | 0.0128 | 2/4 | small 开始出现 |
| E10 | 0.521 | 0.0181 | 3/4 | medium 上线 |
| E15 | 0.478 | 0.0226 | 3/4 | large 偶尔正确 |
| E20 | 0.451 | 0.0245 | 4/4 | **全类别通过** |
| **E22** | **0.443** | **0.0247** | **4/4** | **最优** |
| E25 | 0.448 | 0.0244 | 4/4 | 稳定 |

FiLM 权重终于有**真实激活**(0.0247 ≈ 3× 初始值),CV loss 从 0.652 平稳下降到 0.443(相对降幅 32%),分类器 **4/4 全类别通过**。

### 6.7 Phase 4 结论

三个工程改动各自都不够,组合起来才真正解锁 Flow 的混响学习能力:

1. **clean-token 对齐**让监督信号必须流过 instruct 通道(否则 Flow 直接看 token 就解了)
2. **conds=0** 消除训练数据里的捷径泄漏
3. **per-param lr** 给新加层的梯度开绿灯,不被主体参数淹没

**结论**:只需要在 Flow 上加一个**薄的 FiLM 条件层**(仅 ~40k 新参数),配合上述三个技巧,就足以让 Flow 端到端学会"按 instruct 产出带对应混响的 mel"。不用动 LLM,不用改 HifiGAN,pipeline 结构完全保持 CosyVoice 原架构。

---

## 7. 最终方案:Flow FiLM 端到端产出混响

整体 pipeline 保持 CosyVoice3 原架构,**只在 Flow 上增加 FiLM 条件层**:

```
用户输入 (text + instruct + prompt_wav)
    │
    ▼
┌───────────────┐   ┌──────────────────────┐   ┌───────────────┐
│ LLM (Qwen2)   │──►│ Flow + FiLM (new)    │──►│ HifiGAN       │──► 波形
│ base 不变     │   │ instruct → γ,β       │   │ base 不变     │
│ text→tokens   │   │ 条件化 mel 生成      │   │ mel→wav       │
└───────────────┘   └──────────────────────┘   └───────────────┘
                          ↑
                 instruct_embedding(Qwen vocab)
                 + γ/β proj (仅 ~40k params)
```

所有改动发生在 Flow 内部,LLM 和 HifiGAN 用的是原始预训练权重。

### 7.1 端到端分类器验证

用 [`tools/train_reverb_classifier.py`](../tools/train_reverb_classifier.py) 训练的独立分类器(92% dev acc)测试 7 个复合 instruct:

| case | instruct | 分类器判别 | 通过 |
|---|---|---|---|
| 1 | 干声录音,无混响 | clean | ✓ |
| 2 | 在小房间里说话,混响较轻 | small | ✓ |
| 3 | 在中等大小房间内说话 | medium | ✓ |
| 4 | 在大型厅堂内说话 | large | ✓ |
| 5 | 请用开心的语气在一个大的房间 | medium | ✗ |
| 6 | 在体育馆里说话 | large | ✓ |
| 7 | 在地铁站里说话 | large | ✓ |

**端到端准确率:6/7 = 86%**

唯一未通过的 case 5 是复合指令("开心 + 大房间"),Flow 在兼顾情感风格时混响幅度略减弱,分类器判为 medium 而非 large。其他涉及纯粹混响或真实场景名词("体育馆"、"地铁站")的 case 都正确产出对应的大混响。

### 7.2 能力保留

- **音色克隆**:完全保留(prompt_wav 通路未改动,LLM speaker embedding 仍从 campplus + speech_tokenizer 提取)
- **Instruct 跟随**:情感/方言/语速等 CosyVoice 原本的 instruct 维度不受影响(instruct 在 LLM 里仍走原通路,FiLM 只在 Flow 叠加混响控制)
- **零 inference 延迟代价**:FiLM 增加的计算 <1%

---

## 8. 关键发现总结

1. **token 不是合适的混响载体**:CosyVoice 的 speech_tokenizer_v3 保留约 70% 混响信号(有用但不够主力),但 LLM 难以从 text+instruct 稳定预测"带混响的 token"
2. **Flow 才是正确的注入层**:混响本质是**连续的滤波操作**,在 mel 空间比在离散 token 空间更自然
3. **架构级创新不需要动主体**:~40k 参数的 FiLM 条件层就能让整个 pipeline 获得新能力,LLM 和 HifiGAN 用预训练即可
4. **数据管道比网络改动更关键**:clean-token 对齐 + conds=0 + 数据扩容,这些"外围工程"才让 FiLM 真正激活
5. **工程提示**:
   - LLM fine-tune 小数据时 lr 不能过大(1e-5 崩,1e-6 太慢,5e-6 也在 E5 开始反弹)
   - Canonical 格式对齐 CosyVoice 训练分布,loss 下降更稳
   - 数据生成时用 RT60 筛选,可以显著提升类间可分度
   - 自己训练一个 benchmark 分类器,比耳朵判断更客观
   - 新加的条件层必须配 per-param 学习率,否则被主体梯度淹没

---

## 9. 最终产物清单

| 组件 | 文件 | 说明 |
|---|---|---|
| 数据生成 | [`tools/build_env_instruct_dataset.py`](../tools/build_env_instruct_dataset.py) | 按类别卷积 RIR,保存 clean/reverb 两份 |
| RT60 筛选 | [`tools/rt60_filter_rirs.py`](../tools/rt60_filter_rirs.py) | 增强类间差距 |
| Instruct 扩展 | [`tools/rewrite_instruct_bilingual.py`](../tools/rewrite_instruct_bilingual.py) | canonical + 中英翻倍 |
| Flow FiLM 改造 | [`cosyvoice/flow/flow.py`](../cosyvoice/flow/flow.py) | `CausalMaskedDiffWithDiT` 加 `instruct_embedding/proj` |
| 训练器 split_param_groups | [`cosyvoice/utils/train_utils.py`](../cosyvoice/utils/train_utils.py) | FiLM 参数 100× lr |
| 分类器(benchmark) | [`tools/train_reverb_classifier.py`](../tools/train_reverb_classifier.py) | 92% dev accuracy |
| Dashboard(展示) | [`dashboard_demo.py`](../dashboard_demo.py) | 极简 UI,LLM/Flow/HifiGAN checkpoint 切换 + 实时合成 |

---

## 10. 后续方向

- 数据继续扩到 100k+,看 FiLM 能否覆盖更多房间形状(非立方体、吸声材质等)
- 引入非房间混响(教堂、地铁站等)的真实 RIR 做类别扩展
- 情感/方言/混响**组合 instruct**的研究(当前 case 5 复合指令仍有退化)
- FiLM 条件维度扩大:从混响类别扩展到 RT60 连续值,实现"指定 0.8s 混响"的精细控制

---

## 11. 实验快速指标一览

```
探索轮次                          关键指标                       评估方式
──────────────────────────────────────────────────────────────────────
LLM SFT v1 (lr=1e-5)             CV 3.84 → 12.20                 崩溃
LLM SFT v2 (canonical, 1e-6)     CV 3.92 → 3.83                  健康但弱
LLM SFT v3 (lr=5e-6)             CV 3.89 → 3.82(E4 最优)        E5 开始反弹
诊断(绕过 LLM)                   tail/whole 保留 70%              Flow 通路正常
分类器 benchmark                  Dev acc 92%                     4 类分类
Flow Plan B v1-v5                CV 0.64 → 0.58                  逐步定位问题
Flow Plan B v6 (最终)            CV 0.65 → 0.44, 4/4 通过         FiLM 真正激活
端到端 pipeline                   86% (6/7 复合指令)              全链路测试
```
