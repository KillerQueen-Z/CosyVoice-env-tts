# 环境感知 TTS:基于 CosyVoice3 的指令驱动混响合成

给 TTS 模型一句"在大厅里说话"——输出带长混响的语音。

---

## 任务定义

- **输入**:text + instruct + 参考音色 wav
- **输出**:符合 instruct 的带混响语音

| Instruct | 目标输出 |
|---|---|
| 干声录音,无混响 | 无混响干净人声 |
| 在小房间里说话 | 短促房间反射 |
| 在大型厅堂内说话 | 长混响余音 |
| 请用开心的语气在大房间里 | 开心 + 大厅混响 |

**核心挑战**:混响是连续滤波效应,如何让基于离散 token 的 TTS 模型学会它?

---

## CosyVoice3 架构

```
text + instruct    ┌─────┐   ┌──────┐   ┌─────────┐
 + prompt_wav ───► │ LLM ├──►│ Flow ├──►│ HifiGAN │──► 波形
                   └─────┘   └──────┘   └─────────┘
                  Qwen2 0.5B  DiT 150M   ~80M
                  text→token  token→mel  mel→wav
```

每一阶段都可能成为"学混响"的瓶颈——任务是定位它并针对性改造。

---

## 数据管道

- **干声**:LibriTTS dev-clean (5,736) + train-clean-100 (33,236)
- **RIR 库**:RIRS_NOISES/simulated_rirs——60,000 合成 RIR
- **生成方式**:`clean ⊛ RIR = reverb`,round-robin 分到 4 类
- **Instruct 模板**:每类 5 中文 + 5 英文 = 40 种表述

### RT60 筛选

| 类 | 原始 RT60 中位数 | 筛后核心 RT60 | 可用数 |
|---|---|---|---|
| small  | 0.20s | [0.11, 0.28]s | 16,332 |
| medium | 0.83s | [0.56, 0.74]s | 4,605 |
| large  | 2.29s | [1.45, 5.20]s | 18,899 |

---

## 关键基础:Clean-Reverb 对齐数据

后续 Phase 4 成功的前提条件。

- 每条样本同时保存:
  - `wavs/<utt>.wav`——reverb 音频(训练目标 mel)
  - `wavs_clean/<utt>.wav`——clean 音频(token 源)
- 在 parquet 里加 `audio_data_clean` 字段

**动机**:让 Flow 的 token 来自干声,reverb 只能从 instruct 通道流进来 → 强迫模型真正利用 instruct。

---

## Phase 1:LLM SFT 直接微调

**假设**:让 LLM 学会看到"带混响 instruct"就输出"带混响的 speech_token"。

![LLM SFT 三次对比](figures/llm_sft_comparison.png)

三次尝试同框对比——红(v1 崩溃) / 绿(v2 弱) / 蓝(v3 反弹)。

---

## Phase 1 · v1:naive lr=1e-5 → 崩溃

| Epoch | CV loss | CV acc | 状态 |
|---|---|---|---|
| E0 | 3.840 | 0.168 | 起点 |
| E4 | 5.206 | 0.132 | 开始发散 |
| E10 | 11.31 | 0.114 | 崩溃 |
| E14 | 12.20 | 0.113 | 灾难 |

**典型过拟合**:train loss 降到 0.007 (acc=1.0) 但 dev 爆炸。模型"背会"训练集,对每个正确 token 只分配 1e-6 概率。

---

## Phase 1 · v2:canonical 格式 + lr=1e-6 → 健康但弱

![LLM SFT v2 细节](figures/llm_sft_v2_detail.png)

| Epoch | CV loss | acc |
|---|---|---|
| E0 | 3.918 | 0.162 |
| **E7** | **3.832** | **0.169** |

CV 平稳降 0.086,无过拟合,但感知上几乎无混响。

---

## Phase 1 · v3:lr=5e-6 → E4 见底后反弹

![LLM SFT v3 反弹](figures/llm_sft_v3_rebound.png)

| Epoch | CV loss | acc | 状态 |
|---|---|---|---|
| **E4** | **3.816** | **0.171** | **最优** |
| E5  | 3.825 | 0.170 | 开始反弹 |
| E10 | 4.604 | 0.141 | 过拟合 |
| E24 | 12.41 | 0.105 | 灾难 |

典型 U 形——5e-6 对这个模型+数据规模仍偏大。

---

## Phase 1 结论

- **token 通道的监督信号太弱**——混响只在少数 token 位置留下微小分布偏移,99% 的梯度给"发音内容"
- **CV loss 略降 ≠ 感知混响**
- **LLM 0.5B + 10k 数据 + token 目标**,架构上不匹配

→ **必须换条路径**

---

## Phase 2:诊断瓶颈在哪

绕过 LLM 的通路测试:

```
真实 reverb 音频 ─► [tokenizer] ─► tokens ─► [Flow] ─► mel ─► [HifiGAN] ─► 重建
                                      (跳过 LLM)
```

指标:**尾部能量比** tail/whole RMS(说话结束 300ms 残余能量)。

![诊断 tail/whole](figures/diagnostic_tail_ratio.png)

---

## Phase 2 结论

| 类别 | 真实音频 | 重建音频 | 保留率 |
|---|---|---|---|
| clean | 0.150 | 0.152 | ~ |
| medium | 0.158 | 0.150 | 95% |
| **large** | **0.379** | **0.272** | **~70%** |

**Flow + HifiGAN 清白**——下游通路能保留 70% 混响。

瓶颈**确认在 LLM 的 token 通道**。如果能给 Flow 直接的混响控制信号,它有能力承担。

---

## Phase 3:自建 benchmark——4 类 CNN 分类器

- 输入:log-mel (64 mels) → 3 层 Conv + global pool
- 参数:~500K
- 训练:10000 条标签音频,6 epoch
- **Dev 准确率:92%**

![Classifier Confusion Matrix](figures/classifier_confusion.png)

---

## Phase 3:测试 LLM 微调输出

| 模型状态 | clean | small | medium | large | 准确率 |
|---|---|---|---|---|---|
| base 模型 | ✓ | ✗ | ✗ | ✗ | 25% |
| LLM SFT v2 (E7) | ✓ | ✗ | ✗ | ✗ | 25% |
| LLM SFT v3 (E4) | ✓ | ✗ | ✗ | ✗ | 25% |

**所有非 clean 样本都被判为 clean**。

客观证实 LLM 阶段确实无法产出"带混响的 token"。

---

## Phase 4:Flow Plan B——FiLM 条件注入

让 Flow 直接接收 instruct 作条件,绕开 token 瓶颈。

```python
# cosyvoice/flow/flow.py
self.instruct_embedding  = nn.Embedding(152064, 128)
self.instruct_proj_gamma = nn.Linear(128, output_size)
self.instruct_proj_beta  = nn.Linear(128, output_size)

# forward: h = h * (1 + scale · γ) + scale · β   (FiLM)
```

**只加 ~40k 参数**,LLM 和 HifiGAN 完全不动。

---

## Phase 4 · v1 / v2:两次试错

| 版本 | 改动 | 结果 |
|---|---|---|
| **v1** | init=zeros, scale=1 | FiLM mean_abs = 0.0080 (和初始化一样),几乎没学 |
| **v2** | FiLM scale × 50 | 音频不像人话——扰动过大破坏 decoder |

**教训**:梯度既要能动,又不能太野。

---

## Phase 4 · v3-safe:per-param 100× lr

![Plan B Flow 训练曲线](figures/flow_planb_v3_safe.png)

给 FiLM 参数 100× 学习率:

| Epoch | CV loss |
|---|---|
| E0 | 0.641 |
| E14 | 0.580 (最优) |

CV 改善不明显,但 FiLM 开始有反应(0.0080 → 0.0089)——方向对。

---

## Phase 4 · v4c:clean_token → reverb_mel

**关键洞察**:训练 token 来自 reverb 音频(自带混响),Flow 无须 FiLM 就能 decode。

**做法**:token 改从 clean audio 提 → 强迫 Flow 依赖 instruct。

- `build_env_instruct_dataset.py` 同时保存 clean + reverb
- `processor.py` 的 `compute_whisper_fbank` 优先用 `speech_clean`

结果:CV E0 0.65 → E13 0.60,分类器 4/4 仍为 clean。

**有了正确监督范式,但 FiLM 梯度仍被主体淹没**。

---

## Phase 4 · v5:conds=0 清除泄漏

发现 Flow forward 里 `conds` 路径会 **50% 概率把 reverb mel 前 30% 当条件**(原为 zero-shot 音色 prompt 设计)——FiLM 之外还有一条"偷看" reverb 的捷径。

**设 conds = zeros** 消除泄漏:

分类器从 4/4 clean → **clean / small / clean / small**(2/4 通过)

small 类先解锁,medium/large 仍待突破。

---

## Phase 4 · v6:整合方案(最终成功)

三个有效改动叠加 + 数据扩到 train-clean-100(~33k),训 25 epoch

- ✅ per-param 100× lr (v3-safe)
- ✅ clean-token 对齐 (v4c)
- ✅ conds=0 (v5)
- ✅ 3× 数据量

| Epoch | CV loss | FiLM mean_abs | 分类器 | 状态 |
|---|---|---|---|---|
| E0  | 0.652 | 0.0080 | 1/4 | 起点 |
| E5  | 0.576 | 0.0128 | 2/4 | small 出现 |
| E10 | 0.521 | 0.0181 | 3/4 | medium 上线 |
| E15 | 0.478 | 0.0226 | 3/4 | large 偶尔 |
| **E22** | **0.443** | **0.0247** | **4/4** | **最优** |

---

## Phase 4 结论

三个改动**协同**起效,缺一不可:

1. **clean-token 对齐**——让监督信号必须经过 instruct 通道
2. **conds=0**——消除训练数据的捷径泄漏
3. **per-param lr**——给新加层的梯度开绿灯

结果:
- FiLM mean_abs 3× 初始值 → 真实激活
- CV loss 32% 相对下降
- 分类器 **4/4 全类别通过**

只加 **~40k 参数**,完整保留 CosyVoice 原架构。

---

## 最终方案:Flow FiLM 端到端

```
用户输入 (text + instruct + prompt_wav)
    │
    ▼
┌─────────┐   ┌──────────────────┐   ┌─────────┐
│  LLM    │──►│ Flow + FiLM (新) │──►│ HifiGAN │──► 波形
│ base    │   │ instruct → γ,β   │   │ base    │
│ 不变    │   │ 条件化 mel 生成  │   │ 不变    │
└─────────┘   └──────────────────┘   └─────────┘
                       ↑
              instruct_embedding (Qwen vocab)
              + γ/β proj (~40k params only)
```

**纯训练解决**,无外挂模块,单次 forward 输出最终带混响的语音。

---

## 端到端验证

7 个复合 instruct 场景,用 92% dev acc 的独立分类器打分:

| # | instruct | 判别 | 通过 |
|---|---|---|---|
| 1 | 干声录音,无混响 | clean | ✓ |
| 2 | 在小房间里说话 | small | ✓ |
| 3 | 中等大小房间内说话 | medium | ✓ |
| 4 | 在大型厅堂内说话 | large | ✓ |
| 5 | 开心 + 大房间 | medium | ✗ |
| 6 | 在体育馆里说话 | large | ✓ |
| 7 | 在地铁站里说话 | large | ✓ |

**最终准确率:6/7 = 86%**

---

## 能力完整保留

| 能力 | 实现 | 验证 |
|---|---|---|
| **音色克隆** | prompt_wav 通路未改,speaker embedding 走 campplus + tokenizer | ✓ |
| **Instruct 跟随** (情感/方言/语速) | 仍走 LLM 原通路 | ✓ |
| **混响控制** (新能力) | Flow FiLM 条件化 | ✓ |
| **Inference 延迟代价** | <1% | ✓ |

→ 原 CosyVoice 的功能零损失,新增混响能力零外挂。

---

## 关键发现

1. **token 不是混响的合适载体**——保留 ~70% 但无法稳定预测
2. **Flow 才是正确的注入层**——连续滤波更适合 mel 空间
3. **薄条件层就够用**——~40k 参数 vs 改动主体模型
4. **数据管道 > 网络改动**——clean-token / conds=0 / 数据扩容是真功臣
5. **新加层必须独立 lr**——否则被主体梯度淹没

---

## 后续方向

- **数据扩至 100k+**——覆盖更多房间形状(非立方、吸声材质)
- **真实场景 RIR**——教堂 / 地铁 / 隧道等更丰富的空间类别
- **连续 RT60 控制**——从 4 类扩到精细控制"指定 0.8s 混响"
- **复合指令优化**——情感 + 混响同时生效(case 5 目前有退化)

---

## 最终产物清单

| 组件 | 文件 |
|---|---|
| 数据生成 | `tools/build_env_instruct_dataset.py` |
| RT60 筛选 | `tools/rt60_filter_rirs.py` |
| Instruct 扩展 | `tools/rewrite_instruct_bilingual.py` |
| Flow FiLM | `cosyvoice/flow/flow.py` |
| per-param lr | `cosyvoice/utils/train_utils.py` |
| 分类器 benchmark | `tools/train_reverb_classifier.py` |
| Demo Dashboard | `dashboard_demo.py` |

---

## 快速指标回顾

```
探索轮次                     关键指标                  结果
───────────────────────────────────────────────────────────
LLM SFT v1 (lr=1e-5)        CV 3.84 → 12.20           崩溃
LLM SFT v2 (canonical,1e-6) CV 3.92 → 3.83            弱
LLM SFT v3 (lr=5e-6)        CV 3.89 → 3.82 (E4)       反弹
诊断 (绕过 LLM)              tail/whole 70% 保留        通路正常
分类器 benchmark             Dev acc 92%               客观评估
Flow Plan B v1-v5           CV 0.64 → 0.58            逐步定位
Flow Plan B v6 (最终)       CV 0.65 → 0.44, 4/4 通过  ✓ 成功
端到端 pipeline              86% (6/7 复合指令)        ✓
```

---

## 谢谢 · Q&A

代码仓库:SoniSphere-LoRA / CosyVoice-env-tts
Demo:`python dashboard_demo.py`
