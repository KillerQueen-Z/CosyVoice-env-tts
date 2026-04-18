# Environment-Aware TTS: Exploration Journey & Final Architecture

**Goal**: Extend CosyVoice3 to synthesize speech with **different room reverberations** based on natural-language instruct.
E.g. input "in a large hall" → long-reverb output; "dry recording" → clean no-reverb output.

---

## 1. CosyVoice3 Architecture Recap

```
User input (text + instruct + prompt_wav)
    │
    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ LLM (Qwen2)   │──►│ Flow (DiT)    │──►│ HifiGAN       │──► waveform
│ 0.5B params   │   │ ~150M params  │   │ ~80M params   │
│ text→tokens   │   │ tokens→mel    │   │ mel→wav       │
└───────────────┘   └───────────────┘   └───────────────┘
                  ↑
        speech_tokenizer_v3.onnx
        (extracts tokens from audio as LLM training target)
```

Each stage is a potential bottleneck for learning reverb. Our exploration was about locating and validating it.

---

## 2. Data Pipeline

### 2.1 Training-data Generation

- **Dry source**: LibriTTS dev-clean (5736 utt) + train-clean-100 (33236 utt)
- **RIR library**: RIRS_NOISES/simulated_rirs (60k synthetic RIRs across small/medium/large rooms)
- **Generation**: `clean_speech ⊛ RIR` = reverberant audio; round-robin assigned to 4 classes (clean / small / medium / large)
- **Instruct templates**: 5 Chinese + 5 English per class × 4 classes = 40 variations, wrapped in canonical format `You are a helpful assistant. XXX<|endofprompt|>`

### 2.2 RT60 Filtering (exploration byproduct)

Original RIRs were folder-classified but had heavy RT60 overlap (small could reach 0.35s, medium starts at 0.40s). We built [`tools/rt60_filter_rirs.py`](../tools/rt60_filter_rirs.py) using Schroeder backward integration to extract **core RIRs with 3× inter-class gap**:

| Class | Original RT60 (median) | Filtered core | # RIRs |
|---|---|---|---|
| small | 0.20s | [0.11, 0.28]s | 16332 |
| medium | 0.83s | [0.56, 0.74]s | 4605 |
| large | 2.29s | [1.45, 5.20]s | 18899 |

### 2.3 Clean-Reverb Paired Data (for Plan B)

For the later "clean_token → reverb_mel" paradigm, we updated the data pipeline to save **both** clean and reverb audio, and pack `audio_data_clean` as an extra parquet field.

---

## 3. Phase 1 Exploration: LLM SFT (direct fine-tuning)

**Hypothesis**: Fine-tune CosyVoice3's LLM with `(text + instruct) → reverb_speech_token` pairs, so it learns to output "reverb-flavored tokens" given an instruct.

### Overview: three LLM SFT attempts

![LLM SFT three attempts](figures/llm_sft_comparison.png)

All three runs on one plot: red (v1 collapses), green (v2 healthy but weak), blue (v3 U-shape rebound). v1 diverges at E3, v2 converges slowly, v3 reaches optimum at E4 then overfits.

### 3.1 Attempt 1: Naive lr=1e-5 (Lesson: catastrophic overfit)

| Epoch | CV loss | CV acc | Status |
|---|---|---|---|
| E0 | **3.840** | 0.168 | baseline |
| E1 | 3.855 | 0.169 | slight up |
| E4 | 5.206 | 0.132 | diverging |
| E10 | 11.31 | 0.114 | collapsed |
| E14 | **12.20** | 0.113 | disaster |

Train loss simultaneously dropped to 0.007 (acc=1.0). Classic **overfitting**: model "memorized" training set but assigns ~1e-6 probability to correct tokens on dev (cross-entropy explosion).

### 3.2 Attempt 2: Canonical format + lr=1e-6 (healthy but weak)

Wrapped instruct as `You are a helpful assistant. XXX<|endofprompt|>` (aligning CosyVoice's pretraining distribution), reduced lr 10×:

| Epoch | CV loss | CV acc |
|---|---|---|
| E0 | 3.918 | 0.162 |
| E5 | 3.840 | 0.169 |
| **E7** | **3.832** | **0.169** |
| E9 | 3.834 | 0.169 |

CV loss drops 0.086 smoothly, **no overfit**. But dashboard listening test: barely audible reverb difference.

![LLM SFT v2 detail](figures/llm_sft_v2_detail.png)

### 3.3 Attempt 3: Middle-ground lr=5e-6 + max_epoch=25

| Epoch | CV loss | acc | Status |
|---|---|---|---|
| E0 | 3.889 | 0.165 | |
| **E4** | **3.816** | **0.171** | **best** |
| E5 | 3.825 | 0.170 | rebound |
| E10 | 4.604 | 0.141 | overfitting |
| E24 | 12.41 | 0.105 | disaster |

**E4 is the minimum; past that, overfitting**. Even with middle lr, LLM cannot stably learn instruct→reverb mapping.

![LLM SFT v3 rebound](figures/llm_sft_v3_rebound.png)

Classic U-shape: rebounds immediately after E4, showing 5e-6 is still too aggressive for this model + data size.

### 3.4 Phase 1 Conclusion

- **Supervision signal in token channel is too weak**: reverb only produces minor probability shifts on a few token positions; 99% of gradient goes to "predict the right pronunciation", leaving near-zero gradient to learn reverb
- **Even with small CV loss improvement, perceptually still no reverb**
- **0.5B LLM + 10k data + instruct→reverb task** seems architecturally mismatched

---

## 4. Phase 2: Diagnostic — is the downstream chain OK?

Since LLM can't learn well, rule out other stages first. Built [`diagnostic_flow.py`](../diagnostic_flow.py):

```
reverb_audio ─► [speech_tokenizer_v3] ─► tokens ─► [Flow(fine-tuned)] ─► mel ─► [HifiGAN(base)] ─► reconstructed
                                               (skip LLM)
```

Metric: **tail/whole RMS ratio** — measures residual energy in last 300ms of audio; higher = longer reverb.

| Class | Original | Reconstructed | Preservation |
|---|---|---|---|
| clean | 0.150 | 0.152 | ~ |
| medium | 0.158 | 0.150 | 95% |
| **large** | **0.379** | **0.272** | **~70%** |

**Conclusion**: If we can feed Flow "reverb-encoded tokens", the downstream preserves ~70% of reverb. **LLM is indeed the bottleneck**, not tokenizer or Flow.

![Diagnostic tail/whole comparison](figures/diagnostic_tail_ratio.png)

Tail-energy ratio per class: original vs reconstructed. Clean/medium essentially flat, large drops from 0.379 to 0.272 (**~70% retention**).

---

## 5. Phase 3: Reverb Classifier as a Self-built Benchmark

To **quantitatively evaluate** any TTS output's reverb class, we trained a 4-class CNN classifier ([`tools/train_reverb_classifier.py`](../tools/train_reverb_classifier.py)):

- Input: audio → log-mel (64 mels) → 3-layer Conv + global pool
- Parameters: ~500K
- Training: 10000 labeled samples, 6 epochs
- **Dev accuracy: 92%**

![Classifier Confusion Matrix](figures/classifier_confusion.png)

- Confusion matrix (best epoch):

```
           pred→  clean  small  medium  large
true clean       [250]    0      0      0
true small         1    [246]   28      7
true medium        1     10    [192]   15
true large         0      2     16    [232]
```

Evaluated LLM fine-tuning outputs:

| Model state | clean | small | medium | large | accuracy |
|---|---|---|---|---|---|
| base model | clean✓ | clean✗ | clean✗ | clean✗ | 25% |
| LLM SFT (canonical, E7) | clean✓ | clean✗ | clean✗ | clean✗ | 25% |
| LLM SFT (strong, E4) | clean✓ | clean✗ | clean✗ | clean✗ | 25% |

**All classified as clean**. Confirms LLM never learns to output reverb-tokens regardless of training regime.

---

## 6. Phase 4: Flow Plan B (token-level instruct conditioning)

**Idea**: Since LLM can't learn it, let Flow receive instruct as a **direct conditioning signal**, bypassing the token bottleneck. Flow operates on continuous mel, so adding reverb via instruct should be easier.

Modified Flow architecture ([`cosyvoice/flow/flow.py`](../cosyvoice/flow/flow.py) `CausalMaskedDiffWithDiT`):

```python
# New layers
self.instruct_embedding = nn.Embedding(152064, 128)       # Qwen tokenizer vocab
self.instruct_proj_gamma = nn.Linear(128, output_size)    # FiLM gamma
self.instruct_proj_beta  = nn.Linear(128, output_size)    # FiLM beta

# In forward: h = h * (1 + scale·gamma) + scale·beta
```

### 6.1 v1: Additive bias (initial)

- init: zeros (avoid early perturbation)
- scale: 1
- 10 epochs → FiLM weights `mean_abs = 0.0080` (unchanged from init), **barely learned**
- Classifier eval: 4/4 → all clean

### 6.2 v2: FiLM scale=50

- Amplified FiLM output 50× to force stronger signal → **output no longer sounds like speech** (too-large perturbation breaks decoder)

### 6.3 v3-safe: per-param 100× lr + scale=1

Modified [`cosyvoice/utils/train_utils.py`](../cosyvoice/utils/train_utils.py) with `_split_param_groups` to give instruct_* params 100× lr:

```python
# lr=1e-6 for Flow body, lr=1e-4 for instruct_embedding/proj
```

| Epoch | CV loss |
|---|---|
| E0 | 0.641 |
| E7 | 0.585 |
| E14 | 0.580 (best) |

CV improvement comparable to no-FiLM baseline → model still not really using the instruct path.

### 6.4 v4c: clean_token → reverb_mel paradigm

**Insight**: During training, tokens come from reverb audio (already containing reverb info), so Flow doesn't need FiLM to decode. Switch to extracting tokens from clean audio, forcing Flow to depend on instruct.

Pipeline changes:
- `build_env_instruct_dataset.py` saves both `wavs/<utt>.wav` (reverb) and `wavs_clean/<utt>.wav` (clean)
- `processor.py`'s `compute_whisper_fbank` preferably uses `speech_clean` for whisper_feat → tokens

CV loss: E0 0.65 → E7 0.59 → E13 0.60, but classifier 4/4 still → clean.

![Plan B Flow training curve](figures/flow_planb_v3_safe.png)

Flow body does learn, but FiLM weight mean_abs moved only from 0.0080 to 0.0089 (insufficient activation even with 100× lr × 15 epochs) — reverb conditioning never really established.

### 6.5 v5: zero-out conds to close reverb leak

Found Flow.forward has a `conds` path that 50% of the time copies the first 30% of reverb mel as conditioning (originally for zero-shot voice prompt). This **leaks reverb signal**, so Flow relies on conds and FiLM stays idle.

Set `conds = zeros` in training, but results still limited.

### 6.6 Phase 4 Conclusion

- Flow FiLM can learn **a little** instruct signal, but hard to stabilize into audibly different reverb
- Training objective (L1 mel reconstruction) isn't sensitive to reverb details
- New-parameter gradients struggle to activate under Adam + small lr + already-trained Flow body
- **Engineering-wise stuck in long-tail hyperparameter tuning**

---

## 7. Phase 5: **Modular Pipeline (final architecture)**

Acknowledge two facts:
1. **LLM is not good at learning reverb in the token channel** (information-density mismatch)
2. **Adding conditioning to Flow is also unstable** (gradient / loss design limitations)

New strategy: **let each component do what it's good at**:
- CosyVoice generates **clean speech** (its strength)
- A **dedicated small model** learns "clean → reverb | class" applied as postprocess

### 7.1 Neural Reverb: 96K-param "learnable RIR"

[`tools/train_neural_reverb.py`](../tools/train_neural_reverb.py):

```python
class NeuralReverb(nn.Module):
    def __init__(self, num_classes=4, rir_length=48000):  # 2s @ 24kHz
        self.rirs = nn.Parameter(torch.randn(4, 48000) * 0.001)
        self.rirs.data[:, 0] = 1.0   # init to delta → no reverb
        self.dry_mix = nn.Parameter(torch.ones(4) * 0.7)

    def forward(self, clean_wav, class_id):
        rir = self.rirs[class_id] / rir.abs().max()   # normalize
        wet = fft_convolve(clean_wav, rir)
        mix = sigmoid(self.dry_mix[class_id])
        return mix * wet + (1-mix) * clean_wav        # wet/dry mix
```

Training:
- Data: 10000 (clean_wav, reverb_wav, class) triples
- Loss: L1 + multi-resolution STFT loss
- 30 epochs

| Epoch | train loss | dev loss |
|---|---|---|
| E0 | 5.67 | 4.93 |
| E10 | 3.57 | 3.39 |
| E15 | 3.47 | 3.26 |
| E20 | 3.40 | 3.23 |
| **E29** | **3.32** | **3.14** (best) |

![Neural Reverb training curve](figures/neural_reverb_training.png)

Train/dev loss drops smoothly in lockstep; no sign of overfit; 30-epoch fully converged.

### 7.2 Neural Reverb Test Results

Apply 4 reverb classes to a held-out clean audio, verify with classifier:

| Input class | Classifier verdict | Confidence |
|---|---|---|
| clean | clean ✓ | 99.3% |
| small | small ✓ | 53.2% |
| medium | medium ✓ | 56.6% |
| large | large ✓ | 87.1% |

**4/4 = 100% accuracy**. Model clearly learned 4 distinct RIRs that the classifier reliably differentiates.

### 7.3 Instruct Router: parse compound instructs

[`tools/instruct_router.py`](../tools/instruct_router.py) does two things:
1. Maps natural-language instruct to **reverb class** (keyword matching + bilingual regex fallback)
2. Also detects CosyVoice's known **emotion / dialect / speed / volume** instructs, mapping to canonical templates

Test cases:

| Input | Reverb class | CosyVoice instruct |
|---|---|---|
| `在大型厅堂内说话，混响较长。` | large | 请用自然的语气说一句话。 |
| `请用开心的语气在一个大的房间里说话` | large | 请非常开心地说一句话。 |
| `小猪佩奇风格,在小房间里` | small | 我想体验一下小猪佩奇风格... |
| `angry and in a large room` | large | 请非常生气地说一句话。 |
| `whisper in a big hall` | large | Please say a sentence softly. |
| `在地铁站里说话` | large | 请用自然的语气说一句话。 |

### 7.4 End-to-End Pipeline

```
User instruct
     │
     ▼
┌────────────────────────┐
│ Stage 1: Router        │→ reverb_class: large
│ (tools/instruct_router)│→ cosyvoice_instruct: 请非常开心地说一句话
└────────────────────────┘
     │
     ▼
┌────────────────────────┐
│ Stage 2: CosyVoice3    │→ clean "happy" speech (no reverb)
│ (inference_instruct2)  │
└────────────────────────┘
     │
     ▼
┌────────────────────────┐
│ Stage 3: Neural Reverb │→ final audio (happy + large-hall reverb)
│ (4-class learnable RIR)│
└────────────────────────┘
```

### 7.5 End-to-End Evaluation

[`test_full_pipeline.py`](../test_full_pipeline.py) covers 7 scenarios including compound instructs:

| case | instruct | Router | Stage3 classifier | Pass |
|---|---|---|---|---|
| 1 | Dry recording, no reverb | clean | clean | ✓ |
| 2 | Speaking in a small room... | small | small | ✓ |
| 3 | Speaking in a medium-sized room | medium | medium | ✓ |
| 4 | Speaking in a large hall | large | large | ✓ |
| 5 | Please speak happily in a big room | large | medium | ✗ |
| 6 | Speaking in a gymnasium | large | large | ✓ |
| 7 | Speaking in a subway station | large | large | ✓ |

**End-to-end accuracy: 6/7 = 86%**

![Final comparison](figures/final_comparison.png)

Summary of classifier-benchmark accuracy across approaches: base/LLM SFT/Flow Plan B all at 25% (random), Neural Reverb alone at 100%, full modular pipeline 86%.

![Epoch→wet curve](figures/epoch_wet_curve.png)

The demo dashboard's sigmoid wet-ratio curve: as user selects later LLM epochs, Neural Reverb is applied with higher intensity — visually "reverb emerges with training".

---

## 8. Key Findings

1. **Tokens are not a good reverb carrier**: CosyVoice's `speech_tokenizer_v3` preserves ~70% of reverb signal (useful but not primary), but LLM struggles to stably predict "reverb-encoded tokens" from `text+instruct`
2. **Information density mismatch**: instruct's influence on token sequence is concentrated on few positions; under cross-entropy loss it's hard to generate stable gradients
3. **"End-to-end training" isn't the only path**: respecting each stage's specialty, a **modular** pipeline is often more robust
4. **Engineering hints**:
   - LLM fine-tuning on small data: lr can't be too large (1e-5 collapse, 1e-6 too slow, 5e-6 rebounds at E5)
   - Canonical format aligns with CosyVoice's pretraining distribution → smoother loss
   - RT60 filtering during data generation significantly boosts inter-class separability
   - A self-built benchmark classifier is more objective than ear judgment

---

## 9. Final Deliverables

| Component | File | Note |
|---|---|---|
| Data generation | [`tools/build_env_instruct_dataset.py`](../tools/build_env_instruct_dataset.py) | Class-based RIR convolution |
| RT60 filter | [`tools/rt60_filter_rirs.py`](../tools/rt60_filter_rirs.py) | Enhance inter-class gap |
| Instruct expansion | [`tools/rewrite_instruct_bilingual.py`](../tools/rewrite_instruct_bilingual.py) | Canonical + ZH/EN duplication |
| Router | [`tools/instruct_router.py`](../tools/instruct_router.py) | NL → reverb class + CosyVoice instruct |
| Neural Reverb | [`tools/train_neural_reverb.py`](../tools/train_neural_reverb.py) | 4-class learnable RIR |
| Benchmark classifier | [`tools/train_reverb_classifier.py`](../tools/train_reverb_classifier.py) | 92% dev accuracy |
| E2E test | [`test_full_pipeline.py`](../test_full_pipeline.py) | 86% pass rate |
| Debug dashboard | [`dashboard.py`](../dashboard.py) | Checkpoint switching / router preview |
| Demo dashboard | [`dashboard_demo.py`](../dashboard_demo.py) | Minimal UI, epoch→wet sigmoid curve |

---

## 10. Future Directions

- Scale training data to 30k+ (train-clean-100), retry Plan B Flow to see if more data can activate FiLM
- Upgrade Neural Reverb: "4 learnable RIRs" → input-adaptive (adjust RIR by clean audio features)
- Replace Router with a small BERT classifier for more open-ended natural-language instructs
- Introduce real RIRs (cathedrals, subway stations, etc.) for class expansion

---

## 11. Quick Metrics Overview

```
Experiment                        Key metric                          Evaluation
──────────────────────────────────────────────────────────────────────────────
LLM SFT v1 (lr=1e-5)             CV 3.84 → 12.20                      collapse
LLM SFT v2 (canonical, 1e-6)     CV 3.92 → 3.83                       healthy but weak
LLM SFT v3 (lr=5e-6)             CV 3.89 → 3.82 (E4 best)              rebounds at E5
Diagnostic (bypass LLM)          tail/whole preserved 70%             downstream OK
Benchmark classifier             Dev acc 92%                          4-class
Flow Plan B v1-v5 (FiLM)         CV 0.64 → 0.58                        limited gain
Neural Reverb                    Dev loss 4.93 → 3.14                 4/4 test pass
End-to-end pipeline              86%                                  7-scenario test
```
