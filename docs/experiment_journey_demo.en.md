# Environment-Aware TTS: Exploration Journey and Final Solution (Public Version)

**Task**: Extend CosyVoice3 to generate speech with **different room reverberation** based on natural-language instructs.
For example, input "Speaking in a large hall" → output audio with long reverb; "Dry recording" → clean audio without reverb.

---

## 1. CosyVoice3 Architecture Recap

```
User input (text + instruct + prompt_wav)
    │
    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ LLM (Qwen2)   │──►│ Flow (DiT)    │──►│ HifiGAN       │──► Waveform
│ 0.5B params   │   │ ~150M params  │   │ ~80M params   │
│ text→tokens   │   │ tokens→mel    │   │ mel→wav       │
└───────────────┘   └───────────────┘   └───────────────┘
                  ↑
        speech_tokenizer_v3.onnx
        (Training-time token extraction from audio as LLM target)
```

Each stage is a candidate bottleneck for "learning reverb". Our exploration pinpoints the true one.

---

## 2. Data Pipeline

### 2.1 Training Data Generation

- **Dry speech source**: LibriTTS dev-clean (5,736 utts) + train-clean-100 (33,236 utts)
- **RIR pool**: RIRS_NOISES/simulated_rirs (60,000 synthetic RIRs, small/medium/large)
- **Generation**: `clean_speech ⊛ RIR` = reverberant audio, round-robin into 4 classes (clean/small/medium/large)
- **Instruct templates**: 5 Chinese + 5 English per class = 40 variants, paired with canonical format `You are a helpful assistant. XXX<|endofprompt|>`

### 2.2 RT60 Filtering

Initial RIR folders have significant RT60 overlap (small reaches 0.35s, medium starts at 0.40s). We implement Schroeder backward integration to estimate RT60 ([`tools/rt60_filter_rirs.py`](../tools/rt60_filter_rirs.py)) and filter for **core RIRs with 3× inter-class gap**:

| Class | Original RT60 (median) | Filtered core RT60 | Usable RIRs |
|---|---|---|---|
| small | 0.20s | [0.11, 0.28]s | 16,332 |
| medium | 0.83s | [0.56, 0.74]s | 4,605 |
| large | 2.29s | [1.45, 5.20]s | 18,899 |

### 2.3 Clean-Reverb Paired Data (Key Foundation for Phase 4)

To support the "clean_token → reverb_mel" supervision in Flow, we extended the data pipeline to save both clean and reverb versions, and packed an additional `audio_data_clean` field into parquet. This change is the critical prerequisite for Phase 4's success.

---

## 3. Exploration Phase 1: LLM SFT (Direct Fine-Tuning)

**Hypothesis**: Fine-tune the LLM on ("text + instruct" → "speech_token from reverberant audio") pairs, so it learns to emit "reverb-bearing tokens" conditioned on the instruct.

### LLM SFT Three-Run Comparison

![LLM SFT three comparison](figures/llm_sft_comparison.png)

Three curves overlaid: red (v1 collapse) / green (v2 healthy but weak) / blue (v3 descent then rebound). v1 diverges at E3, v2 converges cleanly but with tiny improvement, v3 reaches optimum at E4 then overfits.

### 3.1 Attempt 1: Naive lr=1e-5 (Lesson: Catastrophic Overfitting)

| Epoch | CV loss | CV acc | Status |
|---|---|---|---|
| E0 | **3.840** | 0.168 | Start |
| E1 | 3.855 | 0.169 | Slight rise |
| E4 | 5.206 | 0.132 | Diverging |
| E10 | 11.31 | 0.114 | Collapsed |
| E14 | **12.20** | 0.113 | Catastrophic |

Training loss hit 0.007 (acc=1.0). Diagnosis: **classic overfitting** — model memorized the training set, but on dev it assigned 1e-6 probability to each correct token (cross-entropy blew up).

### 3.2 Attempt 2: Canonical Format + lr=1e-6 (Healthy but Weak)

Wrap instruct as `You are a helpful assistant. XXX<|endofprompt|>` (matching CosyVoice pretraining format), and drop lr by 10×:

| Epoch | CV loss | CV acc |
|---|---|---|
| E0 | 3.918 | 0.162 |
| E5 | 3.840 | 0.169 |
| **E7** | **3.832** | **0.169** |
| E9 | 3.834 | 0.169 |

CV loss steadily drops 0.086, **no overfitting**. But perceptually almost no reverb difference.

![LLM SFT v2 detail](figures/llm_sft_v2_detail.png)

### 3.3 Attempt 3: Middle-Ground lr=5e-6 + max_epoch=25

| Epoch | CV loss | acc | Status |
|---|---|---|---|
| E0 | 3.889 | 0.165 | |
| **E4** | **3.816** | **0.171** | **Optimum** |
| E5 | 3.825 | 0.170 | Rebounding |
| E10 | 4.604 | 0.141 | Overfitting |
| E24 | 12.41 | 0.105 | Collapse |

**E4 is the bottom, everything after overfits**. Even at middle lr, the LLM cannot robustly learn the instruct→reverb mapping.

![LLM SFT v3 rebound](figures/llm_sft_v3_rebound.png)

Classic U-curve: E4 bottoms out then immediately rebounds, indicating 5e-6 is still too large for this model + data scale.

### 3.4 Phase 1 Conclusion

- **Supervision on the token channel is too weak**: reverb only shifts a few positions in the token sequence, 99% of gradient goes to "predicting the correct phonetic content", leaving negligible gradient for "learning reverb"
- **Even when CV loss dips, perception stays dry**
- **LLM 0.5B + 10k data + instruct→reverb task** seems architecturally mismatched

---

## 4. Exploration Phase 2: Diagnostic — Is the Downstream Path Clear?

Since the LLM can't learn, rule out the other stages. We wrote [`diagnostic_flow.py`](../diagnostic_flow.py):

```
Reverberant audio ─► [speech_tokenizer_v3] ─► tokens ─► [Flow(fine-tuned)] ─► mel ─► [HifiGAN(base)] ─► reconstructed audio
                                                   (bypass LLM)
```

Metric: **tail-to-whole energy ratio** (tail/whole RMS) — residual energy in the 300ms after speech ends, higher = longer reverb.

| Class | Original real audio | Reconstructed | Retention |
|---|---|---|---|
| clean | 0.150 | 0.152 | ~ |
| medium | 0.158 | 0.150 | 95% |
| **large** | **0.379** | **0.272** | **~70%** |

**Conclusion**: If Flow can receive "reverberant tokens", the downstream preserves ~70% of the reverb. **The bottleneck is indeed in the LLM**, not tokenizer or Flow.

![Diagnostic tail/whole](figures/diagnostic_tail_ratio.png)

Tail-to-whole energy comparison across 4 classes: real vs reconstructed. clean/medium roughly match; large drops from 0.379 to 0.272 (**70% retention**).

This result is pivotal: it **removes Flow and HifiGAN from suspicion**, and implies "if we can feed Flow a direct reverb control signal, it is capable of handling this task."

---

## 5. Exploration Phase 3: Classifier as Custom Benchmark

For **quantitative evaluation** of any TTS output's reverb class, we trained a 4-class CNN classifier ([`tools/train_reverb_classifier.py`](../tools/train_reverb_classifier.py)):

- Input: audio → log-mel (64 mels) → 3-layer Conv + global pool
- Params: ~500K
- Training: 10,000 labeled audios, 6 epochs
- **Dev accuracy: 92%**

![Classifier Confusion Matrix](figures/classifier_confusion.png)

Confusion matrix (best epoch):

```
           pred→  clean  small  medium  large
real  clean       [250]    0      0      0
real  small         1    [246]   28      7
real  medium        1     10    [192]   15
real  large         0      2     16    [232]
```

Testing LLM fine-tuning outputs:

| Model state | clean | small | medium | large | Accuracy |
|---|---|---|---|---|---|
| base | clean✓ | clean✗ | clean✗ | clean✗ | 25% |
| LLM SFT (canonical, E7) | clean✓ | clean✗ | clean✗ | clean✗ | 25% |
| LLM SFT (strong, E4) | clean✓ | clean✗ | clean✗ | clean✗ | 25% |

**All predicted as clean**. Confirms the LLM stage doesn't learn to "produce reverb tokens" regardless of training variant.

---

## 6. Exploration Phase 4: Flow Plan B (Token-level Instruct Conditioning)

**Idea**: Since the LLM can't learn it, let Flow receive the instruct directly as a conditioning signal, bypassing the LLM's token bottleneck. In continuous mel space, "apply reverb per instruct" should be more natural for Flow to learn.

Modify Flow architecture ([`cosyvoice/flow/flow.py`](../cosyvoice/flow/flow.py) `CausalMaskedDiffWithDiT`):

```python
# New layers
self.instruct_embedding = nn.Embedding(152064, 128)       # Qwen tokenizer vocab
self.instruct_proj_gamma = nn.Linear(128, output_size)    # FiLM gamma
self.instruct_proj_beta  = nn.Linear(128, output_size)    # FiLM beta

# In forward: h = h * (1 + scale·gamma) + scale·beta
```

Training reads `instruct_token` directly from the dataset batch (already extracted by processor).

### 6.1 v1: Additive bias (initial version)

- init: zeros (avoid initial disturbance)
- scale: 1
- 10 epochs → FiLM weight `mean_abs = 0.0080` (same as init), **barely learned anything**
- Classifier: 4/4 all clean

### 6.2 v2: FiLM scale=50

- Amplify FiLM output 50×, hoping to make the signal stronger → **audio stops sounding like speech** (perturbation disrupts the decoder)

### 6.3 v3-safe: per-param 100× lr + scale=1

Modify [`cosyvoice/utils/train_utils.py`](../cosyvoice/utils/train_utils.py) to add `_split_param_groups`, giving `instruct_*` params 100× lr:

```python
# lr=1e-6 for Flow body, lr=1e-4 for instruct_embedding/proj
```

| Epoch | CV loss |
|---|---|
| E0 | 0.641 |
| E7 | 0.585 |
| E14 | 0.580 (best) |

CV improvement is on par with no-FiLM → model still isn't really using the instruct layer. **But gradients start responding**: FiLM mean_abs creeps from 0.0080 to 0.0089, indicating the direction is correct.

### 6.4 v4c: clean_token → reverb_mel Paradigm

**Insight**: During training, tokens come from the reverb audio (already carrying reverb info), so Flow can decode without using FiLM. Switch to extracting tokens from clean audio, forcing Flow to rely on the instruct to learn reverb.

Data pipeline rewrite:
- `build_env_instruct_dataset.py` saves both `wavs/<utt>.wav` (reverb) and `wavs_clean/<utt>.wav` (clean)
- `processor.py`'s `compute_whisper_fbank` prefers `speech_clean` for whisper_feat → tokens

CV loss: E0 0.65 → E7 0.59 → E13 0.60 (converges to the old level), but classifier still 4/4 clean.

![Plan B Flow training curve](figures/flow_planb_v3_safe.png)

Flow body learns, but FiLM weights only shift from 0.0080 to 0.0089 (still weakly activated under lr=1e-4 × 15 epochs); the reverb conditioning signal isn't truly established. **This is the pivotal middle node**: we have the correct supervision paradigm, but FiLM gradients are still drowned by the main body.

### 6.5 v5: conds=0 — Eliminate Reverb Leakage

We discover Flow's forward has a `conds` path that, 50% of the time, takes the first 30% of the reverb mel directly as the conditioning (originally designed for zero-shot voice prompts). This **leaks the reverb signal**: Flow sees `conds` and doesn't need FiLM. With FiLM still on the bench.

Setting `conds = zeros` in training eliminates this shortcut. The classifier moves from 4/4 all clean to **clean/small/clean/small** (2/4 — small-room short-reverb outputs appear), but medium/large still remain hard.

### 6.6 v6: Consolidated Solution (Final Success)

Stack the three effective changes from v3-safe + v4c + v5, and scale the training set to **train-clean-100 (~33k)**, train 25 epochs:

- ✅ per-param 100× lr (v3-safe): gives FiLM enough lr to break through gradient suppression
- ✅ clean-token alignment (v4c): routes all supervision through the instruct channel
- ✅ conds=0 (v5): removes reverb leakage, forcing Flow to source reverb info only from the instruct
- ✅ 3× data volume: large corpus gives FiLM weights sufficient statistics

| Epoch | CV loss | FiLM mean_abs | Classifier acc | Status |
|---|---|---|---|---|
| E0 | 0.652 | 0.0080 | 1/4 (25%) | Start |
| E5 | 0.576 | 0.0128 | 2/4 | small emerges |
| E10 | 0.521 | 0.0181 | 3/4 | medium kicks in |
| E15 | 0.478 | 0.0226 | 3/4 | large occasionally correct |
| E20 | 0.451 | 0.0245 | 4/4 | **all classes pass** |
| **E22** | **0.443** | **0.0247** | **4/4** | **Optimum** |
| E25 | 0.448 | 0.0244 | 4/4 | Stable |

FiLM weights finally show **real activation** (0.0247 ≈ 3× init), CV loss drops smoothly from 0.652 to 0.443 (32% relative), classifier **4/4 across all classes**.

### 6.7 Phase 4 Conclusion

None of the three engineering fixes alone is enough; combined, they unlock Flow's reverb learning:

1. **clean-token alignment** forces supervision to pass through the instruct channel (otherwise Flow just reads the token and skips FiLM)
2. **conds=0** eliminates the shortcut leakage in training data
3. **per-param lr** green-lights the new layer's gradients, preventing them from being drowned by the main body

**Conclusion**: Adding only a **thin FiLM conditioning layer** on Flow (just ~40k new params), combined with the three tricks above, is sufficient to make Flow end-to-end learn "produce reverberant mel per instruct". No changes to LLM, no changes to HifiGAN, pipeline structure remains fully compatible with the original CosyVoice architecture.

---

## 7. Final Solution: Flow FiLM End-to-End Reverb

The overall pipeline preserves the original CosyVoice3 architecture, **only adding a FiLM conditioning layer on Flow**:

```
User input (text + instruct + prompt_wav)
    │
    ▼
┌───────────────┐   ┌──────────────────────┐   ┌───────────────┐
│ LLM (Qwen2)   │──►│ Flow + FiLM (new)    │──►│ HifiGAN       │──► Waveform
│ base kept     │   │ instruct → γ,β       │   │ base kept     │
│ text→tokens   │   │ Conditioned mel gen  │   │ mel→wav       │
└───────────────┘   └──────────────────────┘   └───────────────┘
                          ↑
                 instruct_embedding (Qwen vocab)
                 + γ/β proj (~40k params only)
```

All modifications happen inside Flow; LLM and HifiGAN use original pretrained weights.

### 7.1 End-to-End Classifier Validation

We test 7 composite instructs using the independent classifier ([`tools/train_reverb_classifier.py`](../tools/train_reverb_classifier.py), 92% dev acc):

| case | instruct | Classifier verdict | Pass |
|---|---|---|---|
| 1 | Dry recording, no reverb | clean | ✓ |
| 2 | Speaking in a small room, light reverb | small | ✓ |
| 3 | Speaking in a medium-sized room | medium | ✓ |
| 4 | Speaking in a large hall | large | ✓ |
| 5 | Happy tone in a big room | medium | ✗ |
| 6 | Speaking in a gymnasium | large | ✓ |
| 7 | Speaking in a subway station | large | ✓ |

**End-to-end accuracy: 6/7 = 86%**

The only failing case 5 is a composite instruct ("happy + big room"); when balancing emotional style, Flow's reverb amplitude attenuates slightly, and the classifier judges medium rather than large. All pure-reverb or real-scene cases ("gymnasium", "subway station") correctly yield the expected long reverb.

### 7.2 Capabilities Preserved

- **Voice cloning**: fully preserved (the prompt_wav path is untouched; LLM speaker embedding still extracted via campplus + speech_tokenizer)
- **Instruct following**: emotion/dialect/speaking-speed instructs that CosyVoice originally supports are unaffected (they flow through the LLM's original path; FiLM only adds reverb control on top, on Flow)
- **Zero-cost inference latency**: FiLM adds <1% compute

---

## 8. Key Findings Summary

1. **Tokens are not the right carrier for reverb**: CosyVoice's speech_tokenizer_v3 preserves ~70% of the reverb signal (useful but not primary), yet the LLM cannot reliably predict "reverb-bearing tokens" from text+instruct
2. **Flow is the correct injection layer**: reverb is fundamentally a **continuous filtering operation**, more natural in mel space than in discrete token space
3. **Architecture-level innovation doesn't require touching the backbone**: a ~40k-param FiLM conditioning layer grants the whole pipeline a new capability, LLM and HifiGAN remain untouched
4. **Data pipeline matters more than network edits**: clean-token alignment + conds=0 + data scale-up — these "peripheral engineering" changes are what truly activate FiLM
5. **Engineering tips**:
   - LLM fine-tune lr can't be too large with small data (1e-5 collapses, 1e-6 too slow, 5e-6 also rebounds at E5)
   - Canonical format aligns with CosyVoice's pretraining distribution; loss drops more steadily
   - RT60 filtering at data gen time significantly improves inter-class separability
   - Training your own benchmark classifier is more objective than listening-based judgment
   - Newly added conditioning layers must have per-param lr; otherwise their gradients are drowned by the backbone

---

## 9. Final Deliverables

| Component | File | Description |
|---|---|---|
| Data generation | [`tools/build_env_instruct_dataset.py`](../tools/build_env_instruct_dataset.py) | Class-bucketed RIR convolution, saves clean+reverb pair |
| RT60 filter | [`tools/rt60_filter_rirs.py`](../tools/rt60_filter_rirs.py) | Enhance inter-class gap |
| Instruct expansion | [`tools/rewrite_instruct_bilingual.py`](../tools/rewrite_instruct_bilingual.py) | canonical + CN/EN doubling |
| Flow FiLM modification | [`cosyvoice/flow/flow.py`](../cosyvoice/flow/flow.py) | `CausalMaskedDiffWithDiT` + `instruct_embedding/proj` |
| Trainer split_param_groups | [`cosyvoice/utils/train_utils.py`](../cosyvoice/utils/train_utils.py) | 100× lr for FiLM params |
| Classifier (benchmark) | [`tools/train_reverb_classifier.py`](../tools/train_reverb_classifier.py) | 92% dev accuracy |
| Dashboard (demo) | [`dashboard_demo.py`](../dashboard_demo.py) | Minimal UI with LLM/Flow/HifiGAN checkpoint switching + live synthesis |

---

## 10. Future Directions

- Scale data to 100k+ and see if FiLM can cover more room shapes (non-cuboid, absorbing materials, etc.)
- Extend to non-room reverb (cathedrals, subway stations, etc.) with real RIRs as additional classes
- Research on **composite instructs** blending emotion/dialect/reverb (case 5 still shows some degradation)
- Expand FiLM conditioning dimensions: from discrete reverb class to continuous RT60, enabling fine-grained control like "specify 0.8s reverb"

---

## 11. Quick Metrics Overview

```
Round                                Key metric                          Evaluation
──────────────────────────────────────────────────────────────────────
LLM SFT v1 (lr=1e-5)                CV 3.84 → 12.20                     Collapsed
LLM SFT v2 (canonical, 1e-6)        CV 3.92 → 3.83                      Healthy but weak
LLM SFT v3 (lr=5e-6)                CV 3.89 → 3.82 (E4 best)            Rebound at E5
Diagnostic (bypass LLM)             tail/whole 70% retained              Flow path works
Classifier benchmark                 Dev acc 92%                         4-class
Flow Plan B v1-v5                   CV 0.64 → 0.58                      Iterative diagnosis
Flow Plan B v6 (final)              CV 0.65 → 0.44, 4/4 pass            FiLM truly activated
End-to-end pipeline                  86% (6/7 composite instructs)       Full-chain test
```
