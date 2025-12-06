# Experiment 13 Results: 4-Speaker Multi-Speaker Training

**Date**: 2025-11-17
**Hypothesis**: Increasing speaker diversity from 2 to 4 will improve cross-speaker generalization
**Result**: ❌ REJECTED - No significant cross-speaker improvement

---

## Experimental Setup

### Dataset
- **Training**: 4 speakers (Claribel, Damien, Andrew, Gracie)
  - 200 samples per speaker = **800 total**
  - Gender balanced: 2 female + 2 male
- **Validation**: 4 speakers (same as training)
  - 20 samples per speaker = **80 total**
- **Cross-Speaker Test**: Viktor Eka (male, completely unseen)
  - 100 samples

### Model Configuration
- Base model: Qwen/Qwen2.5-1.5B (4-bit quantized)
- LoRA: rank 8, alpha 16
- Embedding: WavLM (256D)
- Integration: Concat
- Training: 5 epochs, batch size 4, LR 5e-5

---

## Results Summary

### Validation Loss (Same-Speaker Performance)

| Epoch | Val Loss | Improvement |
|-------|----------|-------------|
| 1 | 0.1344 | - |
| 2 | 0.0823 | ↓ 38.8% |
| 3 | 0.0406 | ↓ 50.7% |
| 4 | 0.0387 | ↓ 4.7% |
| **5** | **0.0316** | **↓ 18.3%** |

**vs Exp 11 (2-speaker)**: 0.0316 vs 0.2468 = **87.2% improvement** ✅

### Cross-Speaker Test (Viktor - Unseen Speaker)

**Test Loss**: **2.5336**

**Comparison with Exp 11**:
- Exp 11 (Andrew test): 2.5503
- Exp 13 (Viktor test): 2.5336
- **Improvement**: 0.65% (0.0167 absolute)

---

## Detailed Comparison

| Metric | Exp 11 (2-speaker) | Exp 13 (4-speaker) | Change |
|--------|-------------------|-------------------|--------|
| **Training Speakers** | 2 (Claribel, Damien) | 4 (C, D, Andrew, Gracie) | +100% |
| **Training Samples** | 550 | 800 | +45% |
| **Val Loss (same-speaker)** | 0.2468 | **0.0316** | **-87.2%** ✅ |
| **Cross-speaker Loss** | 2.5503 (Andrew) | **2.5336** (Viktor) | **-0.65%** ❌ |
| **Degradation Factor** | 32.2× | 80.2× | 2.5× worse |

---

## Key Findings

### 1. Same-Speaker Performance Dramatically Improved

The 4-speaker model achieves **0.0316** validation loss vs 0.2468 for 2-speaker:
- **87.2% relative improvement**
- Much stronger in-domain performance
- Shows model can learn from diverse speakers **when they're in training**

### 2. Cross-Speaker Generalization NOT Improved

Despite 2× more speakers and 45% more data:
- **Only 0.65% improvement** (2.5503 → 2.5336)
- Essentially **no meaningful change**
- Still **80× worse** than same-speaker performance

### 3. The Problem is Worse Than We Thought

The degradation factor actually **increased**:
- Exp 11: 32.2× (0.2468 → 2.5503)
- Exp 13: **80.2×** (0.0316 → 2.5336)

This means the model became MORE speaker-dependent, not less!

---

## Analysis

### Why Did Same-Speaker Performance Improve?

1. **More diverse training data** (4 speakers vs 2)
2. **Better coverage** of emotion/prosody variations
3. **Larger effective dataset** (800 vs 550 samples)

### Why Did Cross-Speaker Performance NOT Improve?

1. **Speaker information is entangled with emotion**
   - Model learns speaker-specific emotion patterns
   - Cannot generalize to new speakers

2. **More speakers → stronger speaker dependence**
   - With 4 speakers, model learns 4 distinct "emotion styles"
   - New speaker (Viktor) doesn't match any learned style
   - The better same-speaker fit = worse cross-speaker generalization

3. **Fundamental architectural limitation**
   - Concat integration doesn't disentangle speaker from emotion
   - No mechanism to extract speaker-invariant features
   - Just adding data won't fix architecture issues

---

## Comparison with Goals

| Goal | Target | Result | Status |
|------|--------|--------|--------|
| Same-speaker val loss | < 0.3 | **0.0316** | ✅ Exceeded |
| Cross-speaker loss | < 2.0 | **2.5336** | ❌ Failed |
| Improvement over Exp 11 | > 20% | **0.65%** | ❌ Failed |

---

## Conclusions

### Main Conclusion
**Increasing speaker diversity alone does NOT solve cross-speaker generalization.**

The hypothesis that more training speakers would improve cross-speaker generalization is **REJECTED**.

### Why This Experiment Matters

This is actually a **critical negative result**:
1. Rules out data-only solutions
2. Confirms the problem is **architectural/methodological**
3. Points to need for:
   - Speaker-invariant representations
   - Adversarial training
   - Speaker conditioning/normalization
   - Disentanglement techniques

### The Speaker-Performance Paradox

We discovered a paradox:
- **Better same-speaker fit** → **Worse cross-speaker generalization**
- More speakers in training → Model becomes MORE speaker-dependent
- This is the opposite of what we expected!

---

## Next Steps (Updated Priorities)

Based on these results, we now know:

### ❌ What DOESN'T Work
1. Adding more speakers (Exp 11 → Exp 13)
2. Acoustic features (Exp 12a)
3. Relative normalization (Exp 10)
4. Just more data

### ✅ What to Try Next (Architecture-Level Solutions)

**High Priority**:
1. **Exp 15**: Speaker Adaptive Normalization
   - Learn speaker-specific normalization parameters
   - Condition on speaker ID during training
   - Remove speaker ID during inference

2. **Exp 16**: Adversarial Speaker Disentanglement
   - Add speaker classifier as discriminator
   - Train emotion encoder to fool speaker classifier
   - Force speaker-invariant representations

3. **Exp 17**: Real Human Speech (IEMOCAP)
   - Test if TTS is the bottleneck
   - Natural prosody variations might generalize better

**Medium Priority**:
4. Contrastive Learning
5. Meta-learning / Few-shot adaptation
6. Speaker embedding subtraction

---

## Detailed Metrics

### Training Progression

| Epoch | Train Loss | Val Loss | Time |
|-------|-----------|----------|------|
| 1 | 1.8532 | 0.1344 | 2.7min |
| 2 | 0.4287 | 0.0823 | 2.6min |
| 3 | 0.0884 | 0.0406 | 2.7min |
| 4 | 0.0312 | 0.0387 | 2.6min |
| 5 | 0.0187 | **0.0316** | 2.6min |

**Total training time**: ~13 minutes

### Cross-Speaker Evaluation

- **Test samples**: 100 (Viktor Eka)
- **Batch size**: 8
- **Test loss**: 2.5336
- **Evaluation time**: ~7 seconds

---

## Files Generated

- Model: `audio_augmented_llm/models/exp13_4speaker/best_model/`
- Training log: `outputs/exp13_training_log.txt`
- Evaluation log: `outputs/exp13_viktor_test_results.txt`
- Dataset: `audio_augmented_llm/data/exp13_4speaker/`
- Test set: `audio_augmented_llm/data/test_cross_speaker_viktor/`

---

## Takeaway for Future Work

This experiment teaches us a crucial lesson:

> **Speaker generalization cannot be solved by adding more speakers to training data alone. The problem requires architectural solutions that explicitly disentangle speaker identity from emotion information.**

The next experiments (15-17) will focus on architectural/methodological improvements rather than just scaling up data.
