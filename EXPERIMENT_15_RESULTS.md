# Experiment 15: Speaker Adaptive Normalization (SAN) - Analysis

**Date**: 2025-11-27
**Status**: ❌ FAILED - Worse than baseline
**Key Finding**: Speaker-invariance ≠ Cross-speaker generalization

---

## 1. Experiment Overview

### Hypothesis
Adding speaker-conditioned normalization during training, then removing it during inference, would create speaker-invariant representations that generalize better to unseen speakers.

### Approach
- **Architecture**: Speaker Adaptive Normalization (SAN)
  - Training: Learn speaker-specific scale/bias parameters for each of 4 speakers
  - Inference: Use averaged parameters (no speaker conditioning)
  - Goal: Force model to learn speaker-invariant emotion features

### Configuration
- **Model**: Qwen-2.5-1.5B + LoRA (4-bit)
- **Training data**: 800 samples (4 speakers: Claribel, Damien, Andrew, Gracie)
- **Validation data**: 80 samples (same 4 speakers)
- **Test data**: 100 samples (Viktor - completely unseen speaker)
- **Embedding**: WavLM 256D
- **Integration**: Concat
- **Inference mode**: Average (use average of all speaker parameters)
- **Hyperparameters**: 5 epochs, batch=4, LR=5e-5

---

## 2. Results Summary

### Critical Failure: The Speaker-Invariance Paradox

| Metric | Exp 13 (Baseline) | Exp 15 (SAN) | Change |
|--------|-------------------|--------------|--------|
| **Validation Loss** | 0.0316 | 0.0863 | +173% ⚠️ |
| **Val Gap** (with/without speaker IDs) | 80.2× | 1.00× | **Perfect!** ✅ |
| **Cross-speaker Test Loss** (Viktor) | 2.5336 | **4.6717** | **+84.4%** ❌ |
| **Degradation Factor** | 80.2× | 54.1× | Improved |

### Key Observations

1. **Perfect Speaker-Invariance Achieved**: Gap = 1.00× (identical performance with/without speaker IDs during validation)
2. **But Cross-Speaker Performance Collapsed**: Test loss almost doubled (2.53 → 4.67)
3. **Even Baseline Validation Got Worse**: 0.0316 → 0.0863

### Training Progression

| Epoch | Train Loss | Val (w/ spk) | Val (no spk) | Gap |
|-------|-----------|--------------|--------------|-----|
| 1 | 1.3336 | 0.1897 | 0.1903 | 1.00× |
| 2 | 0.1502 | 0.1340 | 0.1340 | 1.00× |
| 3 | 0.1257 | 0.1211 | 0.1211 | 1.00× |
| 4 | 0.1164 | 0.1002 | 0.1000 | 1.00× |
| 5 | 0.1076 | 0.0864 | 0.0863 | 1.00× |

---

## 3. Critical Analysis

### What Went Wrong?

#### ❌ The False Equivalence
**Assumption**: Speaker-invariance during validation → Better cross-speaker generalization
**Reality**: Speaker-invariance ≠ Cross-speaker generalization

#### Root Cause Analysis

1. **Overfitting to Training Speakers**
   - SAN learned speaker-specific parameters for Claribel, Damien, Andrew, Gracie
   - Averaging them creates a "representation" that doesn't exist in reality
   - Viktor's features fall outside the learned speaker space

2. **Information Loss**
   - Removing speaker conditioning removes useful variance
   - Some speaker-specific information may actually help the model understand emotion patterns
   - Complete speaker-invariance throws away signal along with noise

3. **Validation vs Test Mismatch**
   - Validation set contains same 4 speakers as training
   - "No speaker IDs" during validation ≠ truly unseen speaker
   - The averaged parameters still encode information about the 4 training speakers

4. **Higher Baseline Validation Loss**
   - Exp 13: 0.0316
   - Exp 15: 0.0863 (2.7× worse)
   - SAN architecture itself may be limiting model capacity or introducing harmful inductive bias

### Why the Approach Failed

```
Training Speakers: {A, B, C, D}
Average Parameters: μ(A,B,C,D)

Test Speaker: V (Viktor)

Problem: μ(A,B,C,D) ≠ V
The average of known speakers ≠ arbitrary unknown speaker
```

The fundamental flaw: **Averaging parameters from 4 specific speakers creates a bias toward those speakers**, not a speaker-agnostic representation.

---

## 4. Comparison with Baseline (Exp 13)

### What Exp 13 Did Right
- No explicit speaker conditioning
- Model forced to learn emotion features that work across all 4 speakers naturally
- Better validation loss (0.0316 vs 0.0863)
- Better cross-speaker performance (2.5336 vs 4.6717)

### What Exp 15 Did Wrong
- Explicit speaker conditioning created dependencies
- Averaging creates artificial representation
- Added complexity without benefit
- Worse on all metrics except the meaningless "gap" metric

---

## 5. Key Insights

### 1️⃣ Speaker-Invariance is Not the Goal
- Low validation gap ≠ good cross-speaker generalization
- The gap metric measures consistency, not generalization ability

### 2️⃣ Some Speaker Information is Useful
- Completely removing speaker variance may throw away useful signal
- Emotion expression has speaker-specific patterns that help the model learn

### 3️⃣ Averaging is Not Speaker-Agnostic
- Average of 4 speakers ≠ neutral representation
- Creates bias toward training speaker distribution

### 4️⃣ Architecture Complexity Has Costs
- SAN added parameters and complexity
- Worse baseline performance suggests harmful inductive bias
- Added regularization (speaker conditioning) may have limited model capacity

---

## 6. Lessons Learned

### ❌ What Doesn't Work
1. Explicit speaker conditioning → averaging during inference
2. Focusing on validation gap as proxy for cross-speaker ability
3. Assuming speaker-invariance == speaker-agnostic

### ✅ What We Know Works Better
1. Simple baseline without explicit speaker conditioning (Exp 13)
2. Letting model naturally learn shared emotion features
3. Measuring actual cross-speaker performance, not validation gaps

---

## 7. Future Directions

### Option A: Abandon SAN Approach
- Evidence strongly suggests this approach is fundamentally flawed
- Return to simpler architectures

### Option B: Fix the Averaging Problem
**Potential modifications** (if we still want to try SAN):
1. **Learnable neutral speaker**: Instead of averaging, learn a neutral speaker embedding
2. **Speaker dropout**: Randomly drop speaker conditioning during training
3. **Adversarial speaker removal**: Add discriminator to force speaker-invariant features

### Option C: Different Architecture (Recommended)
Try approaches that don't rely on speaker conditioning:
1. **Contrastive learning**: Learn emotion features robust to speaker changes
2. **Meta-learning**: Train on speaker-held-out tasks
3. **Data augmentation**: Mix speaker characteristics during training

---

## 8. Conclusion

**Experiment 15 revealed a critical insight through failure**:

> **Speaker-invariance during validation does not imply cross-speaker generalization.**

The SAN approach achieved perfect speaker-invariance (1.00× gap) but made cross-speaker performance significantly worse (+84% higher loss). This demonstrates that:

1. **The validation gap metric is misleading** - it measures consistency among known speakers, not generalization to unknown speakers
2. **Complete speaker-invariance is harmful** - some speaker-specific information helps the model learn emotion patterns
3. **Simple baselines outperform complex normalization** - Exp 13's straightforward approach works better

### Recommendation
**Abandon the SAN approach** and focus on:
1. Understanding why the simple baseline (Exp 13) fails at cross-speaker generalization
2. Exploring data augmentation and contrastive learning approaches
3. Investigating whether the problem is in the emotion embeddings themselves (WavLM may encode too much speaker info)

---

## Files Generated
- `audio_augmented_llm/src/student_training/speaker_adaptive_norm.py` - SAN implementation
- `audio_augmented_llm/src/student_training/student_model_san.py` - Student model with SAN
- `scripts/train_exp15.py` - Training script
- `audio_augmented_llm/models/exp15_san/` - Trained model
- `outputs/exp15_training_log.txt` - Training logs
- `outputs/exp15_viktor_test_results.txt` - Evaluation results

---

**Status**: Experiment failed but provided valuable negative results. SAN approach not recommended for future work.
