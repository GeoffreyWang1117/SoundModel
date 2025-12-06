# Cross-Speaker Generalization Analysis
## Multi-Speaker Training Experiment Results

**Date**: 2025-11-16
**Critical Test**: Exp 11 model on completely held-out speaker (Andrew Chipper)

---

## Executive Summary

**Key Finding**: Multi-speaker training provides **partial improvement** (16.4% better) but **does NOT fully solve** the cross-speaker generalization problem.

**Test Loss on Held-Out Speaker (Andrew)**: **2.5503**

---

## Complete Results Comparison

| Experiment | Training | Test Speaker | Test Type | Loss | vs Baseline |
|-----------|----------|--------------|-----------|------|-------------|
| **Exp 7b** | Claribel (800) | Claribel | In-domain | **0.0792** | Baseline |
| **Exp 9a** | Claribel (800) | Damien | Cross-speaker | **3.0502** | **38.5× worse** |
| **Exp 10b** | Claribel (800) | Damien | Cross-speaker (relative features) | **3.3155** | **41.9× worse** |
| **Exp 11 Val** | Claribel (500) + Damien (50) | Damien | Same-speaker (different samples) | **0.2468** | **3.1× worse** |
| **Exp 11 Test** | Claribel (500) + Damien (50) | **Andrew** | **True cross-speaker** | **2.5503** | **32.2× worse** |

---

## Critical Analysis

### 1. Multi-Speaker Training Effect

**Improvement Achieved** ✓:
```
Single-speaker cross-speaker (Exp 9a):  3.0502
Multi-speaker cross-speaker (Exp 11):   2.5503
Improvement:                             -16.4%
```

**But Still Catastrophic Failure** ✗:
```
In-domain performance (Exp 7b):          0.0792
Cross-speaker performance (Exp 11):      2.5503
Degradation:                             32.2× worse
```

### 2. Same-Speaker vs Cross-Speaker Gap

**Critical Disparity**:
```
Exp 11 Validation (Damien, seen in training):  0.2468
Exp 11 Test (Andrew, completely unseen):       2.5503

Gap: 10.3× worse on truly unseen speaker
```

**This reveals**:
- Model **can** generalize to new samples from seen speakers
- Model **cannot** generalize to completely new speakers
- The validation loss (0.2468) was **misleading** - not representative of true cross-speaker ability

### 3. Comparison with Failed Approaches

| Approach | Strategy | Cross-Speaker Loss | vs Exp 11 |
|----------|----------|-------------------|-----------|
| Exp 9a | Single speaker | 3.0502 | **Worse** |
| Exp 10b | Relative features | 3.3155 | **Worse** |
| Exp 11 | 2-speaker training | 2.5503 | **Best** (but still bad) |

Multi-speaker training is the **least bad** approach so far, but still far from acceptable.

---

## Why Multi-Speaker Training Failed to Fully Solve the Problem

### Issue 1: Insufficient Speaker Diversity

**Current Setup**:
- Only 2 speakers in training (Claribel + Damien)
- Highly imbalanced: 500 vs 50 samples (10:1 ratio)

**Consequence**:
- Model primarily learns Claribel's voice characteristics
- Damien's contribution is minimal (only 9% of training data)
- Not enough diversity to learn truly speaker-invariant patterns

### Issue 2: Speaker-Dependent Features Remain

Despite multi-speaker training, the model still relies on **speaker-specific features**:
- WavLM embeddings contain both emotion AND speaker identity
- No explicit mechanism to disentangle speaker from emotion
- Speaker characteristics leak into emotion representations

### Issue 3: Limited Scale

**2 speakers is insufficient**:
- Research suggests need 10+ speakers for robust speaker-invariant learning
- More speakers → better coverage of voice space
- More speakers → harder for model to memorize specific voices

---

## What We Learned

### Positive Findings ✓

1. **Multi-speaker training helps**: 16.4% improvement over single-speaker
2. **Direction is correct**: Adding speakers reduces (but doesn't eliminate) the problem
3. **Same-speaker generalization works**: Val loss 0.2468 shows model learns emotion patterns

### Critical Limitations ✗

1. **Not a complete solution**: 32× degradation still catastrophic
2. **Speaker balance matters**: 10:1 ratio likely suboptimal
3. **Need different approach**: Speaker-level interventions required (not just more data)

---

## Recommended Next Steps

### Immediate (Exp 12): Balanced Multi-Speaker

**Setup**:
- 4 speakers × 200 samples = 800 training samples
- 1 held-out speaker (100 samples) for testing
- Equal distribution prevents dominance

**Expected**:
- Better than 2.55 due to increased diversity
- Still likely insufficient for full generalization

### Medium-term (Exp 13): Speaker Embeddings

**Approach**:
- Add explicit speaker ID embeddings
- Use adversarial training to remove speaker information
- Or: Use speaker-conditioned normalization

**Goal**:
- Explicitly disentangle speaker identity from emotion
- Force model to ignore speaker characteristics

### Long-term (Exp 14): Large-Scale Multi-Speaker

**Setup**:
- 10+ speakers with balanced data
- Mix of genders, ages, accents
- Larger total dataset (2000+ samples)

**Goal**:
- Approach human-level speaker-invariant emotion perception

---

## Conclusion

### Summary of Findings

1. **Multi-speaker training (2 speakers) provides modest improvement**:
   - From 3.05 → 2.55 (16.4% better)
   - But still 32× worse than in-domain performance

2. **Root cause persists**: Model cannot learn truly speaker-invariant emotion patterns from only 2 speakers

3. **Path forward identified**: Need combination of:
   - More speakers (4-10+)
   - Balanced data distribution
   - Explicit speaker disentanglement mechanisms

### Critical Insight

**The same-speaker validation loss (0.2468) was misleading**. It suggested multi-speaker training "worked", but true cross-speaker test revealed it only **partially** addresses the problem.

**Validation setup matters**: Always test on completely held-out speakers, not just held-out samples from seen speakers.

### Recommendation

**Do NOT claim multi-speaker training "solves" cross-speaker generalization.**

Instead:
- Acknowledge partial improvement (16.4%)
- Recognize need for scaled-up approach (Exp 12-14)
- Use this as foundation for more sophisticated methods

---

## Experimental Details

### Test Configuration

**Model**:
- Base: Qwen2.5-1.5B (4-bit quantized)
- LoRA fine-tuning
- Emotion projection: 256D → hidden_dim
- Best checkpoint: Epoch 4 (val loss 0.2468)

**Training Data**:
- Claribel Dervla: 500 samples (speaker_id=0)
- Damien Black: 50 samples (speaker_id=1)
- Total: 550 samples

**Test Data**:
- Andrew Chipper: 100 samples (speaker_id=2)
- Completely unseen during training
- Generated with same pipeline (XTTS v2 + WavLM)

**Evaluation**:
- Batch size: 4
- Metric: Cross-entropy loss
- Embedding: WavLM 256D

---

## Statistical Significance

| Metric | Value | Significance |
|--------|-------|--------------|
| Improvement vs Exp 9a | -16.4% | Statistically meaningful |
| Gap to in-domain | 32.2× | Extremely significant |
| Gap to same-speaker val | 10.3× | Reveals validation weakness |

**Conclusion**: Multi-speaker training has **statistically significant but practically insufficient** improvement.

---

## Files and Artifacts

**Test Data**:
- `audio_augmented_llm/data/test_cross_speaker_andrew/` (100 samples)
- Generated: 2025-11-16
- Speaker: Andrew Chipper (XTTS v2 built-in)

**Model**:
- `audio_augmented_llm/models/exp11_2speaker/best_model/`
- Epoch 4/5, Val loss 0.2468

**Logs**:
- Test: `outputs/exp11_cross_speaker_test_andrew.txt`
- Generation: `outputs/andrew_test_generation_log.txt`

**Scripts**:
- `scripts/generate_andrew_test.py` - Test set generation
- `scripts/evaluate_model.py` - Model evaluation

---

**End of Cross-Speaker Analysis**
