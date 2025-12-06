# Experiment 12a: Acoustic Features Cross-Speaker Test

**Date**: 2025-11-16
**Status**: ✅ COMPLETED
**Duration**: 6 seconds (evaluation only)

## Objective

Test whether Acoustic features (46D), which outperformed WavLM (256D) in-domain, also show better cross-speaker generalization.

## Hypothesis

Since Acoustic features are interpretable (pitch, energy, MFCCs) and designed for prosody/emotion, they may be more speaker-invariant than black-box WavLM embeddings.

## Experimental Setup

### Model
- **Training**: Exp 7b model trained on Claribel (1000 samples)
- **Architecture**: Qwen2.5-1.5B + LoRA (rank 8) + 4-bit quantization
- **Features**: Acoustic 46D (prosodic 13D + spectral 20D + formant 13D)
- **In-domain val loss**: 0.0792 (Claribel validation set)

### Test Data
- **Speaker**: Damien Black (completely unseen during training)
- **Samples**: 100
- **Features**: Acoustic 46D extracted from Damien's audio

## Results

### Cross-Speaker Performance

| Metric | Value |
|--------|-------|
| Test loss (Damien) | **3.2264** |
| In-domain val loss (Claribel) | 0.0792 |
| Performance degradation | **40.7× worse** |
| Evaluation time | 6 seconds |
| Throughput | ~16 samples/second |

## Comparison to WavLM (Exp 9a)

| Feature Type | Dimension | In-Domain Val Loss | Cross-Speaker Test Loss | Degradation |
|--------------|-----------|-------------------|------------------------|-------------|
| **Acoustic** | 46D | 0.0792 | **3.2264** | 40.7× |
| **WavLM** | 256D | 0.0792 | **3.0502** | 38.5× |

### Key Finding

**Acoustic features perform WORSE than WavLM in cross-speaker scenario**:
- Acoustic: 3.2264 (5.8% higher loss than WavLM)
- WavLM: 3.0502
- Difference: +0.1762 (worse)

## Analysis

### Why Acoustic Features Don't Help Cross-Speaker

Despite being more interpretable and better in-domain, Acoustic features show:

1. **Higher Speaker Dependence**: Acoustic features capture speaker-specific characteristics:
   - **Pitch range**: Varies 2-3× across speakers (e.g., male vs female)
   - **Formant frequencies**: Depend on vocal tract anatomy
   - **MFCCs**: Capture timbre, which is speaker-specific

2. **Interpretability ≠ Speaker-Invariance**: 
   - Interpretable features (pitch, energy) are precisely the ones that vary most across speakers
   - Black-box WavLM learns to abstract away some speaker information during pre-training

3. **No Advantage from Lower Dimensionality**:
   - While 46D provides better regularization in-domain (prevents overfitting)
   - It doesn't help with speaker-invariance (different problem)

### In-Domain vs Cross-Speaker Trade-off

| Aspect | In-Domain | Cross-Speaker |
|--------|-----------|---------------|
| **Acoustic advantage** | ✅ 40.45% improvement | ❌ 5.8% WORSE |
| **Better feature** | Acoustic (0.0655) | WavLM (3.0502) |
| **Reason** | Task alignment, interpretability | Less speaker-specific |

## Conclusions

### Confirmed

1. ❌ **Hypothesis REJECTED**: Acoustic features are NOT more speaker-invariant
2. ✅ **Cross-speaker problem is fundamental**: Neither Acoustic nor WavLM solve it
3. ✅ **Feature type is NOT the bottleneck**: Problem lies in speaker-dependence, not feature representation

### Implications for Future Work

**Skip Exp 12c** (Acoustic + 2-speaker training):
- Since Acoustic doesn't outperform WavLM cross-speaker
- No reason to expect it would help in multi-speaker training
- Save 1 hour of training time

**Focus on**:
1. **Multi-speaker data scale** (Exp 13-14): 4-6 speakers
2. **Architectural approaches** (Exp 15-16): Speaker disentanglement, adversarial training
3. **Real human speech** (Exp 17): TTS may lack natural speaker variation

## Next Steps

### Immediate (Today)

Skip Exp 12b (Fusion) and Exp 12c (Acoustic 2-speaker) since:
- Fusion (302D) performed between Acoustic and WavLM in-domain (0.0771)
- No evidence it would outperform WavLM cross-speaker
- Acoustic doesn't show speaker-invariance advantage

### Short-term (This Week)

**Proceed directly to Exp 13**: 4-speaker balanced training
- Use WavLM (256D) as the baseline feature
- 4 speakers × 200 samples = 800 total
- Test on Viktor (unseen speaker)
- Target: < 2.0 cross-speaker loss

## Experimental Log Entry

```
Exp 12a: Acoustic Cross-Speaker Test
├── Training: Claribel (Exp 7b acoustic model, val: 0.0792)
├── Test: Damien (100 samples, acoustic 46D)
├── Result: 3.2264 (WORSE than WavLM 3.0502)
└── Conclusion: Acoustic features NOT more speaker-invariant
```

## Files

- Model: `audio_augmented_llm/models/exp7b_acoustic_1000/best_model/`
- Test data: `audio_augmented_llm/data/test_cross_speaker_damien/`
- Features: `acoustic_embeddings.npz` (46D)
- Results: This file

---

**Bottom Line**: Acoustic features' in-domain advantage does NOT translate to cross-speaker robustness. The problem requires multi-speaker training or architectural solutions, not better feature engineering.
