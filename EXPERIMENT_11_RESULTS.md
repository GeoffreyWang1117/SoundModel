# Experiment 11: Multi-Speaker Training Results

## Experiment Overview

**Objective**: Solve cross-speaker generalization problem through multi-speaker training

**Hypothesis**: Training on multiple speakers simultaneously will help the model learn speaker-invariant emotion representations, enabling better generalization to new speakers.

**Date**: 2025-11-16

---

## Dataset Configuration

### Training Set (550 samples)
- **Claribel Dervla** (speaker_id=0): 500 samples
- **Damien Black** (speaker_id=1): 50 samples

### Validation Set (50 samples)
- **Damien Black** (speaker_id=1): 50 samples (different from training)

### Important Notes
- Validation set uses same speaker (Damien) as in training, but different samples
- This tests **same-speaker generalization**, not true cross-speaker generalization
- For true cross-speaker test, would need completely held-out speaker

---

## Model Configuration

```
Base Model:     Qwen2.5-1.5B
Quantization:   4-bit
Fine-tuning:    LoRA
Emotion Input:  WavLM embeddings (256D)
Integration:    Concatenation mode
Batch Size:     4
Learning Rate:  5e-5
Epochs:         5
```

---

## Training Results

### Epoch-by-Epoch Performance

| Epoch | Train Loss | Val Loss | Best? | Notes |
|-------|-----------|----------|-------|-------|
| 1/5   | 0.9724    | 0.3862   | ✓     | Initial convergence |
| 2/5   | 0.1483    | 0.2861   | ✓     | Strong improvement |
| 3/5   | 0.1223    | 0.3212   |       | Val loss increased (overfitting) |
| 4/5   | 0.1083    | **0.2468** | ✓   | **Best validation loss** |
| 5/5   | 0.0912    | 0.5725   |       | Clear overfitting |

**Best Model**: Epoch 4 with validation loss **0.2468**

### Training Observations

1. **Rapid convergence**: Training loss dropped from 0.97 → 0.09 in 5 epochs
2. **Overfitting**: Clear overfitting starting at Epoch 3, severe by Epoch 5
3. **Best performance**: Epoch 4 achieved optimal train/val balance
4. **Early stopping**: Training should ideally stop at epoch 4-5 based on val loss

---

## Comparison with Previous Experiments

### Cross-Speaker Generalization Comparison

| Experiment | Approach | Test Type | Loss | Improvement |
|-----------|----------|-----------|------|-------------|
| **Exp 7b** | Single-speaker (Claribel) | In-domain | 0.0792 | Baseline (in-domain) |
| **Exp 9a** | Single-speaker (Claribel) | Cross-speaker (Damien) | 3.0502 | **38.5× degradation** |
| **Exp 10b** | Relative features | Cross-speaker (Damien) | 3.3155 | Failed (worse) |
| **Exp 11 Val** | Multi-speaker (2 speakers) | Same-speaker val (Damien) | 0.2468 | 12.4× better (misleading) |
| **Exp 11 Test** | Multi-speaker (2 speakers) | **True cross-speaker (Andrew)** | **2.5503** | **16.4% better than Exp 9a** |

### 🔥 TRUE Cross-Speaker Test Results (Andrew Chipper)

**Test Date**: 2025-11-16
**Test Speaker**: Andrew Chipper (100 samples, completely unseen)
**Test Loss**: **2.5503**

**Critical Findings**:
1. **Partial improvement**: 16.4% better than Exp 9a (3.0502 → 2.5503)
2. **Still catastrophic**: 32.2× worse than in-domain (0.0792 → 2.5503)
3. **Validation was misleading**: 10.3× gap between val (0.2468) and true cross-speaker test (2.5503)

### Key Findings

#### 1. Multi-Speaker Training Provides Partial Improvement ⚠️

**True cross-speaker test (Andrew)**: 2.5503
**Improvement vs single-speaker (Exp 9a)**: -16.4% (3.0502 → 2.5503)

Multi-speaker training helps, but **does NOT fully solve** the cross-speaker generalization problem.

#### 2. Validation Was Misleading ❌

| Test Type | Loss | Gap |
|-----------|------|-----|
| Validation (Damien, seen speaker) | 0.2468 | Baseline |
| True cross-speaker (Andrew, unseen) | 2.5503 | **10.3× worse** |

**Critical lesson**: Validation on seen speakers (even different samples) does NOT predict cross-speaker performance.

#### 3. Problem Remains Catastrophic ✗

Despite multi-speaker training:
- **32.2× worse** than in-domain performance (0.0792 → 2.5503)
- Still severe degradation on unseen speakers
- Only marginally better than single-speaker baseline

#### 3. Comparison to In-Domain Performance

Even on same-speaker validation, Exp 11 (0.2468) is **3.1× worse** than Exp 7b's in-domain performance (0.0792). This could be due to:
- Smaller Damien training set (50 vs 800+ Claribel samples)
- Multi-speaker training complexity
- Different data distributions

---

## Analysis

### Why Multi-Speaker Training Should Help

**Theoretical Benefits**:
1. **Speaker-invariant features**: Model must learn emotion patterns that work across speakers
2. **Reduced speaker-specific overfitting**: Can't rely on single speaker's characteristics
3. **Better feature disentanglement**: Forces separation of speaker identity from emotion

### Current Results Interpretation

**What We Learned** ✓:
- Model successfully trains on 2-speaker data
- Achieves reasonable validation loss (0.2468) on seen speaker
- No catastrophic failure like in Exp 9a/10b

**What We Still Don't Know** ?:
- True cross-speaker generalization (need held-out speaker test)
- Whether 2 speakers is sufficient (may need 3-4+ speakers)
- Optimal training data distribution across speakers

### Limitations of Current Experiment

1. **Not a true cross-speaker test**: Validation speaker (Damien) appears in training
2. **Imbalanced speaker distribution**: 500 Claribel vs 50 Damien samples
3. **Limited speaker diversity**: Only 2 speakers (1 female, 1 male)
4. **Overfitting observed**: Val loss increases after epoch 4

---

## Next Steps

### Immediate: True Cross-Speaker Evaluation

**Goal**: Test Exp 11 model on completely held-out speaker

**Options**:
1. Generate new speaker data (e.g., "Andrew Chipper")
2. Hold out all Damien data and use as pure test set
3. Use existing test_cross_speaker_damien (100 samples)

### Future Experiments

#### Exp 12: Balanced Multi-Speaker Training
- 250 samples × 3 speakers = 750 training samples
- 100 samples from 4th speaker for held-out testing
- Equal distribution prevents speaker-specific overfitting

#### Exp 13: More Speakers
- Scale to 4-6 speakers
- Test if more speaker diversity improves generalization

#### Exp 14: Speaker Embeddings
- Add explicit speaker ID embeddings
- Help model disentangle speaker identity from emotion

---

## Conclusion

### Summary

Experiment 11 demonstrates that **multi-speaker training is feasible** and produces models that:
- ✓ Train successfully on 2-speaker data
- ✓ Achieve reasonable validation loss (0.2468)
- ✓ Show 12.4× improvement over single-speaker cross-speaker failures*

*With caveat: comparison is between different test conditions

### Critical Next Step

**We must test true cross-speaker generalization** (completely held-out speaker) to determine if multi-speaker training actually solves the generalization problem identified in Experiments 9a and 10b.

### Recommendation

**Proceed with cross-speaker evaluation** using one of:
1. Hold out all Damien data → Train on Claribel only → Test on Damien
2. Generate 3rd speaker → Train on Claribel + Damien → Test on 3rd speaker

This will provide definitive evidence whether multi-speaker training addresses the 38.5× cross-speaker degradation problem.

---

## Technical Details

### Model Checkpoint
```
Location: ./audio_augmented_llm/models/exp11_2speaker/best_model
Epoch: 4/5
Val Loss: 0.2468
Contents:
  - LoRA adapters
  - Emotion projection layer (256D → hidden_dim)
  - Training config
```

### Data Directories
```
Training:   ./audio_augmented_llm/data/exp11_2speaker_train (550 samples)
Validation: ./audio_augmented_llm/data/exp11_2speaker_val (50 samples)
```

### Training Time
- Total: ~15 minutes (5 epochs)
- Per epoch: ~3 minutes
- Hardware: CUDA GPU

---

## Files Modified

1. `scripts/train_exp11.py` - New training script for separate train/val dirs
2. `scripts/create_2speaker_dataset.py` - Dataset creation script
3. `scripts/extract_missing_embeddings.py` - WavLM embedding extraction
4. `audio_augmented_llm/src/student_training/dataset.py` - Added relative embedding support

---

**End of Experiment 11 Results**
