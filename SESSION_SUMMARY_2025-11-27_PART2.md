# Session Summary - 2025-11-27 (Part 2)

## Overview

This session continued the cross-speaker generalization research with a systematic investigation into why audio embeddings fail to generalize across speakers. The work completed Tasks 1-3 from the research roadmap:

1. ✅ **Task 1**: Test acoustic features (46D) for cross-speaker performance
2. ✅ **Task 2**: Analyze WavLM embeddings for speaker information content
3. ✅ **Task 3**: Design and implement contrastive learning approach (partial)

---

## Task 1: Acoustic Features Cross-Speaker Performance (Experiment 16)

### Objective
Test whether simpler hand-crafted acoustic features generalize better across speakers than WavLM embeddings.

### Hypothesis
Acoustic features (46D: prosodic + spectral + formant) might be more speaker-invariant than deep WavLM embeddings (256D), leading to better cross-speaker generalization.

### Implementation

1. **Fixed acoustic extraction script** (`scripts/extract_acoustic_features_batch.py`)
   - Bug: Script used relative paths that couldn't be found
   - Fix: Modified to construct absolute paths by joining `data_path` with `audio_path`
   - Result: 100% success rate for val (80 samples) and Viktor test (100 samples)

2. **Trained Experiment 16**:
   - Model: Qwen2.5-1.5B + LoRA (rank 8, 4-bit)
   - Features: Acoustic 46D
   - Data: 4-speaker training (800 samples), Viktor test (100 samples)
   - Training: 5 epochs, batch size 4, lr=5e-5

### Results

| Metric | Exp 13 (WavLM 256D) | Exp 16 (Acoustic 46D) | Winner |
|--------|---------------------|----------------------|--------|
| **Validation Loss** | 0.0316 | 0.0261 | Acoustic (-17%) |
| **Cross-Speaker (Viktor)** | 2.5336 | **5.2808** | **WavLM (-108%)** |
| **Degradation Factor** | 80.2× | 202.5× | WavLM |

### Critical Finding: The Feature-Generalization Paradox

**Acoustic features perform 17% BETTER in-domain but 108% WORSE for cross-speaker generalization.**

**Implication**:
- In-domain performance does NOT predict cross-speaker generalization
- Lower dimensionality does NOT guarantee better generalization
- Hand-crafted features do NOT automatically generalize better than learned features

**Root Cause Analysis**:
- Acoustic features (F0, formants) are inherently speaker-dependent
- Model likely overfit to specific F0 ranges and formant patterns of training speakers
- 46D may lack sufficient capacity to simultaneously encode emotion + speaker-invariance

**Document**: `EXPERIMENT_16_RESULTS.md`

---

## Task 2: WavLM Embedding Analysis

### Objective
Quantify how much speaker information is encoded in WavLM embeddings and determine if it causes cross-speaker generalization failure.

### Methodology

1. **Speaker Classification**: Logistic regression to predict speaker from embeddings
2. **Emotion Classification**: Logistic regression to predict emotion from embeddings
3. **Distance Analysis**: Within vs between speaker/emotion Euclidean distances
4. **Visualization**: t-SNE projection colored by speaker and emotion
5. **Cross-Speaker Transfer**: Emotion classification on Viktor (unseen speaker)

### Implementation

Created comprehensive analysis script: `scripts/analyze_wavlm_embeddings.py`
- Supports both WavLM and acoustic embeddings
- Computes classification accuracies, separation ratios, distance statistics
- Generates t-SNE visualizations
- Tests cross-speaker transfer performance

### Results: WavLM Embeddings (256D)

#### 1. Speaker Information Content

**Speaker Classification Accuracy: 95.0%** (vs 25% random baseline)
- Andrew Chipper: 91% precision, 100% recall
- Claribel Dervla: 97% precision, 95% recall
- Damien Black: 93% precision, 97% recall
- Gracie Wise: 100% precision, 88% recall

**Conclusion**: WavLM embeddings encode EXTREMELY HIGH speaker information.

#### 2. Emotion Information Content

**Emotion Classification Accuracy: 93.75%** (vs 16.67% random baseline)
- All emotions achieve 85-100% precision/recall
- Emotions are well-separated in embedding space

#### 3. The Speaker-Emotion Entanglement Problem

**Critical Discovery**:

| Metric | Speaker | Emotion | Ratio |
|--------|---------|---------|-------|
| Separation Ratio | 1.065× | 1.072× | 1.01× |

**Speaker and emotion information have NEARLY IDENTICAL separability** (1.065× vs 1.072×, only 1% difference).

**This means**:
- Speaker identity and emotional content are **ENTANGLED** in the embedding space
- Changes in speaker produce embedding changes as large as changes in emotion
- The model cannot distinguish "speaker variation" from "emotion variation"

#### 4. Cross-Speaker Catastrophic Failure

**Training Performance**: 93.75% emotion accuracy
**Viktor Test Performance**: 21.0% emotion accuracy (barely above 16.67% random)

**Performance Degradation: 77.6%**

**Breakdown by emotion on Viktor**:
- Anger, Fear, Joy, Neutral, Surprise: 0% recall (complete failure)
- Sadness: 100% recall (classifier collapsed to always predicting sadness)

**Root Cause**:
When the model encounters Viktor:
1. Viktor's embeddings occupy a different region than training speakers
2. The learned classifier was trained on training speakers' embedding regions
3. Viktor's embeddings fall outside these regions
4. The classifier has no basis for accurate prediction

**The model learned**: "Andrew's anger looks like X, Gracie's joy looks like Y" (speaker+emotion patterns)
**The model did NOT learn**: "Anger in general looks like Z" (speaker-invariant patterns)

### Key Insight

The entanglement was **invisible during validation** because all validation speakers were seen during training. The model appeared to learn emotion well (93.75% accuracy) but actually learned joint speaker-emotion patterns.

**Documents**:
- `EMBEDDING_ANALYSIS_RESULTS.md` - Comprehensive analysis report
- `outputs/wavlm_analysis_results.json` - Quantitative results
- `outputs/wavlm_tsne_visualization.png` - t-SNE plots
- `outputs/distance_distributions.png` - Distance analysis

---

## Task 3: Contrastive Learning Approach

### Objective
Design a training method that disentangles speaker identity from emotional content to enable cross-speaker generalization.

### Approach: Supervised Contrastive Learning

**Core Idea**: Explicitly train the emotion projection layer to produce representations where:
- **Positive pairs**: Same emotion, different speakers → embeddings should be similar
- **Negative pairs**: Different emotions (any speakers) → embeddings should be dissimilar

This directly addresses the entanglement by making speaker differences irrelevant while emphasizing emotion differences.

### Architecture

```
Audio → WavLM Encoder (frozen) → Emotion Projection → L2 Normalize → Contrastive Loss
                                                    ↓
                                               LLM Input
```

### Loss Function Design

**Supervised Contrastive Loss**:

For anchor sample i with emotion e_i and speaker s_i:
- Positive set P(i) = {j | emotion_j == e_i AND speaker_j != s_i}
- Negative set N(i) = {j | emotion_j != e_i}

```
Loss_i = -log [ sum_{p in P(i)} exp(sim(z_i, z_p) / τ) /
                sum_{j in P(i) ∪ N(i)} exp(sim(z_i, z_j) / τ) ]
```

**Critical Detail**: Positive set EXCLUDES same speaker. This forces learning of speaker-invariant patterns.

**Combined Loss**:
```
Total_Loss = λ_contrastive * Loss_contrastive + λ_lm * Loss_LM
```

### Implementation

#### Completed:

1. **Contrastive Loss Module** (`audio_augmented_llm/src/student_training/contrastive_loss.py`):
   - `SupervisedContrastiveLoss`: Base class with forward pass
   - `SupConLossWithStats`: Extended version that returns debugging statistics
   - Includes unit tests (✅ tested and passed)
   - Handles edge cases (e.g., no positive pairs in batch)

2. **Design Document** (`CONTRASTIVE_LEARNING_DESIGN.md`):
   - Complete architecture specification
   - Three experimental variants (17A, 17B, 17C)
   - Hyperparameter search strategy
   - Expected results (optimistic, realistic, pessimistic)
   - Success criteria and evaluation metrics

#### Remaining Work:

3. **Dataset Modifications**: Extend dataset to return emotion_label and speaker_id
4. **Model Modifications**: Extend StudentModel to support contrastive learning
5. **Training Script**: Create `scripts/train_exp17.py` for Experiment 17B (joint training)
6. **Training**: Run Experiment 17B and evaluate on Viktor test set
7. **Analysis**: Re-run embedding analysis to confirm disentanglement

### Recommended Experiment: 17B (Joint Training)

**Configuration**:
- Direct joint training (5 epochs)
- λ_contrastive = 0.3, λ_lm = 1.0
- Temperature τ = 0.07
- Batch size = 16 (4 samples per speaker to ensure cross-speaker pairs)
- Train both emotion projection and LoRA

**Success Criteria**:
- Viktor test emotion accuracy > 50% (vs current 21%)
- Training speaker classification < 70% (vs current 95%)
- Training emotion classification > 85%

**Expected Improvement**:
- Realistic: Viktor loss 2.5336 → 2.0 (20% improvement)
- Optimistic: Viktor loss 2.5336 → 1.5 (41% improvement)

---

## Scientific Contributions

### 1. The Feature-Generalization Paradox (Exp 16)

**Finding**: Better in-domain performance does NOT predict better cross-speaker generalization.

**Evidence**: Acoustic features (46D) achieved 17% better validation loss but 108% worse cross-speaker performance than WavLM (256D).

**Implication**: Dimensionality reduction and hand-crafted features are NOT sufficient for cross-speaker generalization.

### 2. The Speaker-Emotion Entanglement Problem (Task 2)

**Finding**: WavLM embeddings encode speaker identity (95% accuracy) and emotion (94% accuracy) with nearly equal separability (1.065× vs 1.072×).

**Evidence**:
- Speaker separation ratio: 1.065×
- Emotion separation ratio: 1.072×
- Difference: only 1.01×

**Implication**: Speaker and emotion are fundamentally entangled in WavLM's representation space. The model cannot distinguish speaker variation from emotion variation.

### 3. Cross-Speaker Generalization is a Disentanglement Problem

**Finding**: Cross-speaker generalization requires disentangling speaker identity from emotional content, not just reducing speaker information.

**Evidence**:
- Training accuracy (known speakers): 93.75%
- Viktor accuracy (unknown speaker): 21%
- The model learned speaker+emotion joint patterns, not speaker-invariant patterns

**Implication**: Solutions must explicitly train for speaker-invariance (e.g., contrastive learning, adversarial training), not just reduce dimensionality or use simpler features.

---

## Files Created

### Documentation
- `EXPERIMENT_16_RESULTS.md` - Acoustic vs WavLM comparison and paradox analysis
- `EMBEDDING_ANALYSIS_RESULTS.md` - Comprehensive embedding analysis report
- `CONTRASTIVE_LEARNING_DESIGN.md` - Complete contrastive learning design spec

### Code
- `scripts/analyze_wavlm_embeddings.py` - Embedding analysis tool (supports WavLM & acoustic)
- `audio_augmented_llm/src/student_training/contrastive_loss.py` - Contrastive loss implementation

### Data
- `audio_augmented_llm/data/exp13_4speaker/val/acoustic_embeddings.npz` - Validation acoustic features
- `audio_augmented_llm/data/test_cross_speaker_viktor/acoustic_embeddings.npz` - Viktor test acoustic features

### Results
- `outputs/wavlm_analysis_results.json` - Quantitative analysis results
- `outputs/wavlm_tsne_visualization.png` - Speaker/emotion t-SNE plots
- `outputs/distance_distributions.png` - Distance analysis visualization
- `outputs/wavlm_analysis_log.txt` - Analysis execution log
- `outputs/exp16_training_log.txt` - Experiment 16 training log
- `outputs/exp16_viktor_test_results.txt` - Experiment 16 cross-speaker results

### Models
- `audio_augmented_llm/models/exp16_acoustic_4speaker/best_model/` - Trained acoustic model

---

## Key Metrics Summary

| Experiment | Features | Val Loss | Viktor Test | Speaker Acc | Emotion Acc | Separation Ratio |
|-----------|----------|----------|-------------|-------------|-------------|------------------|
| Exp 13 | WavLM 256D | 0.0316 | 2.5336 | 95.0% | 93.8% | S: 1.065×, E: 1.072× |
| Exp 16 | Acoustic 46D | 0.0261 | **5.2808** | N/A | N/A | N/A |

**Cross-Speaker Degradation**:
- Exp 13: 80.2× (Val → Viktor)
- Exp 16: 202.5× (Val → Viktor)

**Viktor Emotion Accuracy**:
- Simple logistic regression on WavLM: 21% (from 94% on training speakers)
- Performance degradation: 77.6%

---

## Next Steps (Priority Order)

### Immediate (Next Session)

1. **Complete Contrastive Learning Implementation**:
   - Modify dataset to return `emotion_label` and `speaker_id`
   - Extend `StudentModel` to support L2 normalization and contrastive loss
   - Create `scripts/train_exp17.py` training script

2. **Run Experiment 17B**:
   - Train with joint contrastive + language modeling loss
   - Hyperparameters: λ_contrastive=0.3, λ_lm=1.0, τ=0.07, batch_size=16
   - Monitor: contrastive loss, within-emotion-cross-speaker distance, speaker classification accuracy

3. **Evaluate and Analyze**:
   - Test on Viktor cross-speaker set
   - Re-run embedding analysis to verify disentanglement
   - Compare speaker/emotion classification before vs after
   - Target: Viktor accuracy > 50%, speaker accuracy < 70%

### If Exp 17B Succeeds

4. **Hyperparameter Tuning**:
   - Adjust λ_contrastive (0.2, 0.3, 0.5, 0.7) to optimize trade-off
   - Test two-stage training (Exp 17C)
   - Try temperature variations (0.05, 0.07, 0.1)

5. **Paper Update**:
   - Add Experiment 16 results (Feature-Generalization Paradox)
   - Add embedding analysis findings (Speaker-Emotion Entanglement)
   - Add Experiment 17 results (Contrastive Learning Solution)

### If Exp 17B Fails or Shows Limited Improvement

6. **Alternative Approaches**:
   - Speaker adversarial training (gradient reversal)
   - Domain adaptation techniques
   - Meta-learning for cross-speaker transfer

7. **Task 4: Real Emotional Speech Data**:
   - Investigate IEMOCAP, RAVDESS, EmoV-DB datasets
   - Measure speaker-emotion entanglement in real vs TTS data
   - Test if TTS-generated data has artificial correlations

---

## Research Insights

### What We've Learned

1. **Dimensionality alone doesn't solve the problem**: 46D acoustic features performed worse than 256D WavLM
2. **Speaker information is extremely high**: 95% classification accuracy in WavLM
3. **Speaker and emotion are entangled**: Equal separability means they're indistinguishable
4. **In-domain metrics are misleading**: 94% emotion accuracy on training speakers, 21% on unseen speaker
5. **The problem is disentanglement, not just speaker information**: Need explicit training for speaker-invariance

### What Doesn't Work

1. ❌ Data scaling (Exp 13): Minimal cross-speaker improvement
2. ❌ Speaker Adaptive Normalization (Exp 15): Made it worse (+84%)
3. ❌ Acoustic features (Exp 16): Made it even worse (+108%)
4. ❌ Dimensionality reduction: Doesn't address entanglement

### What Should Work (Hypothesis)

1. ✅ Contrastive learning: Explicitly trains for speaker-invariance
2. ✅ Speaker adversarial training: Forces speaker-invariant representations
3. ✅ Real emotional speech data: May have less artificial entanglement

---

## Conclusion

This session made significant progress in understanding and addressing the cross-speaker generalization problem:

1. **Identified the root cause**: Speaker-emotion entanglement (not just speaker information)
2. **Disproved multiple hypotheses**: Acoustic features, dimensionality reduction
3. **Designed a targeted solution**: Supervised contrastive learning
4. **Implemented core components**: Contrastive loss function (tested and working)

The next session should focus on completing the contrastive learning implementation and testing whether explicit speaker-invariance training can solve the entanglement problem. If successful, this would represent a major breakthrough in cross-speaker generalization for emotion-aware LLMs.

**Current State**: Cross-speaker generalization is unsolved, but we now understand why and have a promising solution ready to implement.

---

**Session Duration**: Extensive (completed 3 major tasks)
**Token Usage**: ~100K (comprehensive analysis and implementation)
**Status**: Ready for Experiment 17 implementation and training
