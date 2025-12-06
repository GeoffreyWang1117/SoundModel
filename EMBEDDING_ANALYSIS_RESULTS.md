# Embedding Analysis Results: WavLM vs Acoustic Features

## Executive Summary

This analysis quantifies speaker information content in WavLM embeddings and provides critical insights into why cross-speaker generalization fails. The key finding: **WavLM embeddings encode extremely high speaker information (95% classification accuracy) that is entangled with emotion information, causing catastrophic failure on unseen speakers**.

---

## Analysis Setup

### Dataset
- **Training Data**: exp13_4speaker/train (800 samples)
  - Speakers: Andrew Chipper, Claribel Dervla, Damien Black, Gracie Wise
  - Emotions: anger, fear, joy, neutral, sadness, surprise
- **Test Data**: test_cross_speaker_viktor (100 samples)
  - Unseen speaker: Viktor Eka

### Methods
1. **Speaker Classification**: Logistic regression trained to predict speaker from embeddings
2. **Emotion Classification**: Logistic regression trained to predict emotion from embeddings
3. **Distance Analysis**: Within-speaker vs between-speaker Euclidean distances
4. **Visualization**: t-SNE projection colored by speaker and emotion
5. **Cross-Speaker Transfer**: Emotion classification performance on Viktor (unseen speaker)

---

## Results: WavLM Embeddings (256D)

### 1. Speaker Information Content

**Speaker Classification Accuracy: 95.0%**
- Baseline (random): 25.0%
- Information gain: 70.0%

**Conclusion**: WavLM embeddings encode **EXTREMELY HIGH** speaker information.

#### Per-Speaker Performance:
| Speaker | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Andrew Chipper | 0.91 | 1.00 | 0.95 |
| Claribel Dervla | 0.97 | 0.95 | 0.96 |
| Damien Black | 0.93 | 0.97 | 0.95 |
| Gracie Wise | 1.00 | 0.88 | 0.93 |

All speakers are highly distinguishable in the embedding space.

### 2. Emotion Information Content

**Emotion Classification Accuracy: 93.75%**
- Baseline (random): 16.67%
- Information gain: 77.08%

**Conclusion**: WavLM embeddings also encode strong emotion information.

#### Per-Emotion Performance:
| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Anger | 0.89 | 0.94 | 0.91 |
| Fear | 0.96 | 0.85 | 0.90 |
| Joy | 1.00 | 0.93 | 0.96 |
| Neutral | 0.95 | 0.95 | 0.95 |
| Sadness | 0.85 | 1.00 | 0.92 |
| Surprise | 1.00 | 0.96 | 0.98 |

Emotions are well-separated in the embedding space.

### 3. Distance Analysis

**Speaker-Based Distances:**
- Within-speaker mean: 0.3475
- Between-speaker mean: 0.3700
- **Separation ratio: 1.065×**

**Emotion-Based Distances:**
- Within-emotion mean: 0.3437
- Between-emotion mean: 0.3686
- **Separation ratio: 1.072×**

**Relative Information Content:**
- Speaker separation: 1.065×
- Emotion separation: 1.072×
- **Ratio**: Emotion has only **1.01× MORE** information than speaker

**Critical Finding**: Speaker and emotion information are **ALMOST EQUALLY SEPARABLE** in WavLM embedding space. The separation ratios are nearly identical (both ~1.07×), indicating that speaker identity and emotional content are **ENTANGLED**.

### 4. Cross-Speaker Generalization

**Training Set Emotion Accuracy**: 93.75%
**Test Set (Viktor) Emotion Accuracy**: 21.00%

- Baseline (random): 16.67%
- **Performance degradation: 77.6%** (from 93.75% to 21%)
- Viktor test accuracy is barely above random chance

#### Viktor Test Performance Breakdown:
| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Anger | 0.00 | 0.00 | 0.00 | 25 |
| Fear | 0.00 | 0.00 | 0.00 | 12 |
| Joy | 0.00 | 0.00 | 0.00 | 13 |
| Neutral | 0.00 | 0.00 | 0.00 | 17 |
| Sadness | 0.21 | 1.00 | 0.35 | 21 |
| Surprise | 0.00 | 0.00 | 0.00 | 12 |

**Critical Observation**: The classifier essentially collapsed to predicting only "sadness" for Viktor. This is a complete failure of generalization.

---

## Key Insights

### 1. The Speaker-Emotion Entanglement Problem

**Problem Statement**: WavLM embeddings simultaneously encode:
- **High speaker information** (95% classification accuracy)
- **High emotion information** (94% classification accuracy)
- **Nearly equal separability** (1.065× vs 1.072×)

This means the embedding space does NOT disentangle speaker identity from emotional content. Instead, they are **entangled**: changes in speaker produce changes in the embedding that are as large as changes in emotion.

### 2. Why Cross-Speaker Generalization Fails

When the model encounters Viktor (unseen speaker):
1. Viktor's embedding space occupies a **different region** than the 4 training speakers
2. The learned emotion classifier was trained on the **training speakers' embedding regions**
3. Viktor's embeddings fall **outside** these regions
4. The classifier has no basis for accurate emotion prediction

**Analogy**: The model learned "Andrew's anger looks like X, Gracie's joy looks like Y, etc." but never learned "anger in general looks like Z (regardless of speaker)".

### 3. Why This Wasn't Obvious from Training Performance

Training validation showed:
- Excellent emotion classification: 93.75%
- Low validation loss: 0.0316 (Exp 13)

The model appeared to learn emotion well, but it actually learned:
- **Joint speaker-emotion patterns**: "Andrew + anger", "Gracie + joy"
- NOT **speaker-invariant emotion patterns**: "anger (for any speaker)"

The entanglement was invisible during validation because all validation speakers were seen during training.

### 4. Comparison with Experiment Results

| Experiment | Features | Val Loss | Cross-Speaker (Viktor) | Speaker Info | Emotion Info |
|-----------|----------|----------|----------------------|--------------|--------------|
| Exp 13 | WavLM 256D | 0.0316 | 2.5336 | **95.0%** | 93.8% |
| Exp 16 | Acoustic 46D | 0.0261 | 5.2808 | Unknown* | Unknown* |

*Acoustic analysis failed due to data issues, but training results show even worse cross-speaker performance.

**Conclusion**: WavLM's high speaker information content (95%) explains why cross-speaker performance is poor, but acoustic features with even lower dimensional representation perform WORSE (5.2808 vs 2.5336), suggesting the problem is not just speaker information but also insufficient representation capacity.

---

## Implications for Future Work

### What This Analysis Confirms

1. ✅ **WavLM embeddings DO contain high speaker information** (95% accuracy)
2. ✅ **Speaker and emotion are entangled** (equal separation ratios)
3. ✅ **This entanglement causes cross-speaker generalization failure** (77.6% degradation)

### What This Rules Out

1. ❌ **Simply reducing dimensionality doesn't help** (acoustic 46D performed worse)
2. ❌ **The problem isn't "too much speaker info"** - it's **entangled speaker-emotion info**
3. ❌ **Current architectures cannot disentangle these factors** (all experiments failed)

### What We Need Next

The analysis points to specific solutions:

#### Immediate Priority: Disentangle Speaker from Emotion

**Contrastive Learning Approach** (Task 3):
1. **Positive pairs**: Same emotion, different speakers
   - Force embeddings for "Andrew-anger" and "Gracie-anger" to be similar
2. **Negative pairs**: Different emotions (regardless of speaker)
   - Force embeddings for "Andrew-anger" and "Andrew-joy" to be dissimilar
3. **Goal**: Maximize emotion separability WHILE minimizing speaker separability

This directly addresses the entanglement problem by explicitly training for speaker-invariant emotion representations.

#### Alternative Approaches to Consider:

1. **Speaker Adversarial Training**:
   - Add a speaker classifier with gradient reversal
   - Train to maximize emotion classification, minimize speaker classification
   - Forces embeddings to be speaker-invariant

2. **Domain Adaptation Techniques**:
   - Treat each speaker as a "domain"
   - Use techniques like Domain-Adversarial Neural Networks (DANN)
   - Align speaker distributions while preserving emotion information

3. **Meta-Learning**:
   - Train on "speaker-held-out" tasks
   - Force model to learn from speaker diversity during training
   - Explicitly optimize for cross-speaker transfer

4. **Real Emotional Speech Data** (Task 4):
   - Current TTS-generated data may have artificial speaker-emotion correlations
   - Real human emotional speech may have different entanglement properties
   - Investigate datasets: IEMOCAP, RAVDESS, EmoV-DB

---

## Visualizations Generated

1. **t-SNE Visualization** (`outputs/wavlm_tsne_visualization.png`):
   - Left: Colored by speaker (shows clear speaker clustering)
   - Right: Colored by emotion (shows emotion patterns)
   - Visual confirmation of speaker-emotion entanglement

2. **Distance Distributions** (`outputs/distance_distributions.png`):
   - Left: Within-speaker vs between-speaker distances
   - Right: Within-emotion vs between-emotion distances
   - Shows similar separation for both factors

---

## Recommendations

### Short-Term (Next Experiment)

**Priority 1**: Implement contrastive learning approach (Task 3)
- Design: Supervised contrastive loss with speaker-agnostic emotion pairs
- Hypothesis: Will reduce speaker separability ratio while maintaining emotion separability
- Success metric: Viktor test emotion accuracy > 50% (vs current 21%)

**Priority 2**: Analyze impact of contrastive learning
- Re-run this analysis script on contrastive-learned embeddings
- Compare speaker/emotion classification accuracies before vs after
- Target: Speaker accuracy < 60%, Emotion accuracy > 85%

### Long-Term

**Priority 3**: Investigate real emotional speech datasets (Task 4)
- Measure speaker-emotion entanglement in real vs TTS data
- May reveal fundamental limitations of TTS approach

**Priority 4**: Hybrid approaches
- Combine contrastive learning + adversarial training
- Use both TTS and real data
- Multi-objective optimization: emotion classification + speaker invariance

---

## Conclusion

This analysis provides definitive evidence that **WavLM embeddings contain high speaker information (95% accuracy) that is entangled with emotion information**, explaining the catastrophic cross-speaker generalization failure observed in all experiments.

The next step is clear: **Design and implement a contrastive learning approach to disentangle speaker identity from emotional content**. This is the most direct solution to the identified problem and should be the focus of the next experiment.

---

**Files Generated:**
- Analysis results: `outputs/wavlm_analysis_results.json`
- Visualizations:
  - `outputs/wavlm_tsne_visualization.png`
  - `outputs/distance_distributions.png`
- Logs: `outputs/wavlm_analysis_log.txt`
