# Experiment 16: Acoustic Features for Cross-Speaker Generalization

## Hypothesis

Since acoustic features (46D) outperformed WavLM (256D) for in-domain tasks in Experiment 7, we hypothesized they might also generalize better across speakers due to being more interpretable, lower-dimensional, and potentially containing less speaker-specific information.

## Setup

- **Model**: Qwen2.5-1.5B with LoRA (rank 8, 4-bit quantization)
- **Emotion Features**: Acoustic 46D (prosodic 13D + spectral 20D + formant 13D)
- **Integration**: Concat strategy (prepend emotion embedding to text)
- **Training Data**: 4-speaker dataset (800 samples)
  - Speakers: Claribel Dervla, Damien Black, Andrew Chipper, Gracie Wise
- **Validation Data**: 4-speaker validation (80 samples)
- **Test Data**: Viktor cross-speaker test (100 samples, completely unseen speaker)

## Training Configuration

```json
{
  "embedding_type": "acoustic",
  "emotion_dim": 46,
  "num_epochs": 5,
  "batch_size": 4,
  "learning_rate": 5e-5,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "quantization": "4bit"
}
```

## Results

### Training Performance

| Epoch | Train Loss | Val Loss | Notes |
|-------|-----------|----------|-------|
| 1 | 0.1049 | 0.0309 | - |
| 2 | 0.0748 | 0.0285 | - |
| 3 | 0.0653 | 0.0277 | - |
| 4 | 0.0607 | 0.0270 | - |
| 5 | 0.0569 | **0.0261** | Best model |

**Final validation loss**: 0.0261

### Cross-Speaker Test (Viktor)

**Test loss**: 5.2808
- Samples evaluated: 100
- Model: Best from epoch 5

## Comparison: Experiment 16 vs Experiment 13

| Metric | Exp 13 (WavLM 256D) | Exp 16 (Acoustic 46D) | Change | Winner |
|--------|---------------------|----------------------|---------|--------|
| **Validation Loss** | 0.0316 | 0.0261 | -17.4% | **Acoustic** |
| **Cross-Speaker Test (Viktor)** | 2.5336 | 5.2808 | **+108.5%** | **WavLM** |
| **Degradation Factor** | 80.2× | 202.5× | **+152.4%** | **WavLM** |

**Degradation Factor** = Cross-speaker loss / Validation loss

### Key Findings

1. **In-Domain Performance**: Acoustic features achieve 17% better validation loss
2. **Cross-Speaker Performance**: Acoustic features achieve 108% WORSE test loss
3. **Generalization Gap**: Acoustic features have 2.5× worse degradation factor

## Critical Insight: The Feature-Generalization Paradox

**Discovery**: Better in-domain performance does NOT predict better cross-speaker generalization.

### The Paradox

```
In-Domain:  Acoustic (46D) > WavLM (256D)  [Val: 0.0261 vs 0.0316]
Cross-Speaker: WavLM (256D) > Acoustic (46D)  [Test: 2.5336 vs 5.2808]
```

This reveals a fundamental trade-off:
- **Acoustic features**: Better for fitting known speakers, worse for generalizing to new speakers
- **WavLM features**: Worse for fitting known speakers, better for generalizing to new speakers

## Why Acoustic Features Fail for Cross-Speaker

### Hypothesis 1: Speaker-Specific Feature Extraction
Acoustic features like F0 (pitch), formants, and spectral characteristics are inherently speaker-dependent:
- **F0 range**: Males (85-180 Hz), Females (165-255 Hz)
- **Formant frequencies**: Depend on vocal tract length
- **Spectral envelope**: Shaped by individual voice timbre

The model may overfit to the specific F0 ranges and formant patterns of the 4 training speakers.

### Hypothesis 2: Lack of Learned Normalization
WavLM embeddings are learned from massive multi-speaker data and may have implicit speaker normalization built in. Acoustic features are raw measurements without such normalization.

### Hypothesis 3: Insufficient Representation Capacity
46D may not have enough capacity to simultaneously encode:
1. Emotion-specific patterns
2. Speaker-invariant characteristics
3. Linguistic content correlation

WavLM's 256D provides more room for disentangling these factors.

### Hypothesis 4: Missing Contextual Information
Acoustic features are frame-level statistics. WavLM captures temporal dependencies and phonetic context through its transformer architecture, which may be crucial for speaker-robust emotion encoding.

## Why WavLM Generalizes Better (Despite Hypothesis)

Prior to this experiment, we hypothesized that WavLM embeddings might contain "too much speaker information" and thus generalize poorly. **This experiment disproves that hypothesis.**

### Possible Explanations

1. **Learned Speaker Robustness**: Pre-trained on diverse speakers, WavLM may have learned speaker-invariant speech representations
2. **Contextual Encoding**: Transformer-based features capture prosodic patterns relative to speaker baseline rather than absolute values
3. **Higher-Level Abstractions**: 256D embeddings may encode emotion at a more abstract level that transfers across speakers
4. **Implicit Disentanglement**: WavLM may naturally separate speaker identity from emotional content

## Implications for Future Work

### What This Rules Out
1. ❌ **Dimensionality reduction**: Lower-dimensional features don't improve cross-speaker generalization
2. ❌ **Hand-crafted features**: Domain knowledge doesn't guarantee better generalization
3. ❌ **In-domain validation**: Cannot use validation loss as proxy for cross-speaker performance

### What This Suggests
1. ✅ **Deep embeddings are necessary**: Pre-trained representations capture crucial information for generalization
2. ✅ **WavLM contains valuable patterns**: Despite encoding speaker info, it has speaker-robust emotion signals
3. ✅ **Need better understanding**: Next step is to analyze WHAT makes WavLM generalize better

## Next Experiments

Based on these findings, the research plan prioritizes:

### Immediate (Task 2)
**Analyze WavLM embeddings for speaker information content**
- Quantify how much speaker information is actually encoded
- Measure emotion vs speaker separability in embedding space
- Understand what makes WavLM generalize despite speaker information

This will inform whether we need:
- Speaker adversarial training
- Contrastive learning (Task 3)
- Better feature disentanglement approaches

### If Analysis Shows High Speaker Information
- Design contrastive learning to force speaker-invariance
- Implement speaker adversarial training
- Test speaker ID classification + gradient reversal

### If Analysis Shows Low Speaker Information
- WavLM is already speaker-robust
- Focus on other factors (data quality, model architecture, training strategy)
- Consider using real emotional speech datasets (Task 4)

## Conclusion

Experiment 16 provides a critical negative result: **acoustic features, despite being better for in-domain performance, are significantly worse for cross-speaker generalization**. This disproves our initial hypothesis and reveals that WavLM's deep learned representations contain valuable patterns for speaker-robust emotion encoding.

The next crucial step is to understand WHY WavLM generalizes better by analyzing its embeddings for speaker information content (Task 2), which will guide the design of improved approaches.

---

**Model Location**: `./audio_augmented_llm/models/exp16_acoustic_4speaker/best_model/`
**Training Log**: `outputs/exp16_training_log.txt`
**Test Results**: `outputs/exp16_viktor_test_results.txt`
