# Experiment 9: Generalization Testing

## Experiment 9a: Cross-Speaker Generalization

**Date**: 2025-11-15
**Objective**: Test if acoustic features trained on one TTS speaker generalize to different TTS speakers

### Setup
- **Training speaker**: Claribel Dervla (female voice from XTTS v2)
- **Test speaker**: Damien Black (male voice from XTTS v2)
- **Model**: Best checkpoint from Exp 7b (acoustic features, 46D)
- **Test set size**: 100 samples
- **Test set generation**: Same texts as training set, re-synthesized with different speaker

### Results

| Test Set | Val Loss | vs Baseline | Speaker |
|----------|----------|-------------|---------|
| In-domain | **0.0792** | - | Claribel Dervla |
| Cross-speaker | **3.0502** | +3752% | Damien Black |

### Analysis

**Key Finding**: **Catastrophic generalization failure across speakers**

The model shows a massive performance degradation when tested on a different TTS speaker:
- **38.5× worse loss** on cross-speaker test (3.0502 vs 0.0792)
- Loss increases by **3752%** compared to in-domain performance

**Interpretation**:

1. **Speaker-specific overfitting**: The acoustic features (pitch, MFCCs, formants) are capturing speaker-specific voice characteristics rather than generalizable emotion patterns

2. **Voice-dependent features**: Prosodic features like pitch mean/range and formants are inherently speaker-dependent:
   - Female voices (Claribel): ~200-400 Hz fundamental frequency
   - Male voices (Damien): ~100-150 Hz fundamental frequency
   - These absolute values differ by gender/speaker, not just by emotion

3. **Model learned speaker identity, not emotion**: The model appears to have memorized the specific acoustic signature of Claribel Dervla rather than learning emotion-invariant patterns

**Implications**:

This result challenges the hypothesis that hand-crafted acoustic features generalize better than learned representations like WavLM. It suggests:

- **Normalization needed**: Acoustic features may need speaker normalization (z-score within speaker, relative pitch changes vs absolute values)
- **WavLM advantage**: Deep learned representations may actually capture more speaker-invariant emotion patterns
- **Feature engineering gap**: Simple statistical aggregation (mean, std) loses temporal dynamics that convey emotion independently of voice characteristics

### Comparison with Literature

- Human emotion recognition generalizes ~60-80% across speakers
- This result (3.0502 vs 0.0792 = 2.6% relative performance) is far below human-level generalization
- Suggests current approach needs significant improvement for real-world applicability

### Next Steps

1. **Test WavLM generalization**: Does WavLM show similar degradation or better cross-speaker transfer?
2. **Speaker normalization**: Apply z-score normalization per speaker before feature aggregation
3. **Relative features**: Use pitch contours (deltas) rather than absolute pitch values
4. **Multi-speaker training**: Train on diverse speakers to force learning of speaker-invariant patterns
