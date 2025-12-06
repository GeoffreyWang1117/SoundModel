# Session Summary - 2025-11-27

## Work Completed

### 1. Experiment 15: Speaker Adaptive Normalization (SAN)

#### Implementation
- **Created new architecture**: `audio_augmented_llm/src/student_training/speaker_adaptive_norm.py`
  - `SpeakerAdaptiveNorm`: Speaker-conditioned normalization layer
  - `SpeakerConditionedProjection`: Emotion projection with adaptive normalization

- **Extended student model**: `audio_augmented_llm/src/student_training/student_model_san.py`
  - `StudentModelWithSAN`: Student model with speaker adaptive normalization
  - Supports training with speaker IDs, inference without conditioning

- **Updated dataset**: `audio_augmented_llm/src/student_training/dataset.py`
  - Added speaker_id support in `__getitem__` and `collate_fn`

- **Training script**: `scripts/train_exp15.py`
  - Dual evaluation: with and without speaker IDs
  - Tracks speaker-invariance gap during validation

#### Training Results
- **Configuration**: 4 speakers, 800 train + 80 val samples, WavLM 256D
- **Validation Loss**: 0.0863 (worse than Exp 13's 0.0316)
- **Speaker-Invariance**: Perfect (gap = 1.00×)
- **Cross-Speaker Test (Viktor)**: 4.6717 (84% worse than Exp 13's 2.5336)

#### Critical Finding: The Speaker-Invariance Paradox
**Discovery**: Achieving perfect speaker-invariance does NOT lead to better cross-speaker generalization.

**Key Insight**:
> Validation gap metric is misleading - it measures consistency among known speakers, not generalization to unknown speakers.

**Root Causes**:
1. **False equivalence**: Speaker-invariance ≠ speaker-agnostic
2. **Harmful averaging**: Average of speaker parameters encodes bias toward training speakers
3. **Information loss**: Complete speaker-invariance removes useful variance
4. **Complexity cost**: Additional parameters worsened baseline performance

**Recommendation**: Abandon SAN approach - fundamentally flawed.

---

### 2. Documentation

#### EXPERIMENT_15_RESULTS.md
Comprehensive analysis document covering:
- Full results breakdown
- The Speaker-Invariance Paradox
- Root cause analysis
- Comparison with baseline (Exp 13)
- Key insights and lessons learned
- Future directions

---

### 3. Paper Updates

#### Added to Results Section (5_results.tex)

**New Subsection: The Speaker-Performance Paradox (Exp 13)**
- Table comparing 2-speaker vs 4-speaker training
- Counterintuitive finding: Better same-speaker performance (87.2% improvement) does not lead to better cross-speaker generalization (only 0.65% improvement)
- Degradation factor worsened: 32.2× → 80.2×
- Explanation of the paradox

**New Subsection: Speaker Adaptive Normalization Failure (Exp 15)**
- Table showing SAN results
- Perfect validation gap (1.00×) vs catastrophic test performance (+84%)
- Root cause analysis
- Key insight about misleading validation metrics

#### System Architecture Figure (figures/architecture.tex)
- Created comprehensive TikZ diagram showing 3-stage pipeline
- Stage 1: Data generation (GPT-4 → XTTS → Audio)
- Stage 2: Feature extraction (Acoustic 46D vs WavLM 256D)
- Stage 3: Student training (Projection → Qwen2.5-1.5B + LoRA)
- Compiled to PDF: `figures/architecture.pdf`
- Added to Method section (3_method.tex) with Figure \ref{fig:architecture}

#### Paper Compilation
- Successfully compiled to 14-page PDF
- Minor Unicode warnings (≠ symbol) but PDF generated correctly
- Location: `paper/main.pdf`

---

### 4. Evaluation Script Fix

#### scripts/evaluate_model.py
- Updated to support both standard and SAN models
- Detects `use_adaptive_norm` from training_config.json
- Loads `StudentModelWithSAN` when needed
- Handles `SpeakerConditionedProjection` state_dict correctly

---

## Experimental Timeline

1. **Exp 13** (4-speaker baseline): Revealed Speaker-Performance Paradox
   - Val loss: 0.0316 (87% better than 2-speaker)
   - Cross-speaker: 2.5336 (only 0.65% better than 2-speaker)
   - Degradation: 80.2× (much worse than 2-speaker's 32.2×)

2. **Exp 15** (Speaker Adaptive Normalization): Failed catastrophically
   - Perfect speaker-invariance achieved (1.00× gap)
   - Cross-speaker loss: 4.6717 (84% worse than Exp 13)
   - Proved that speaker-invariance ≠ cross-speaker generalization

---

## Key Scientific Contributions

### Negative Results (Valuable!)

1. **Data Scaling Doesn't Solve the Problem**
   - Adding more speakers improves same-speaker fit but not cross-speaker generalization
   - More speakers can actually worsen the degradation factor

2. **Speaker-Invariance is NOT the Solution**
   - Explicit speaker conditioning + averaging fails
   - Removing all speaker information is harmful
   - Validation metrics can be highly misleading

3. **Architecture Complexity Has Costs**
   - SAN added parameters but worsened baseline performance
   - Simple approaches (Exp 13) outperform complex normalization

### Insights for Future Work

1. **Problem is Deeper Than Architecture**
   - Both data scaling and architectural solutions failed
   - May require fundamentally different approach

2. **Potential Directions**:
   - Investigate WavLM embeddings (may encode too much speaker info)
   - Try contrastive learning for speaker-robust features
   - Data augmentation: mix speaker characteristics
   - Meta-learning: train on speaker-held-out tasks

3. **What Works (Relatively)**:
   - Simple baselines without explicit speaker conditioning
   - Acoustic features (46D) outperform WavLM (256D) for in-domain
   - Small improvement from multi-speaker training (16.4%)

---

## Files Created/Modified

### New Files
- `audio_augmented_llm/src/student_training/speaker_adaptive_norm.py`
- `audio_augmented_llm/src/student_training/student_model_san.py`
- `scripts/train_exp15.py`
- `EXPERIMENT_15_RESULTS.md`
- `paper/figures/architecture.tex`
- `paper/figures/architecture.pdf`

### Modified Files
- `audio_augmented_llm/src/student_training/dataset.py` (added speaker_id support)
- `scripts/evaluate_model.py` (added SAN model support)
- `paper/sections/5_results.tex` (added Exp 13 and 15 results)
- `paper/sections/3_method.tex` (added architecture figure)

### Output Files
- `outputs/exp15_training_log.txt`
- `outputs/exp15_viktor_test_results.txt`
- `audio_augmented_llm/models/exp15_san/best_model/`
  - Trained model weights
  - training_config.json
  - emotion_projection.pt

---

## Current State

### Completed Tasks ✅
- [x] Design and implement Speaker Adaptive Normalization
- [x] Train Exp 15 model with 4-speaker dataset
- [x] Evaluate on Viktor cross-speaker test set
- [x] Analyze failure and document findings
- [x] Add Exp 13 and Exp 15 results to paper
- [x] Create system architecture diagram
- [x] Update paper and compile successfully

### Paper Status
- **Sections**: Complete (Introduction, Method, Results, Discussion, Conclusion)
- **Figures**: Architecture diagram added
- **Tables**: All experimental results documented
- **Compilation**: Successful (14 pages)

### Research Status
- **In-domain performance**: SOLVED (acoustic features work well)
- **Cross-speaker generalization**: **UNSOLVED** - Major open problem
- **Attempted solutions**:
  - ❌ Data scaling (Exp 13): Minimal improvement
  - ❌ Speaker Adaptive Normalization (Exp 15): Made it worse
  - ❌ Relative features (Exp 10): Didn't help

---

## Next Steps (Recommendations)

### Immediate
1. Test acoustic features (46D) on cross-speaker task
   - Exp 15 used WavLM (256D)
   - Acoustic features performed better in-domain
   - May be more speaker-invariant

2. Analyze WavLM embeddings
   - Extract embeddings from different speakers saying same text
   - Measure speaker information content
   - May be encoding too much speaker identity

### Short-term
3. Try contrastive learning approach
   - Positive pairs: same emotion, different speakers
   - Negative pairs: different emotions
   - Force speaker-robust emotion representations

4. Data augmentation experiments
   - Mix prosodic features across speakers
   - Generate synthetic "hybrid" speakers
   - Test if it improves cross-speaker robustness

### Long-term
5. Fundamental rethink
   - Question: Is TTS the right approach?
   - Alternative: Use real emotional speech datasets
   - May need human-recorded data for true generalization

---

## Summary

This session completed Experiment 15 (Speaker Adaptive Normalization), which provided valuable negative results demonstrating that forcing speaker-invariance through explicit conditioning does not solve cross-speaker generalization. The work has been fully documented, added to the paper with a comprehensive architecture diagram, and the paper compiles successfully.

**Key takeaway**: Cross-speaker generalization remains a fundamental unsolved challenge that likely requires approaches beyond current architecture and data scaling strategies.
