# Session Summary - 2025-11-27 (Part 4)

## Overview

This session continued from Experiment 17B evaluation and pivoted to using real emotional speech datasets (RAVDESS) to address the cross-speaker generalization problem after the contrastive learning approach failed due to representation collapse.

---

## Work Completed

### 1. Experiment 17B vs Exp 13 Comparison

| Metric | Exp 13 (Baseline) | Exp 17B (Contrastive) | Change |
|--------|-------------------|----------------------|--------|
| **Val Loss** | 0.0316 | 0.0152 | -52% (better) |
| **Viktor Test Loss** | 2.5336 | **3.1465** | **+24% (worse)** |
| **Degradation Factor** | 80.2× | 207.0× | **+158% (worse)** |

#### Critical Finding: Experiment 17B Made Cross-Speaker Performance WORSE

**Why Exp 17B Failed:**
- **Representation collapse**: Positive similarity (0.52-0.58) ≈ Negative similarity (0.52-0.58)
- **Too few positive pairs**: Only 1.4-1.6 avg positive pairs per batch (batch size 4, 4 speakers)
- **Contrastive loss provided no useful gradient signal**
- **L2 normalization may have destroyed emotion information**

### 2. Pivot to Real Emotional Speech Datasets

After analyzing the failure, proposed 4 alternative approaches:
1. Data augmentation + more speakers
2. **Real emotional speech datasets** (IEMOCAP, RAVDESS, EmoV-DB) - **SELECTED BY USER**
3. Simplified emotion representation with regularization
4. Multi-task learning

User explicitly requested: "使用方案2吧" (Use approach 2)

### 3. Dataset Research and Selection

Researched three major emotional speech datasets:

| Dataset | Speakers | Emotions | Access | License | Recommendation |
|---------|----------|----------|--------|---------|----------------|
| **IEMOCAP** | 10 actors | 4 main + dimensional | Requires USC registration | Research only | Delayed access |
| **RAVDESS** | **24 actors (12M/12F)** | **7 emotions × 2 intensities** | **FREE immediate** | **CC BY-NC-SA 4.0** | **✅ SELECTED** |
| **EmoV-DB** | 4 speakers (2M/2F) | 5 emotions | MEGA/OpenSLR | Non-commercial | Too few speakers |

**Decision: RAVDESS** because:
- Immediately accessible (no approval needed)
- 24 speakers (vs 10 for IEMOCAP, 4 for EmoV-DB)
- Excellent for cross-speaker splits (e.g., train on 16, val on 4, test on 4)
- Free and open (CC BY-NC-SA 4.0)
- Well-documented and widely used in research
- Gender balanced

### 4. RAVDESS Dataset Download (In Progress)

**Download Started**: 2025-11-27 21:56 UTC
**Source**: https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip
**Total Size**: 208.5 MB
**Current Progress**: 8.2 MB (as of 22:14 UTC)
**Status**: ⏳ Downloading in background

**Contents**:
- 1,440 audio files (60 trials per actor × 24 actors)
- Format: 16-bit, 48kHz WAV
- Emotions: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised (8 total)
- Intensities: Normal, Strong
- Statements: "Kids are talking by the door", "Dogs are sitting by the door"
- 2 repetitions per statement

### 5. Created Data Processing Pipeline

#### 5.1 Dataset Organization Script

**File**: `scripts/process_ravdess_dataset.py`

**Features**:
- Parses RAVDESS filename convention (Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor)
- Extracts metadata (emotion, speaker, gender, statement, intensity)
- Organizes into train/val/test splits based on speaker IDs
- Default split: Train (actors 1-16), Val (actors 17-20), Test (actors 21-24)
- Copies files with descriptive names: `{idx:04d}_{emotion}_{actor:02d}.wav`
- Generates metadata.json and summary.json for each split

**Usage**:
```bash
python scripts/process_ravdess_dataset.py \
    --ravdess_dir ./audio_augmented_llm/data/ravdess/Audio_Speech_Actors_01-24 \
    --output_dir ./audio_augmented_llm/data/ravdess_processed \
    --train_actors "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16" \
    --val_actors "17,18,19,20" \
    --test_actors "21,22,23,24"
```

**Expected Output**:
- Train: 960 samples (16 speakers × 60 trials)
- Val: 240 samples (4 speakers × 60 trials)
- Test: 240 samples (4 speakers × 60 trials)

#### 5.2 WavLM Embedding Extraction Script

**File**: `scripts/extract_wavlm_ravdess.py`

**Features**:
- Loads microsoft/wavlm-base-plus model
- Extracts 768D embeddings from audio (will project to 256D later)
- Mean pooling over last hidden state
- Handles resampling (48kHz → 16kHz), stereo → mono conversion
- Pads/trims to 3 seconds
- Saves embeddings and texts to `emotion_embeddings.npz` and `texts.npy`
- Text format: `[{EMOTION}] {statement}` (e.g., `[ANGRY] Kids are talking by the door`)

**Usage**:
```bash
source /home/coder-gw/miniconda3/etc/profile.d/conda.sh && conda activate audio_llm
python scripts/extract_wavlm_ravdess.py \
    --ravdess_processed_dir ./audio_augmented_llm/data/ravdess_processed \
    --device cuda
```

---

## Next Steps (Priority Order)

### Immediate (Once Download Completes)

1. **Extract and Organize RAVDESS** (~5 minutes)
   ```bash
   cd audio_augmented_llm/data/ravdess
   unzip Audio_Speech_Actors_01-24.zip
   cd ../../..

   python scripts/process_ravdess_dataset.py \
       --ravdess_dir ./audio_augmented_llm/data/ravdess/Audio_Speech_Actors_01-24 \
       --output_dir ./audio_augmented_llm/data/ravdess_processed
   ```

2. **Extract WavLM Embeddings** (~10-15 minutes with GPU)
   ```bash
   source /home/coder-gw/miniconda3/etc/profile.d/conda.sh && conda activate audio_llm
   export CUDA_VISIBLE_DEVICES=1

   python scripts/extract_wavlm_ravdess.py \
       --ravdess_processed_dir ./audio_augmented_llm/data/ravdess_processed \
       --device cuda
   ```

3. **Train on RAVDESS (Experiment 18)** (~10-15 minutes)
   ```bash
   python scripts/train_exp11.py \
       --train_dir ./audio_augmented_llm/data/ravdess_processed/train \
       --val_dir ./audio_augmented_llm/data/ravdess_processed/val \
       --embedding_type wavlm \
       --emotion_dim 256 \
       --num_epochs 5 \
       --batch_size 8 \
       --learning_rate 5e-5 \
       --output_dir ./audio_augmented_llm/models/exp18_ravdess
   ```

4. **Evaluate Cross-Speaker Performance** (~1 minute)
   ```bash
   python scripts/evaluate_model.py \
       --model_dir ./audio_augmented_llm/models/exp18_ravdess/best_model \
       --data_dir ./audio_augmented_llm/data/ravdess_processed/test \
       --embedding_type wavlm
   ```

5. **Analyze Results**
   - Compare RAVDESS test loss vs TTS test loss
   - Check if speaker-emotion entanglement is reduced
   - Re-run embedding analysis on RAVDESS data
   - Determine if real data improves cross-speaker generalization

### If RAVDESS Shows Improvement

6. **Compare TTS vs Real Data**
   | Metric | TTS (Exp 13) | Real (Exp 18) | Improvement |
   |--------|-------------|---------------|-------------|
   | Val Loss | 0.0316 | ? | ? |
   | Test Loss (Cross-Speaker) | 2.5336 | ? | ? |
   | Degradation Factor | 80.2× | ? | ? |
   | Speaker Classification | 95% | ? | ? |
   | Emotion Classification | 94% | ? | ? |

7. **Analyze Speaker-Emotion Entanglement in Real Data**
   ```bash
   python scripts/analyze_wavlm_embeddings.py \
       --data_dir ./audio_augmented_llm/data/ravdess_processed/train \
       --test_dir ./audio_augmented_llm/data/ravdess_processed/test \
       --embedding_type wavlm \
       --output_dir ./outputs
   ```

8. **Document Findings**
   - Create `EXPERIMENT_18_RESULTS.md`
   - Update `PROGRESS_SUMMARY.md`
   - Add to paper if significant improvement

### If RAVDESS Doesn't Show Improvement

9. **Investigate Alternative Approaches**
   - Try contrastive learning on RAVDESS (larger batch sizes possible)
   - Implement speaker adversarial training (gradient reversal layer)
   - Domain adaptation techniques (DANN)
   - Meta-learning for cross-speaker transfer

10. **Consider Hybrid Approach**
    - Pre-train on large TTS dataset
    - Fine-tune on small real dataset
    - Test on unseen real speakers

---

## Key Research Questions for Experiment 18

1. **Is TTS-generated data the problem?**
   - Hypothesis: TTS creates artificial speaker-emotion correlations
   - Test: Do real emotional speech datasets have less entanglement?

2. **How much does real data improve cross-speaker generalization?**
   - Baseline: TTS test loss = 2.5336 (80.2× degradation)
   - Target: Real test loss < 2.0 (< 50× degradation)

3. **Is speaker-emotion entanglement dataset-dependent?**
   - TTS: Speaker separation 1.065×, Emotion separation 1.072× (entangled)
   - Real: ? (to be measured)

4. **What is the optimal training set size?**
   - RAVDESS train: 960 samples (16 speakers)
   - TTS train (Exp 13): 800 samples (4 speakers)
   - More speakers vs more samples per speaker?

---

## Files Created/Modified

### Created
- `scripts/process_ravdess_dataset.py` - RAVDESS data organization pipeline
- `scripts/extract_wavlm_ravdess.py` - WavLM embedding extraction for RAVDESS
- `audio_augmented_llm/data/ravdess/` - RAVDESS download directory
- `SESSION_SUMMARY_2025-11-27_PART4.md` - This document

### Modified
- None (scripts are new)

---

## Current Work State

**RAVDESS Download**: ⏳ In progress (8.2 MB / 208.5 MB as of 22:14 UTC)
**Processing Scripts**: ✅ Complete and ready to use
**Next Action**: Wait for download to complete, then process and train

**Estimated Time to Completion**:
- Download: ~30-60 minutes (depends on connection)
- Processing: ~5 minutes
- Embedding extraction: ~15 minutes
- Training: ~15 minutes
- **Total**: ~1-1.5 hours from now

---

## Todo List Status

| Task | Status |
|------|--------|
| Research available real emotional speech datasets | ✅ Completed |
| Download RAVDESS dataset from Zenodo | ⏳ In progress (8.2 MB / 208.5 MB) |
| Create data processing pipeline | ✅ Completed |
| Create WavLM extraction script | ✅ Completed |
| Extract zip file | ⏸ Pending (waiting for download) |
| Organize RAVDESS dataset structure | ⏸ Pending |
| Extract WavLM embeddings from RAVDESS audio | ⏸ Pending |
| Train model on RAVDESS dataset (Exp 18) | ⏸ Pending |
| Evaluate cross-speaker performance on RAVDESS | ⏸ Pending |
| Compare RAVDESS vs TTS results | ⏸ Pending |
| Analyze speaker-emotion entanglement in real data | ⏸ Pending |

---

## Experiment 17B Post-Mortem

### Why Contrastive Learning Failed

1. **Insufficient Cross-Speaker Pairs**
   - Batch size 4 with 4 speakers = ~1 sample per speaker
   - Average 1.4-1.6 positive pairs per batch
   - Need minimum 4-8 positive pairs for effective contrastive learning

2. **Representation Collapse**
   - Positive similarity: 0.52-0.58
   - Negative similarity: 0.52-0.58
   - All embeddings compressed to same region
   - Contrastive loss provided no useful gradient signal

3. **L2 Normalization Issue**
   - Forced all embeddings to unit sphere
   - May have destroyed emotion information
   - Distance-based similarity metric insufficient

### Lessons Learned

1. **Batch size matters for contrastive learning**
   - Need enough samples to form meaningful positive/negative pairs
   - 4-speaker × 4 batch size is insufficient
   - Should try 8-16 speaker × 16-32 batch size

2. **TTS data may have fundamental limitations**
   - Perfect execution creates artificial speaker-emotion correlations
   - Real speech has natural variation that may help generalization

3. **Architecture choices have consequences**
   - L2 normalization appropriate for speaker verification
   - May not be appropriate for emotion-aware language modeling

---

## Scientific Contributions (Cumulative)

### 1. The Feature-Generalization Paradox (Exp 16)
Better in-domain performance does NOT predict cross-speaker generalization.

### 2. The Speaker-Emotion Entanglement Problem (Embedding Analysis)
WavLM embeddings encode speaker (95%) and emotion (94%) with nearly equal separability (1.065× vs 1.072×).

### 3. Contrastive Learning Failure Modes (Exp 17B)
Small batch sizes cause representation collapse, making cross-speaker performance worse.

### 4. TTS Data Limitations Hypothesis (Exp 18 - Pending)
**Hypothesis**: TTS-generated data creates artificial speaker-emotion correlations that prevent cross-speaker generalization.
**Test**: Real emotional speech datasets (RAVDESS) should show better generalization.
**Status**: ⏳ Awaiting experiment completion

---

## Session Statistics

- **Duration**: ~1 hour
- **Files Created**: 3 (2 scripts, 1 summary)
- **Lines of Code**: ~350 (processing + extraction scripts)
- **Dataset Selected**: RAVDESS (24 speakers, 1,440 samples)
- **Download Progress**: 8.2 MB / 208.5 MB (4%)
- **Status**: Pipeline ready, awaiting download completion

---

**Next Session Goal**: Complete RAVDESS download, process dataset, train Experiment 18, and evaluate whether real emotional speech data improves cross-speaker generalization.
