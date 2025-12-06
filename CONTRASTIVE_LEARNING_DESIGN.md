# Experiment 17: Contrastive Learning for Speaker-Invariant Emotion Representations

## Problem Statement

**Current Issue**: WavLM embeddings encode both speaker identity (95% classification accuracy) and emotion information (94% accuracy) with nearly equal separability (1.065× vs 1.072×). This entanglement causes catastrophic cross-speaker generalization failure (93.75% → 21% on unseen speaker).

**Goal**: Design a training approach that produces **speaker-invariant emotion representations** while maintaining high emotion classification accuracy.

---

## Approach: Supervised Contrastive Learning

### Core Idea

Use contrastive learning to explicitly force the emotion projection layer to learn representations where:
1. **Positive pairs**: Same emotion, different speakers → embeddings should be similar
2. **Negative pairs**: Different emotions (any speakers) → embeddings should be dissimilar

This directly addresses the entanglement by making speaker differences irrelevant while emphasizing emotion differences.

### Architecture

```
Audio → WavLM Encoder (frozen) → Emotion Projection → L2 Normalize → Contrastive Loss
                                                    ↓
                                               LLM Input
```

**Key Components**:
1. **WavLM Encoder**: Frozen (we accept its speaker information but don't train it)
2. **Emotion Projection**: Trainable layer that transforms WavLM embeddings to speaker-invariant space
3. **L2 Normalization**: Projects to unit hypersphere for contrastive learning
4. **Dual Loss**: Contrastive loss + Language modeling loss

---

## Loss Function Design

### 1. Supervised Contrastive Loss (Primary)

For a batch of samples with emotion labels and speaker IDs:

```python
For anchor sample i with emotion e_i and speaker s_i:
  Positive set P(i) = {j | emotion_j == e_i AND speaker_j != s_i}
  Negative set N(i) = {j | emotion_j != e_i}

  Loss_i = -log [ sum_{p in P(i)} exp(sim(z_i, z_p) / τ) /
                  sum_{j in P(i) ∪ N(i)} exp(sim(z_i, z_j) / τ) ]
```

**Key Properties**:
- Forces same-emotion-different-speaker pairs to be close
- Forces different-emotion pairs (any speaker) to be far
- Temperature τ controls the concentration (typical: 0.07 - 0.1)

**Critical Detail**: Positive set MUST exclude same speaker (speaker_j != s_i). This ensures the model learns speaker-invariant patterns.

### 2. Language Modeling Loss (Secondary)

Standard next-token prediction loss:

```python
Loss_LM = CrossEntropy(model_output, target_tokens)
```

### 3. Combined Loss

```python
Total_Loss = λ_contrastive * Loss_contrastive + λ_lm * Loss_LM
```

**Hyperparameters**:
- λ_contrastive: Weight for contrastive loss (start: 0.1 - 0.5)
- λ_lm: Weight for language modeling loss (start: 1.0)
- τ (temperature): Controls contrastive concentration (start: 0.07)

---

## Implementation Details

### Training Strategy

**Two-Stage Training**:

**Stage 1: Contrastive Pre-training (Optional)**
- Train only the emotion projection layer with contrastive loss
- Freeze LLM (LoRA)
- Goal: Learn good speaker-invariant emotion space
- Epochs: 3-5

**Stage 2: Joint Training**
- Train both emotion projection and LLM (LoRA)
- Use combined loss
- Goal: Optimize for language modeling while maintaining speaker invariance
- Epochs: 5

**Alternative: Direct Joint Training**
- Skip Stage 1, train everything together from start
- Simpler but may be less stable
- Use if Stage 1 doesn't show clear benefit

### Batch Construction

**Critical Requirement**: Each batch must contain multiple speakers for each emotion.

**Strategy**:
- Batch size: 16-32
- Ensure at least 2 speakers per emotion per batch
- Use stratified sampling: sample speakers uniformly, then sample emotions

**Example Batch (N=16, 4 speakers, 4 emotions)**:
```
Speaker A: anger, joy, sadness, neutral
Speaker B: anger, joy, sadness, neutral
Speaker C: anger, joy, sadness, neutral
Speaker D: anger, joy, sadness, neutral
```

This ensures every emotion has cross-speaker positive pairs.

### Evaluation Metrics

1. **Cross-Speaker Test Loss**: Primary metric (Viktor test set)
2. **Emotion Classification Accuracy**: On Viktor embeddings
3. **Speaker Classification Accuracy**: Should DECREASE on training embeddings
4. **Emotion/Speaker Separation Ratios**: Track disentanglement progress

**Success Criteria**:
- Viktor test emotion accuracy > 50% (vs current 21%)
- Training speaker classification < 70% (vs current 95%)
- Emotion classification > 85% (maintain high emotion separability)

---

## Experiment Configuration

### Experiment 17A: Contrastive Pre-training Only

**Goal**: Test if contrastive loss alone can learn speaker-invariant space

**Config**:
- Stage 1 only (3 epochs)
- λ_contrastive = 1.0, λ_lm = 0.0
- Freeze LoRA, train only emotion projection
- Temperature τ = 0.07

**Expected Outcome**:
- Emotion projection learns speaker-invariant space
- Need Stage 2 to adapt LLM to this space

### Experiment 17B: Joint Training (Recommended)

**Goal**: Train everything together with balanced losses

**Config**:
- Direct joint training (5 epochs)
- λ_contrastive = 0.3, λ_lm = 1.0
- Train both emotion projection and LoRA
- Temperature τ = 0.07
- Batch size = 16 (4 samples per speaker)

**Expected Outcome**:
- Balanced learning of speaker-invariance and language modeling
- Better overall performance

### Experiment 17C: Two-Stage Training

**Goal**: Best of both worlds

**Config**:
- Stage 1: 3 epochs, λ_contrastive = 1.0, λ_lm = 0.0
- Stage 2: 5 epochs, λ_contrastive = 0.3, λ_lm = 1.0
- Temperature τ = 0.07

**Expected Outcome**:
- Stage 1 establishes good emotion space
- Stage 2 fine-tunes for language generation

---

## Implementation Plan

### 1. Contrastive Loss Module

```python
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, emotion_labels, speaker_ids):
        """
        embeddings: (batch_size, embedding_dim) - L2 normalized
        emotion_labels: (batch_size,) - emotion class indices
        speaker_ids: (batch_size,) - speaker class indices
        """
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create masks for positive and negative pairs
        emotion_mask = emotion_labels.unsqueeze(0) == emotion_labels.unsqueeze(1)
        speaker_mask = speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)

        # Positive: same emotion, different speaker
        positive_mask = emotion_mask & ~speaker_mask

        # Negative: different emotion (any speaker)
        negative_mask = ~emotion_mask

        # Compute contrastive loss
        # ... (detailed implementation)
```

### 2. Modified Student Model

Extend existing StudentModel to:
- Support L2 normalization of emotion embeddings
- Compute contrastive loss during training
- Accept emotion_labels and speaker_ids in forward pass

### 3. Modified Training Loop

```python
for batch in dataloader:
    # Batch must contain: input_ids, emotion_embeddings, emotion_labels, speaker_ids

    # Forward pass
    outputs = model(
        input_ids=batch['input_ids'],
        emotion_embedding=batch['emotion_embeddings'],
        labels=batch['labels']
    )

    # Get normalized emotion representations
    emotion_repr = model.get_normalized_emotion_embedding(batch['emotion_embeddings'])

    # Compute losses
    loss_lm = outputs.loss
    loss_contrastive = contrastive_criterion(
        emotion_repr,
        batch['emotion_labels'],
        batch['speaker_ids']
    )

    # Combined loss
    total_loss = lambda_lm * loss_lm + lambda_contrastive * loss_contrastive
```

### 4. Dataset Modifications

Extend dataset to return:
- `emotion_label`: Emotion class index (0-5 for 6 emotions)
- `speaker_id`: Speaker class index (0-3 for 4 speakers)

### 5. Evaluation

After training, run:
- Standard model evaluation on Viktor test set
- Embedding analysis (re-run analyze_wavlm_embeddings.py on new model)
- Compare speaker/emotion classification accuracies before vs after

---

## Hyperparameter Search Strategy

### Initial Configuration (Exp 17B)

```json
{
  "lambda_contrastive": 0.3,
  "lambda_lm": 1.0,
  "temperature": 0.07,
  "batch_size": 16,
  "num_epochs": 5
}
```

### If Cross-Speaker Performance Still Poor

**Increase contrastive weight**:
- λ_contrastive: 0.3 → 0.5 → 0.7
- May sacrifice some LM performance for better speaker invariance

### If Language Modeling Degrades

**Decrease contrastive weight**:
- λ_contrastive: 0.3 → 0.2 → 0.1
- Or use two-stage training (Exp 17C)

### If Contrastive Loss Doesn't Converge

**Adjust temperature**:
- τ: 0.07 → 0.1 (softer)
- Or τ: 0.07 → 0.05 (harder)

---

## Expected Results

### Optimistic Scenario

- Viktor test loss: 2.5336 → **1.5** (41% improvement)
- Viktor emotion accuracy: 21% → **60%+** (3× improvement)
- Training speaker accuracy: 95% → **50-60%** (speaker info reduced)
- Training emotion accuracy: 94% → **85-90%** (minor decrease acceptable)

### Realistic Scenario

- Viktor test loss: 2.5336 → **2.0** (20% improvement)
- Viktor emotion accuracy: 21% → **40-50%** (2× improvement)
- Training speaker accuracy: 95% → **70-80%**
- Training emotion accuracy: 94% → **90%**

### Pessimistic Scenario

- Viktor test loss: No significant improvement
- Speaker information reduction happens but doesn't help cross-speaker
- **Conclusion**: Problem may require real emotional speech data (Task 4)

---

## Alternative: Hard Negative Mining

If standard contrastive learning doesn't work well, try **hard negative mining**:

### Modified Negative Selection

Instead of all different-emotion samples, select:
- **Hard negatives**: Different emotion but similar acoustic properties
- Example: anger vs. fear (both high arousal) are harder than anger vs. neutral

```python
# Compute acoustic similarity (e.g., F0 range, energy)
hard_negative_mask = (different_emotion) & (acoustic_similarity > threshold)
```

This forces the model to learn fine-grained emotion distinctions that aren't based on speaker characteristics.

---

## Success Indicators During Training

Monitor these metrics to assess if contrastive learning is working:

1. **Contrastive Loss Decreasing**: Should steadily decrease
2. **Within-Emotion-Cross-Speaker Distance**: Should decrease
3. **Between-Emotion Distance**: Should increase
4. **Validation Loss**: Should not increase significantly

**Red Flags**:
- Contrastive loss stuck or increasing → adjust temperature or λ
- LM loss exploding → reduce λ_contrastive
- Both losses not converging → check batch composition

---

## Conclusion

This contrastive learning approach directly targets the root cause identified in Task 2: speaker-emotion entanglement in WavLM embeddings. By explicitly training for speaker-invariant emotion representations, we expect significant improvement in cross-speaker generalization.

**Next Steps**:
1. Implement SupervisedContrastiveLoss
2. Modify StudentModel for contrastive learning
3. Update dataset to include emotion_label and speaker_id
4. Create training script for Exp 17B (joint training)
5. Train and evaluate
6. Re-run embedding analysis to confirm disentanglement
