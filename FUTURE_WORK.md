# Future Work - Audio-Augmented LLM Research

**Created**: 2025-11-16
**Purpose**: Document all planned experiments and improvements before paper writing

---

## Priority 1: Immediate Follow-up Experiments

### Experiment 12: Balanced Multi-Speaker Training

**Motivation**: Exp 11 showed 2 speakers are insufficient (only 16.4% improvement)

**Setup**:
```
Training Data:
  - 4 speakers × 200 samples = 800 total
  - Speakers: Claribel Dervla, Damien Black, Andrew Chipper, Gracie Wise
  - Balanced distribution (no 10:1 imbalance like Exp 11)

Test Data:
  - 1 held-out speaker (e.g., Viktor Eka) × 100 samples
  - Completely unseen during training

Model:
  - Same as Exp 11: Qwen2.5-1.5B + LoRA + 4bit
  - WavLM 256D embeddings
  - Concat integration
```

**Expected Results**:
- Better than Exp 11 (2.5503) due to increased diversity
- Target: < 2.0 cross-speaker loss
- Still may not reach in-domain performance (0.0792)

**Time Estimate**:
- Data generation: 4-5 hours (800 samples TTS)
- Training: 30-40 minutes
- Total: ~6 hours

**Scripts Needed**:
- `scripts/generate_4speaker_dataset.py`
- `scripts/train_exp12.py` (reuse Exp 11 script)

---

### Experiment 13: Speaker Embedding with Adversarial Training

**Motivation**: Explicitly disentangle speaker identity from emotion

**Approach 1: Speaker-Conditioned Normalization**
```python
class SpeakerConditionedProjection(nn.Module):
    def __init__(self, emotion_dim, speaker_dim, hidden_dim):
        self.emotion_proj = nn.Linear(emotion_dim, hidden_dim)
        self.speaker_proj = nn.Linear(speaker_dim, hidden_dim)
        # Use speaker embedding to modulate emotion features
        self.scale = nn.Linear(speaker_dim, hidden_dim)
        self.shift = nn.Linear(speaker_dim, hidden_dim)

    def forward(self, emotion_emb, speaker_id):
        speaker_emb = self.speaker_proj(speaker_id)
        emotion_feat = self.emotion_proj(emotion_emb)

        # Conditional normalization
        scale = self.scale(speaker_emb)
        shift = self.shift(speaker_emb)
        normalized = emotion_feat * scale + shift

        return normalized
```

**Approach 2: Gradient Reversal Layer**
```python
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class SpeakerInvariantModel(nn.Module):
    def __init__(self):
        self.emotion_encoder = EmotionEncoder()
        self.emotion_projection = nn.Linear(256, hidden_dim)
        self.speaker_classifier = nn.Linear(hidden_dim, num_speakers)

    def forward(self, audio, speaker_id, lambda_grl=1.0):
        # Emotion features
        emotion_feat = self.emotion_encoder(audio)
        emotion_proj = self.emotion_projection(emotion_feat)

        # Adversarial speaker classification
        reversed_feat = GradientReversalLayer.apply(emotion_proj, lambda_grl)
        speaker_pred = self.speaker_classifier(reversed_feat)

        return emotion_proj, speaker_pred
```

**Training**:
- Minimize LLM loss (emotion task)
- Maximize speaker classification loss (make features speaker-invariant)
- Balance with lambda schedule

**Expected Results**:
- Should significantly reduce speaker-dependent information
- Target: < 1.5 cross-speaker loss
- May approach in-domain performance

**Time Estimate**: 2-3 days (architecture changes + training)

---

### Experiment 14: Large-Scale Multi-Speaker

**Motivation**: Scale up to achieve robust speaker-invariance

**Setup**:
```
Training Data:
  - 10 speakers × 200 samples = 2000 total
  - Mix of genders, ages, accents
  - Include XTTS built-in speakers:
    * Female: Claribel Dervla, Gracie Wise, Sofia Hellen
    * Male: Damien Black, Andrew Chipper, Viktor Eka
    * Additional: Sample from XTTS's 58 built-in speakers

Test Data:
  - 2 held-out speakers × 100 samples each
  - Cover different gender/accent combinations

Model:
  - Larger base: Qwen2.5-3B or 7B
  - LoRA rank 16 → 32 (more capacity)
  - May need 8-bit quantization for 7B
```

**Expected Results**:
- Should achieve < 1.0 cross-speaker loss
- Potentially approach in-domain performance
- True speaker-invariant emotion perception

**Time Estimate**: 1-2 weeks
- Data generation: 10-12 hours
- Training: 2-4 hours (larger model)
- Multiple iterations likely needed

---

## Priority 2: Alternative Approaches

### Option 1: Acoustic Features with Speaker Normalization

**Current Issue**: Exp 7 showed acoustic features (46D) beat WavLM (256D) in-domain, but untested cross-speaker

**Experiment**:
1. Extract acoustic features for cross-speaker test
2. Apply speaker normalization:
   - Z-score normalization per speaker
   - Percentile-based normalization
   - Speaker-mean subtraction
3. Test on Damien/Andrew

**Expected**: May outperform WavLM even cross-speaker

**Time**: 1-2 days

---

### Option 2: Hybrid Features

**Idea**: Combine acoustic + WavLM features

```python
acoustic_feat = extract_acoustic_features(audio)  # 46D
wavlm_feat = extract_wavlm_features(audio)        # 256D

# Learnable fusion
fusion_feat = torch.cat([acoustic_feat, wavlm_feat], dim=-1)  # 302D
fused = fusion_layer(fusion_feat)  # -> hidden_dim
```

**Hypothesis**:
- Acoustic captures interpretable patterns
- WavLM captures complex patterns
- Fusion may be best of both worlds

**Tested in Exp 8**: Fusion (302D) achieved val loss 0.0771
**Untested**: Cross-speaker performance

**Time**: 2-3 days

---

### Option 3: Multi-Task Learning

**Approach**: Train on multiple tasks simultaneously

```python
Tasks:
1. Emotion-aware text generation (main task)
2. Speaker identification (auxiliary)
3. Emotion classification (auxiliary)
4. Prosody prediction (auxiliary)
```

**Architecture**:
```python
class MultiTaskModel(nn.Module):
    def __init__(self):
        self.shared_encoder = EmotionEncoder()
        self.text_head = LLMHead()
        self.speaker_head = ClassificationHead(num_speakers)
        self.emotion_head = ClassificationHead(num_emotions)

    def forward(self, audio, text):
        shared_feat = self.shared_encoder(audio)

        text_output = self.text_head(shared_feat, text)
        speaker_logits = self.speaker_head(shared_feat)
        emotion_logits = self.emotion_head(shared_feat)

        return text_output, speaker_logits, emotion_logits
```

**Expected**: Better feature learning through multi-task supervision

**Time**: 1 week

---

## Priority 3: Architectural Improvements

### Improvement 1: Attention-Based Fusion

**Current**: Simple concatenation of emotion embedding with text tokens

**Proposed**: Cross-attention between emotion and text

```python
class EmotionTextCrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_embeddings, emotion_embedding):
        # text_embeddings: [batch, seq_len, hidden_dim]
        # emotion_embedding: [batch, emotion_dim]

        emotion_expanded = emotion_embedding.unsqueeze(1)  # [batch, 1, emotion_dim]

        Q = self.query_proj(text_embeddings)
        K = self.key_proj(emotion_expanded)
        V = self.value_proj(emotion_expanded)

        attention = torch.softmax(Q @ K.transpose(-2, -1) / sqrt(hidden_dim), dim=-1)
        attended = attention @ V

        return text_embeddings + attended  # Residual connection
```

**Expected**: Better integration of emotion into text generation

**Time**: 3-4 days

---

### Improvement 2: Hierarchical Emotion Encoding

**Current**: Single 256D global emotion embedding

**Proposed**: Multi-scale emotion features

```python
class HierarchicalEmotionEncoder(nn.Module):
    def __init__(self):
        self.frame_encoder = FrameLevelEncoder()    # Fine-grained
        self.segment_encoder = SegmentLevelEncoder()  # Mid-level
        self.utterance_encoder = UtteranceLevelEncoder()  # Global

    def forward(self, audio):
        frame_feat = self.frame_encoder(audio)      # [batch, T, dim]
        segment_feat = self.segment_encoder(audio)  # [batch, S, dim]
        utterance_feat = self.utterance_encoder(audio)  # [batch, dim]

        return {
            'frame': frame_feat,
            'segment': segment_feat,
            'utterance': utterance_feat
        }
```

**Use**: Different granularities for different tokens

**Time**: 1 week

---

## Priority 4: Data Improvements

### Dataset 1: Real Human Speech

**Current**: All audio is TTS-generated (XTTS v2)

**Limitation**:
- TTS may not capture authentic emotion
- Prosody may be artificial
- Limited emotional range

**Proposed**:
- Use emotional speech datasets:
  * IEMOCAP (acted emotions)
  * MSP-Podcast (natural conversations)
  * CREMA-D (diverse speakers)
- Generate text from audio using Whisper
- Create teacher responses using GPT-4

**Expected**: More realistic emotion patterns

**Challenge**: Alignment between audio emotion and text content

**Time**: 2-3 weeks (data collection + processing)

---

### Dataset 2: More Emotions

**Current**: 5 basic emotions (joy, anger, sadness, neutral, fear)

**Proposed**: Expand to 10+ emotions
- Basic: happy, sad, angry, fearful, surprised, disgusted
- Complex: proud, ashamed, grateful, frustrated, anxious, excited

**Impact**: Richer emotion modeling

**Time**: 1-2 days (data generation)

---

### Dataset 3: Longer Contexts

**Current**: Short teacher responses (1-3 sentences)

**Proposed**: Multi-turn dialogues
- Emotion evolves across turns
- Model maintains emotion consistency
- More realistic conversation flow

**Time**: 1 week (data generation + training)

---

## Priority 5: Evaluation Improvements

### Metric 1: Emotion-Specific Evaluation

**Current**: Single loss metric

**Proposed**: Per-emotion analysis
```python
for emotion in ['joy', 'anger', 'sadness', 'neutral', 'fear']:
    emotion_samples = filter_by_emotion(test_set, emotion)
    loss = evaluate(model, emotion_samples)
    print(f"{emotion}: {loss:.4f}")
```

**Benefit**: Identify which emotions are harder to model

**Time**: 1 day

---

### Metric 2: Human Evaluation

**Setup**:
1. Generate responses with/without emotion conditioning
2. Human raters judge:
   - Emotional appropriateness (1-5)
   - Response quality (1-5)
   - Naturalness (1-5)

**Sample Size**: 100 examples × 3 raters

**Time**: 1 week (setup + evaluation)

---

### Metric 3: Downstream Task Performance

**Test on real applications**:
1. Emotional chatbot
2. Empathetic counseling
3. Sentiment-aware QA

**Metrics**: Task-specific (user satisfaction, task completion, etc.)

**Time**: 2-4 weeks

---

## Priority 6: Theoretical Analysis

### Analysis 1: Feature Attribution

**Question**: Which acoustic features matter most for which emotions?

**Method**:
- SHAP values
- Integrated gradients
- Attention visualization

**Expected Insights**:
- Pitch → valence
- Energy → arousal
- MFCC → emotion identity

**Time**: 3-4 days

---

### Analysis 2: Speaker Invariance Analysis

**Question**: What makes features speaker-dependent?

**Method**:
- t-SNE visualization of embeddings colored by speaker
- CKA (Centered Kernel Alignment) between speaker representations
- Linear probing: can you predict speaker from emotion embeddings?

**Expected**: Quantify speaker leakage

**Time**: 2-3 days

---

### Analysis 3: Ablation Studies

**Already done** (Exp 8):
- ✓ Prosodic vs Spectral features
- ✓ Feature group importance

**Additional ablations**:
- LoRA rank (4, 8, 16, 32, 64)
- Model size (1.5B, 3B, 7B)
- Training data size (100, 500, 1000, 2000)
- Emotion embedding dimension (64, 128, 256, 512)

**Time**: 1-2 weeks (many training runs)

---

## Priority 7: Deployment Considerations

### Optimization 1: Model Compression

**Current**: 4-bit quantization + LoRA

**Additional**:
- Pruning (remove 30-50% of weights)
- Distillation (student 0.5B from teacher 1.5B)
- Mixed precision training

**Target**: < 500MB model size, < 100ms latency

**Time**: 1-2 weeks

---

### Optimization 2: Real-Time Emotion Extraction

**Current**: Offline WavLM extraction

**Proposed**: Streaming emotion encoder
- Process audio in chunks (100ms)
- Update emotion embedding incrementally
- Enable real-time applications

**Time**: 1-2 weeks

---

## Priority 8: Novel Research Directions

### Direction 1: Emotion Disentanglement

**Idea**: Separate emotion into dimensions
- Valence (positive/negative)
- Arousal (calm/excited)
- Dominance (submissive/dominant)

**Architecture**:
```python
emotion_feat = encoder(audio)
valence = valence_head(emotion_feat)
arousal = arousal_head(emotion_feat)
dominance = dominance_head(emotion_feat)

# Combine for text generation
emotion_vector = torch.cat([valence, arousal, dominance], dim=-1)
```

**Benefit**: More interpretable and controllable

**Time**: 2 weeks

---

### Direction 2: Cross-Lingual Emotion Transfer

**Question**: Do emotions transfer across languages?

**Experiment**:
- Train on English audio
- Test on Chinese/Spanish/French audio
- Use multilingual LLM (e.g., mBERT, XLM-R base)

**Expected**: Some transfer if prosody is universal

**Time**: 3-4 weeks

---

### Direction 3: Emotion-Conditioned Generation

**Reverse direction**: Generate audio with specified emotion

**Pipeline**:
1. Text → LLM response
2. Emotion label → TTS with emotion control
3. Controllable emotional speech synthesis

**Application**: Voice assistants with emotional range

**Time**: 2-3 weeks

---

## Not Planned (Out of Scope)

### ❌ Video-Based Emotion
- Requires facial expression analysis
- Different modality
- Not aligned with current audio focus

### ❌ Multimodal Fusion (Audio + Text + Vision)
- Too complex for current stage
- Better as separate follow-up work

### ❌ Online Learning / Continual Learning
- Interesting but orthogonal to core contribution
- Future research direction

---

## Summary: Recommended Roadmap

### Phase 1: Complete Current Investigation (Before Paper)
- ✅ Exp 1-11 complete
- 📝 Write paper with current results
- 🔖 Mark Exp 12-14 as "future work"

### Phase 2: Immediate Follow-up (After Paper Submission)
- Exp 12: 4-speaker balanced (1 week)
- Exp 13: Speaker embedding + adversarial (2 weeks)
- Exp 14: Large-scale 10+ speakers (2 weeks)

### Phase 3: Extended Work (Next Paper)
- Real human speech data
- Alternative architectures
- Downstream task evaluation
- Human evaluation study

### Phase 4: Production (If successful)
- Model compression
- Real-time optimization
- API deployment

---

## Paper Writing Checklist

### Must Include:
- ✅ Motivation: Why emotion matters for LLM
- ✅ Method: Pipeline + acoustic features
- ✅ Exp 7-8: Feature comparison + ablation
- ✅ Exp 9-11: Cross-speaker challenge + partial solution
- ✅ Honest discussion of limitations

### Can Defer to Future Work:
- Exp 12-14 (larger multi-speaker)
- Speaker disentanglement methods
- Real human speech
- Human evaluation
- Downstream tasks

### Must Acknowledge:
- Cross-speaker generalization partially solved (16.4% improvement)
- Not production-ready (32× worse than in-domain)
- Need for larger-scale multi-speaker training
- Validation setup matters (same-speaker ≠ cross-speaker)

---

**End of Future Work Documentation**

**Next Action**: Begin LaTeX paper writing with clear scope based on Exp 1-11 results.
