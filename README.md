# Audio-Augmented Small LLM for Emotion- and Trend-Aware Reasoning

Research project exploring emotion-augmented knowledge distillation for small language models using synthesized speech.

## 🎯 Project Goal

Train small LLMs (1.5B parameters) that leverage emotion embeddings extracted from TTS-synthesized audio to improve reasoning capabilities, targeting ACL/ICML 2026 submission.

## 🔥 Key Findings

### Breakthrough Result (Experiment 5)
- **27.82% improvement** with emotion embeddings at 1000 samples
- Non-linear scaling effect discovered: text-only models hit performance ceiling, emotion embeddings break through

### Scale Validation
| Samples | Text-Only | Text+Audio | Improvement |
|---------|-----------|------------|-------------|
| 100     | 0.1818    | 0.1585     | 12.82%      |
| 500     | 0.1321    | 0.1212     | 8.25%       |
| 1000    | 0.1330    | 0.0960     | **27.82%** ⭐ |

### Task-Specific Effectiveness (Experiment 6)
- ✅ **Dialogue tasks**: 8-28% improvement (implicit emotion)
- ❌ **Narrative text**: 0.34% improvement (explicit emotion)
- **Theory**: Emotion embeddings effective when emotion is paralinguistic, not when already explicit in text

## 📂 Project Structure

```
.
├── audio_augmented_llm/
│   ├── src/
│   │   ├── tts_pipeline/        # XTTS v2 TTS synthesis
│   │   ├── emotion_encoder/     # WavLM + Acoustic features
│   │   └── student_training/    # Student model with emotion embeddings
│   └── data/                    # Training datasets (not in git)
├── scripts/                     # Experiment scripts
├── outputs/                     # Training logs (not in git)
└── EXPERIMENT_LOG.md           # Detailed experiment records

```

## 🔬 Current Experiments

**Phase 1**: Proof of Concept ✅  
**Phase 2**: Scale Validation ✅  
**Phase 3**: Data Type Analysis ✅  
**Phase 4**: Feature Engineering 🔄 (Acoustic features vs WavLM)  
**Phase 5**: Production Ready ⏳

## 🛠️ Technical Stack

- **Base Model**: Qwen2.5-1.5B (4-bit quantization + LoRA)
- **TTS**: XTTS v2 (Coqui, speaker: Claribel Dervla)
- **Emotion Encoder**: 
  - WavLM-base-plus (256D) - Deep learning approach
  - Acoustic features (46D) - Speech science approach
- **Training**: PyTorch 2.5.1, Transformers 4.45.2

## 📊 Datasets

### Synthetic Dialogues
- 100/500/1000 samples of teacher-student dialogues
- Template-based generation with emotion contexts
- TTS synthesis with emotion embeddings

### Literary Texts (TinyStories)
- 442 story snippets for comparison
- Result: Emotion embeddings ineffective on narrative text

## 🚀 Quick Start

```bash
# Setup environment
conda create -n audio_llm python=3.10
conda activate audio_llm
pip install torch transformers trl peft TTS librosa

# Run acoustic feature extraction test
python audio_augmented_llm/src/emotion_encoder/acoustic_features.py

# Generate synthetic data
python scripts/generate_teacher_data.py --num_samples 100

# Train model
python scripts/train_student_simple.py \
  --data_dir ./audio_augmented_llm/data/train_100 \
  --use_emotion \
  --num_epochs 5
```

## 📝 Experiment Log

See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for detailed experiment records.

## 🎓 Research Contributions

1. **Non-linear scaling discovery**: Emotion embedding advantage increases dramatically with scale
2. **Task-specific theory**: Implicit vs explicit emotion framework
3. **Acoustic feature engineering**: Prosody + spectral + formant features (46D)
4. **Negative results**: Literary text comparison validates theoretical framework

## 📧 Contact

GeoffreyWang1117  
Email: 173976389+GeoffreyWang1117@users.noreply.github.com

## 📄 License

Research project - License TBD

---

*Last updated: 2025-11-15*  
*Branch: alpha (experimental development)*
