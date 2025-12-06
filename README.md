# Audio-Augmented Emotion-Aware Language Models

Research on integrating emotional prosody from speech into language models for emotion-aware reasoning and natural language understanding.

## Overview

This project explores methods for training language models that can perceive and respond to emotional context from speech. By integrating audio features (both handcrafted acoustic features and learned representations) into LLMs, we investigate how emotional prosody affects model performance and cross-speaker generalization.

## Key Findings

### 1. Data Quality Matters More Than Architecture
Training on real emotional speech (RAVDESS dataset, 24 speakers) achieved **48% reduction in cross-speaker degradation** compared to TTS-generated data:

| Metric | Synthetic Speech | Real Speech | Improvement |
|--------|------------------|-------------|-------------|
| Training Speakers | 4 | 16 | +300% |
| Cross-Speaker Degradation Factor | 80.2× | **41.7×** | **-48%** ✅ |

**Key Insight**: Synthetic speech creates artificial speaker-emotion correlations that models overfit to. Real emotional speech with natural prosodic variation is essential for robust cross-speaker generalization.

### 2. The Feature-Generalization Paradox
Better in-domain performance does not predict cross-speaker generalization:

| Feature Type | Dimensions | In-Domain Val Loss | Cross-Speaker Degradation |
|--------------|-----------|-------------------|---------------------------|
| Handcrafted Acoustic | 46D | 0.0064 (best) | 124× (worst) |
| WavLM Learned | 256D | 0.0139 | 80× (better) |

**Key Insight**: Validation loss on seen speakers is misleading. True robustness requires evaluation on completely unseen speakers with proper cross-speaker splits.

### 3. Speaker-Emotion Entanglement
Audio representations encode speaker identity and emotion with nearly identical separability, making it difficult for models to distinguish emotional state from speaker identity.

## Technical Approach

- **Base Model**: Qwen2.5-1.5B-Instruct with 4-bit quantization and LoRA fine-tuning
- **Audio Features**:
  - Handcrafted acoustic features (46D): pitch, energy, MFCCs, formants
  - WavLM embeddings (256D): learned self-supervised representations
- **Training Strategy**: Cross-speaker evaluation with strict speaker splits
- **Datasets**: TTS-generated data, RAVDESS (real emotional speech)

## Research Contributions

1. **TTS Data Limitation Problem**: Demonstrated that synthetic speech fundamentally limits cross-speaker generalization regardless of speaker count
2. **Feature-Generalization Paradox**: Identified that in-domain performance inversely correlates with cross-speaker robustness
3. **Data Quality Priority**: Showed that training on real emotional speech is more effective than complex architectural solutions
4. **Evaluation Standards**: Established need for strict cross-speaker evaluation protocols

## Project Structure

```
.
├── audio_augmented_llm/
│   ├── src/                    # Source code
│   ├── data/                   # Training and test datasets
│   └── models/                 # Trained model checkpoints
├── scripts/                    # Data generation, feature extraction, training scripts
├── paper/                      # Research paper LaTeX sources
├── outputs/                    # Training logs and analysis results
└── README.md                   # This file
```

## Quick Start

```bash
# Setup environment
conda create -n audio_llm python=3.10
conda activate audio_llm
pip install torch transformers peft TTS librosa scipy

# Extract acoustic features
python scripts/extract_acoustic_features_batch.py \
  --data_dir ./audio_augmented_llm/data/train

# Train model
python scripts/train_student_simple.py \
  --data_dir ./audio_augmented_llm/data/train \
  --embedding_type acoustic \
  --num_epochs 5

# Evaluate on cross-speaker test set
python scripts/evaluate_model.py \
  --model_dir ./audio_augmented_llm/models/best_model \
  --data_dir ./audio_augmented_llm/data/test
```

## Requirements

- Python 3.10+
- PyTorch 2.5+
- Transformers 4.45+
- PEFT 0.13+
- librosa 0.10+
- scipy

## License

Research project - License TBD

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{emotion-aware-llm-2025,
  title={Audio-Augmented Emotion-Aware Language Models},
  author={},
  year={2025},
  note={Research project on emotion-aware language modeling}
}
```

---

**Note**: This is an active research project. Results and methodologies are subject to change as research progresses.
