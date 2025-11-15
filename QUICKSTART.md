# Quick Start Guide

## Environment Setup Complete ✓

Your Audio-Augmented LLM research environment is now fully configured and ready to use!

### What's Been Set Up

1. **Conda Environment**: `audio_llm` with Python 3.10
2. **Deep Learning Stack**:
   - PyTorch 2.5.1 with CUDA 12.1 support
   - Dual RTX 3090 GPUs detected and ready
   - Transformers 4.57.1
   - PEFT, Accelerate, BitsAndBytes for efficient training

3. **TTS & Audio Libraries**:
   - Coqui TTS (XTTS v2)
   - librosa, soundfile, audioread for audio processing

4. **Project Structure**:
   ```
   audio_augmented_llm/
   ├── src/
   │   ├── tts_pipeline/       # TTS synthesis
   │   ├── emotion_encoder/    # Emotion embedding extraction
   │   ├── student_training/   # Student model training
   │   ├── evaluation/         # Benchmarks and metrics
   │   └── utils/              # Helper functions
   ├── configs/                # Experiment configurations
   ├── data/                   # Datasets
   ├── scripts/                # Training/evaluation scripts
   └── notebooks/              # Jupyter notebooks
   ```

---

## Activate Environment

Every time you start working, activate the conda environment:

```bash
conda activate audio_llm
```

---

## Verify Installation

Run the verification script to ensure everything is working:

```bash
python scripts/verify_environment.py
```

Expected output:
```
✓ PyTorch version: 2.5.1+cu121
✓ CUDA available: True
✓ Number of GPUs: 2
✓ GPU 0: NVIDIA GeForce RTX 3090
✓ GPU 1: NVIDIA GeForce RTX 3090
✓ All checks passed! Environment is ready.
```

---

## Next Steps

### 1. Explore the Quick Start Notebook

```bash
cd audio_augmented_llm/notebooks
jupyter notebook 01_quick_start.ipynb
```

This notebook demonstrates:
- TTS synthesis
- Emotion embedding extraction
- Basic workflow visualization

### 2. Download Datasets

#### Emotion Speech Data

**IEMOCAP** (requires license):
- Visit: https://sail.usc.edu/iemocap/
- Request access and download
- Place in: `audio_augmented_llm/data/iemocap/`

**MELD** (public):
```bash
# Create download script
mkdir -p audio_augmented_llm/data/meld
# Manual download from: https://affective-meld.github.io/
```

#### Dialogue Data (via HuggingFace)

These will be auto-downloaded when you run training scripts:
- `empathetic_dialogues`
- `daily_dialog`

### 3. Train Emotion Encoder (First Experiment)

Once you have IEMOCAP or MELD data:

```bash
python scripts/train_emotion_encoder.py \
    --config audio_augmented_llm/configs/experiment_config.yaml \
    --dataset meld \
    --output_dir ./outputs/emotion_encoder
```

### 4. Generate Teacher Data

Use a teacher LLM to generate responses + TTS audio:

```bash
python scripts/generate_teacher_data.py \
    --config audio_augmented_llm/configs/experiment_config.yaml \
    --num_samples 1000 \
    --output_dir ./audio_augmented_llm/data/train
```

### 5. Train Student Model

Multi-modal distillation with text + emotion embeddings:

```bash
accelerate launch --multi_gpu --num_processes=2 \
    scripts/train_student.py \
    --config audio_augmented_llm/configs/experiment_config.yaml
```

### 6. Evaluate

```bash
python scripts/evaluate.py \
    --config audio_augmented_llm/configs/experiment_config.yaml \
    --checkpoint ./outputs/checkpoints/best_model
```

---

## Key Files to Customize

### Configuration
Edit `audio_augmented_llm/configs/experiment_config.yaml` to:
- Change model names/sizes
- Adjust hyperparameters
- Modify dataset paths
- Configure training settings

### Module Implementations

**TTS Engine**: `audio_augmented_llm/src/tts_pipeline/tts_engine.py`
- Add CosyVoice support
- Customize emotion control

**Emotion Encoder**: `audio_augmented_llm/src/emotion_encoder/emotion_model.py`
- Switch between WavLM/Whisper
- Modify embedding dimensions
- Add custom pooling strategies

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config:
```yaml
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

### TTS Model Download Issues

If XTTS download is slow, manually download:
```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
```

### Missing Dependencies

Reinstall all:
```bash
pip install -r requirements.txt
```

---

## Development Workflow

1. **Prototype** (Nov-Dec 2025):
   - Small-scale experiments (1k samples)
   - Verify pipeline works end-to-end
   - Tune emotion encoder

2. **Scale Up** (Dec 2025-Jan 2026):
   - Full datasets (10k-100k samples)
   - Systematic ablations
   - Multiple student model sizes

3. **Write & Submit** (Late Dec-Early Jan 2026):
   - ACL 2026: Jan 5 deadline
   - ICML 2026: Jan 28 deadline

---

## Useful Commands

### Check GPU Usage
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Real-time monitoring
```

### Monitor Training
```bash
tensorboard --logdir ./outputs/logs
```

### Export Environment
```bash
conda env export > environment_backup.yml
```

---

## Support

- **Documentation**: See `README.md` for full project overview
- **Code Examples**: Check `notebooks/` directory
- **Configuration**: Refer to `configs/experiment_config.yaml`

---

## Research Timeline

| Phase | Timeline | Goals |
|-------|----------|-------|
| Prototype | Nov-Dec 2025 | Verify concept, tune encoder |
| Scale | Dec 2025-Jan 2026 | Full experiments, ablations |
| Writing | Late Dec-Jan 2026 | Paper preparation |
| **Deadline** | **Jan 5/28, 2026** | **ACL/ICML submission** |

---

**Happy researching! 🚀**

For questions or issues, refer to the main `README.md` or check individual module documentation.
