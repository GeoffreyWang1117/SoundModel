#!/bin/bash
# Baseline comparison: Text-only vs Text+Audio

set -e  # Exit on error

echo "======================================================================="
echo "Audio-Augmented LLM: Baseline Comparison"
echo "======================================================================="
echo ""
echo "This script will train two models:"
echo "  1. Text-only baseline (no emotion embeddings)"
echo "  2. Text+Audio model (with emotion embeddings)"
echo ""
echo "Each training run takes approximately 30-60 minutes."
echo "======================================================================="
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate audio_llm

# Set data directory
DATA_DIR="./audio_augmented_llm/data/train_100"
BASE_MODEL="Qwen/Qwen2.5-1.5B"
BATCH_SIZE=2
EPOCHS=3
LEARNING_RATE=2e-4

# 1. Train text-only baseline
echo "======================================================================="
echo "Experiment 1: Text-Only Baseline"
echo "======================================================================="
echo ""
echo "Training without emotion embeddings..."
echo ""

python scripts/train_student_simple.py \
    --data_dir "$DATA_DIR" \
    --model_name "$BASE_MODEL" \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_length 512 \
    --output_dir "./outputs/baseline_text_only" \
    --device cuda

echo ""
echo "✓ Text-only baseline training complete!"
echo ""

# 2. Train text+audio model
echo "======================================================================="
echo "Experiment 2: Text+Audio (with Emotion Embeddings)"
echo "======================================================================="
echo ""
echo "Training with emotion embeddings..."
echo ""

python scripts/train_student_simple.py \
    --data_dir "$DATA_DIR" \
    --model_name "$BASE_MODEL" \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_length 512 \
    --use_emotion \
    --integration_mode concat \
    --emotion_dim 256 \
    --output_dir "./outputs/baseline_text_audio" \
    --device cuda

echo ""
echo "✓ Text+audio model training complete!"
echo ""

# Summary
echo "======================================================================="
echo "✅ Baseline Comparison Complete!"
echo "======================================================================="
echo ""
echo "Results saved to:"
echo "  - Text-only:  ./outputs/baseline_text_only/"
echo "  - Text+audio: ./outputs/baseline_text_audio/"
echo ""
echo "Next steps:"
echo "  1. Compare validation losses from training logs"
echo "  2. Run generation quality evaluation"
echo "  3. Analyze emotion-aware capabilities"
echo ""
echo "======================================================================="
