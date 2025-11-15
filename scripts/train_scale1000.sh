#!/bin/bash
# Train both models on 1000-sample dataset

# Activate environment
source /home/coder-gw/miniconda3/etc/profile.d/conda.sh
conda activate audio_llm

echo "========================================================================"
echo "Training on 1000-sample dataset"
echo "========================================================================"
echo ""

# Train text-only baseline
echo "Step 1/2: Training text-only baseline..."
python scripts/train_student_simple.py \
    --data_dir ./audio_augmented_llm/data/train_1000 \
    --output_dir ./outputs/scale1000_text_only \
    --model_name Qwen/Qwen2.5-1.5B \
    --use_emotion false \
    --batch_size 2 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    > ./outputs/scale1000_text_only_log.txt 2>&1

echo "✓ Text-only training completed"
echo ""

# Train text+audio model
echo "Step 2/2: Training text+audio model..."
python scripts/train_student_simple.py \
    --data_dir ./audio_augmented_llm/data/train_1000 \
    --output_dir ./outputs/scale1000_text_audio \
    --model_name Qwen/Qwen2.5-1.5B \
    --use_emotion true \
    --batch_size 2 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    > ./outputs/scale1000_text_audio_log.txt 2>&1

echo "✓ Text+audio training completed"
echo ""

echo "========================================================================"
echo "Training complete! Check outputs/scale1000_*_log.txt for details."
echo "========================================================================"
