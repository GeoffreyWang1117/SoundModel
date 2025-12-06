#!/bin/bash
#
# Auto-process RAVDESS dataset once download completes
# This script monitors the download and automatically processes the dataset
#

set -e

echo "=== RAVDESS Auto-Processing Script ==="
echo "Waiting for download to complete (target: 208.5 MB)..."

# Monitor download until it reaches target size
TARGET_SIZE=$((208 * 1024 * 1024))  # 208 MB in bytes
ZIP_FILE="audio_augmented_llm/data/ravdess/Audio_Speech_Actors_01-24.zip"

while true; do
    if [ -f "$ZIP_FILE" ]; then
        CURRENT_SIZE=$(stat -f%z "$ZIP_FILE" 2>/dev/null || stat -c%s "$ZIP_FILE" 2>/dev/null || echo "0")
        CURRENT_MB=$((CURRENT_SIZE / 1024 / 1024))
        echo "[$(date '+%H:%M:%S')] Current size: ${CURRENT_MB} MB / 208 MB"

        if [ "$CURRENT_SIZE" -ge "$TARGET_SIZE" ]; then
            echo "Download appears complete! Waiting 30 seconds to ensure completion..."
            sleep 30
            FINAL_SIZE=$(stat -f%z "$ZIP_FILE" 2>/dev/null || stat -c%s "$ZIP_FILE" 2>/dev/null || echo "0")
            if [ "$FINAL_SIZE" -eq "$CURRENT_SIZE" ]; then
                echo "Download confirmed complete: ${FINAL_SIZE} bytes"
                break
            fi
        fi
    else
        echo "[$(date '+%H:%M:%S')] Waiting for download to start..."
    fi

    sleep 60  # Check every minute
done

echo ""
echo "=== Step 1: Extracting ZIP file ==="
cd audio_augmented_llm/data/ravdess
unzip -q Audio_Speech_Actors_01-24.zip
echo "✅ Extraction complete"

cd ../../..

echo ""
echo "=== Step 2: Organizing dataset into train/val/test splits ==="
python scripts/process_ravdess_dataset.py \
    --ravdess_dir ./audio_augmented_llm/data/ravdess/Audio_Speech_Actors_01-24 \
    --output_dir ./audio_augmented_llm/data/ravdess_processed
echo "✅ Dataset organization complete"

echo ""
echo "=== Step 3: Extracting WavLM embeddings ==="
source /home/coder-gw/miniconda3/etc/profile.d/conda.sh
conda activate audio_llm
export CUDA_VISIBLE_DEVICES=1

python scripts/extract_wavlm_ravdess.py \
    --ravdess_processed_dir ./audio_augmented_llm/data/ravdess_processed \
    --device cuda
echo "✅ WavLM embedding extraction complete"

echo ""
echo "=== All RAVDESS processing complete! ==="
echo ""
echo "Next step: Train Experiment 18 with:"
echo "  python scripts/train_exp11.py \\"
echo "    --train_dir ./audio_augmented_llm/data/ravdess_processed/train \\"
echo "    --val_dir ./audio_augmented_llm/data/ravdess_processed/val \\"
echo "    --embedding_type wavlm \\"
echo "    --emotion_dim 256 \\"
echo "    --num_epochs 5 \\"
echo "    --batch_size 8 \\"
echo "    --learning_rate 5e-5 \\"
echo "    --output_dir ./audio_augmented_llm/models/exp18_ravdess"
