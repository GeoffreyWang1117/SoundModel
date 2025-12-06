#!/bin/bash
# Experiment 8: Feature Importance Analysis
# Run ablation studies on feature groups

source /home/coder-gw/miniconda3/etc/profile.d/conda.sh
conda activate audio_llm

DATA_DIR="./audio_augmented_llm/data/train_1000"
OUTPUT_BASE="./audio_augmented_llm/models/exp8"

echo "========================================"
echo "Experiment 8: Feature Ablation Studies"
echo "========================================"
echo ""

# Experiment 8a: Prosodic Only (13D)
echo "🔬 Exp 8a: Prosodic Features Only (13D)"
echo "----------------------------------------"
mkdir -p ${OUTPUT_BASE}_prosodic_only

# Copy prosodic features as acoustic_embeddings.npz for the dataset loader
cp ${DATA_DIR}/prosodic_only.npz ${DATA_DIR}/acoustic_embeddings_backup.npz
cp ${DATA_DIR}/prosodic_only.npz ${DATA_DIR}/acoustic_embeddings.npz

python scripts/train_student_simple.py \
  --data_dir ${DATA_DIR} \
  --embedding_type acoustic \
  --emotion_dim 13 \
  --use_emotion \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --output_dir ${OUTPUT_BASE}_prosodic_only \
  2>&1 | tee outputs/exp8a_prosodic_only_log.txt

echo ""
echo "✅ Exp 8a complete"
echo ""

# Experiment 8b: Spectral Only (30D)
echo "🔬 Exp 8b: Spectral Features Only (30D)"
echo "----------------------------------------"
mkdir -p ${OUTPUT_BASE}_spectral_only

cp ${DATA_DIR}/spectral_only.npz ${DATA_DIR}/acoustic_embeddings.npz

python scripts/train_student_simple.py \
  --data_dir ${DATA_DIR} \
  --embedding_type acoustic \
  --emotion_dim 30 \
  --use_emotion \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --output_dir ${OUTPUT_BASE}_spectral_only \
  2>&1 | tee outputs/exp8b_spectral_only_log.txt

echo ""
echo "✅ Exp 8b complete"
echo ""

# Experiment 8c: No Prosodic (33D)
echo "🔬 Exp 8c: Without Prosodic Features (33D)"
echo "----------------------------------------"
mkdir -p ${OUTPUT_BASE}_no_prosodic

cp ${DATA_DIR}/no_prosodic.npz ${DATA_DIR}/acoustic_embeddings.npz

python scripts/train_student_simple.py \
  --data_dir ${DATA_DIR} \
  --embedding_type acoustic \
  --emotion_dim 33 \
  --use_emotion \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --output_dir ${OUTPUT_BASE}_no_prosodic \
  2>&1 | tee outputs/exp8c_no_prosodic_log.txt

echo ""
echo "✅ Exp 8c complete"
echo ""

# Restore original acoustic features
echo "🔄 Restoring original acoustic features..."
mv ${DATA_DIR}/acoustic_embeddings_backup.npz ${DATA_DIR}/acoustic_embeddings.npz

echo ""
echo "========================================"
echo "✅ All ablation experiments complete!"
echo "========================================"
echo ""
echo "Results summary:"
grep "Best validation loss" outputs/exp8a_prosodic_only_log.txt outputs/exp8b_spectral_only_log.txt outputs/exp8c_no_prosodic_log.txt
echo ""
