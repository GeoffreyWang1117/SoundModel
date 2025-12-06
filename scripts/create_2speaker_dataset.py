#!/usr/bin/env python3
"""
Create 2-speaker dataset from existing data for Experiment 11.
Uses Claribel + Damien data to enable multi-speaker training.
"""

import json
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

def create_2speaker_dataset():
    """
    Create 2-speaker dataset:
    - Training: Claribel (500) + Damien (50) = 550 samples
    - Validation: Damien (50) = 50 samples
    """

    print("=" * 80)
    print("Creating 2-Speaker Dataset for Experiment 11")
    print("=" * 80)

    random.seed(42)

    # Create output directories
    train_dir = Path("audio_augmented_llm/data/exp11_2speaker_train")
    val_dir = Path("audio_augmented_llm/data/exp11_2speaker_val")

    for dir in [train_dir, val_dir]:
        dir.mkdir(parents=True, exist_ok=True)
        (dir / "audio").mkdir(exist_ok=True)

    # === Load Claribel data ===
    print("\n1. Loading Claribel Dervla data...")
    claribel_dir = Path("audio_augmented_llm/data/train_1000")

    with open(claribel_dir / "metadata.json", 'r') as f:
        claribel_metadata = json.load(f)

    claribel_embeddings = np.load(claribel_dir / "emotion_embeddings.npz")

    # Sample 500 Claribel samples for training
    claribel_indices = random.sample(range(len(claribel_metadata)), 500)

    train_samples = []
    train_embeddings = {}

    print("   Copying 500 Claribel samples to training set...")
    for idx in tqdm(claribel_indices):
        sample = claribel_metadata[idx]
        old_id = sample['id']
        new_id = f"claribel_{old_id}"

        # Copy audio
        old_audio = Path(sample['audio_path'])
        new_audio = train_dir / "audio" / f"{new_id}.wav"
        shutil.copy(old_audio, new_audio)

        # Add sample
        train_samples.append({
            "id": new_id,
            "context": sample['context'],
            "teacher_response": sample['teacher_response'],
            "emotion": sample['emotion'],
            "dataset": "exp11_multispeaker",
            "speaker": "Claribel Dervla",
            "speaker_id": 0,  # Add speaker ID
            "audio_path": str(new_audio),
            "has_embedding": True
        })

        # Copy embedding
        if old_id in claribel_embeddings:
            train_embeddings[new_id] = claribel_embeddings[old_id]

    print(f"   ✓ Added {len(claribel_indices)} Claribel samples to training")

    # === Load Damien data ===
    print("\n2. Loading Damien Black data...")
    damien_dir = Path("audio_augmented_llm/data/test_cross_speaker_damien")

    with open(damien_dir / "metadata.json", 'r') as f:
        damien_metadata = json.load(f)

    # Damien doesn't have wavlm embeddings yet, will extract later
    damien_embeddings = {}

    # Shuffle and split Damien data
    random.shuffle(damien_metadata)
    damien_train = damien_metadata[:50]  # First 50 for training
    damien_val = damien_metadata[50:]    # Remaining 50 for validation

    # Add Damien training samples
    print("   Copying 50 Damien samples to training set...")
    for sample in tqdm(damien_train):
        old_id = sample['id']
        new_id = f"damien_{old_id}"

        # Copy audio
        old_audio = Path(sample['audio_path'])
        new_audio = train_dir / "audio" / f"{new_id}.wav"
        shutil.copy(old_audio, new_audio)

        # Add sample (Damien data uses 'text' field instead of context/response split)
        train_samples.append({
            "id": new_id,
            "context": "",  # Damien data doesn't separate context
            "teacher_response": sample.get('text', ''),  # Use text field
            "emotion": sample['emotion'],
            "dataset": "exp11_multispeaker",
            "speaker": "Damien Black",
            "speaker_id": 1,  # Different speaker ID
            "audio_path": str(new_audio),
            "has_embedding": True
        })

        # Copy embedding
        if old_id in damien_embeddings:
            train_embeddings[new_id] = damien_embeddings[old_id]

    print(f"   ✓ Added {len(damien_train)} Damien samples to training")

    # Add Damien validation samples
    val_samples = []
    val_embeddings = {}

    print("   Copying 50 Damien samples to validation set...")
    for sample in tqdm(damien_val):
        old_id = sample['id']
        new_id = f"damien_{old_id}"

        # Copy audio
        old_audio = Path(sample['audio_path'])
        new_audio = val_dir / "audio" / f"{new_id}.wav"
        shutil.copy(old_audio, new_audio)

        # Add sample (Damien data uses 'text' field)
        val_samples.append({
            "id": new_id,
            "context": "",  # Damien data doesn't separate context
            "teacher_response": sample.get('text', ''),  # Use text field
            "emotion": sample['emotion'],
            "dataset": "exp11_multispeaker",
            "speaker": "Damien Black",
            "speaker_id": 1,
            "audio_path": str(new_audio),
            "has_embedding": True
        })

        # Copy embedding
        if old_id in damien_embeddings:
            val_embeddings[new_id] = damien_embeddings[old_id]

    print(f"   ✓ Added {len(damien_val)} Damien samples to validation")

    # === Save training data ===
    print("\n3. Saving training data...")
    with open(train_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    np.savez_compressed(train_dir / "emotion_embeddings.npz", **train_embeddings)
    print(f"   ✓ Training: {len(train_samples)} samples, {len(train_embeddings)} embeddings")

    # === Save validation data ===
    print("\n4. Saving validation data...")
    with open(val_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)
    np.savez_compressed(val_dir / "emotion_embeddings.npz", **val_embeddings)
    print(f"   ✓ Validation: {len(val_samples)} samples, {len(val_embeddings)} embeddings")

    # === Summary ===
    print("\n" + "=" * 80)
    print("✅ 2-Speaker Dataset Created Successfully!")
    print("=" * 80)
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training set: {len(train_samples)} samples")
    print(f"     - Claribel Dervla (speaker_id=0): 500 samples")
    print(f"     - Damien Black (speaker_id=1): 50 samples")
    print(f"   Validation set: {len(val_samples)} samples")
    print(f"     - Damien Black (speaker_id=1): 50 samples")
    print(f"\n📁 Output directories:")
    print(f"   Training: {train_dir}")
    print(f"   Validation: {val_dir}")
    print(f"\n🎯 Experiment 11 Goal:")
    print(f"   Train on 2 speakers to learn speaker-invariant emotion patterns")
    print(f"   Validate generalization on same speakers (different samples)")


if __name__ == "__main__":
    create_2speaker_dataset()
