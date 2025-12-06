#!/usr/bin/env python3
"""
Create multi-speaker dataset by combining existing data and generating new speakers.
This is more efficient than regenerating everything from scratch.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_multispeaker_dataset():
    """
    Create multi-speaker dataset from existing data.

    Strategy:
    - Use existing train_1000 (Claribel, 1000 samples) -> sample 250
    - Use existing test_cross_speaker_damien (Damien, 100 samples) -> keep as held-out test
    - Generate Andrew Chipper (250 samples) - NEW
    - Generate Gracie Wise (250 samples) - NEW

    Training: Claribel (250) + Andrew (250) + Gracie (250) = 750 samples
    Testing: Damien (100 samples) - held-out speaker
    """

    print("=" * 80)
    print("Creating Multi-Speaker Dataset from Existing Data")
    print("=" * 80)

    # Create output directory
    output_dir = Path("audio_augmented_llm/data/multispeaker_train")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Audio directory
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    all_samples = []
    all_embeddings = {}

    # 1. Sample from existing Claribel data
    print("\n1. Sampling Claribel Dervla data (250 from 1000)...")
    claribel_dir = Path("audio_augmented_llm/data/train_1000")

    with open(claribel_dir / "metadata.json", 'r') as f:
        claribel_metadata = json.load(f)

    # Load embeddings
    claribel_embeddings = np.load(claribel_dir / "emotion_embeddings.npz")

    # Sample 250 items
    import random
    random.seed(42)  # For reproducibility
    sampled_indices = random.sample(range(len(claribel_metadata)), 250)

    for idx in tqdm(sampled_indices, desc="Copying Claribel"):
        sample = claribel_metadata[idx]
        old_id = sample['id']
        new_id = f"claribel_dervla_{old_id}"

        # Copy audio file
        old_audio_path = Path(sample['audio_path'])
        new_audio_path = audio_dir / f"{new_id}.wav"
        shutil.copy(old_audio_path, new_audio_path)

        # Update sample
        new_sample = {
            "id": new_id,
            "context": sample['context'],
            "teacher_response": sample['teacher_response'],
            "emotion": sample['emotion'],
            "dataset": "multispeaker",
            "speaker": "Claribel Dervla",
            "audio_path": str(new_audio_path),
            "has_embedding": True
        }
        all_samples.append(new_sample)

        # Copy embedding
        if old_id in claribel_embeddings:
            all_embeddings[new_id] = claribel_embeddings[old_id]

    print(f"✓ Sampled {len(sampled_indices)} Claribel samples")

    # 2. Keep Damien as held-out test set (don't include in training)
    print("\n2. Damien Black will be used as held-out test set (100 samples)")
    print("   Location: audio_augmented_llm/data/test_cross_speaker_damien")

    # 3. & 4. Generate Andrew Chipper and Gracie Wise (250 each)
    print("\n3. Andrew Chipper and Gracie Wise need to be generated separately")
    print("   This will be done in the next step due to TTS generation time")
    print("   For now, we'll create the structure with Claribel data only")

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved metadata: {len(all_samples)} samples")

    # Save embeddings
    embeddings_path = output_dir / "emotion_embeddings.npz"
    np.savez_compressed(embeddings_path, **all_embeddings)
    print(f"✓ Saved embeddings: {len(all_embeddings)} embeddings")

    print("\n" + "=" * 80)
    print("✅ Phase 1 Complete: Claribel data prepared")
    print("=" * 80)
    print(f"Current training samples: {len(all_samples)}")
    print(f"Output directory: {output_dir}")
    print("\nNext steps:")
    print("  1. Generate Andrew Chipper samples (250)")
    print("  2. Generate Gracie Wise samples (250)")
    print("  3. Merge all speakers into final training set")

    return output_dir


if __name__ == "__main__":
    create_multispeaker_dataset()
