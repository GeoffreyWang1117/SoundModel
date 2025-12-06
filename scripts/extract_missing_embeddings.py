#!/usr/bin/env python3
"""
Extract WavLM embeddings for samples that don't have them yet.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from audio_augmented_llm.src.emotion_encoder.emotion_model import EmotionEncoder


def extract_missing_embeddings(data_dir: str):
    """Extract embeddings for samples that don't have them."""

    data_path = Path(data_dir)

    print(f"Processing: {data_dir}")

    # Load metadata
    with open(data_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    # Load existing embeddings
    embeddings_path = data_path / "emotion_embeddings.npz"
    if embeddings_path.exists():
        existing_embeddings = dict(np.load(embeddings_path))
        print(f"  Loaded {len(existing_embeddings)} existing embeddings")
    else:
        existing_embeddings = {}
        print(f"  No existing embeddings found")

    # Find samples without embeddings
    missing_samples = []
    for sample in metadata:
        if sample['id'] not in existing_embeddings:
            missing_samples.append(sample)

    print(f"  Found {len(missing_samples)} samples without embeddings")

    if len(missing_samples) == 0:
        print(f"  ✓ All samples already have embeddings!")
        return

    # Initialize emotion encoder
    print(f"  Loading WavLM encoder...")
    encoder = EmotionEncoder(model_type="wavlm")
    encoder.eval()

    # Extract embeddings
    print(f"  Extracting embeddings...")
    new_embeddings = {}

    with torch.no_grad():
        for sample in tqdm(missing_samples, desc="  Extracting"):
            sample_id = sample['id']
            audio_path = sample['audio_path']

            try:
                # Load audio
                import librosa
                audio, sr = librosa.load(audio_path, sr=16000)

                # Convert to tensor
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

                # Extract embedding
                embedding, _ = encoder.forward(audio_tensor)
                embedding_np = embedding.detach().cpu().numpy()[0]

                new_embeddings[sample_id] = embedding_np

            except Exception as e:
                print(f"\n  ⚠ Failed to process {sample_id}: {e}")
                # Use zero embedding as fallback
                new_embeddings[sample_id] = np.zeros(256, dtype=np.float32)

    # Merge with existing embeddings
    all_embeddings = {**existing_embeddings, **new_embeddings}

    # Save
    np.savez_compressed(embeddings_path, **all_embeddings)
    print(f"  ✓ Saved {len(all_embeddings)} total embeddings ({len(new_embeddings)} new)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", nargs="+", required=True,
                        help="Data directories to process")
    args = parser.parse_args()

    print("=" * 80)
    print("Extracting Missing WavLM Embeddings")
    print("=" * 80)

    for data_dir in args.data_dirs:
        print()
        extract_missing_embeddings(data_dir)

    print("\n" + "=" * 80)
    print("✅ Embedding extraction complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
