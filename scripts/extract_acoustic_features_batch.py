#!/usr/bin/env python3
"""
Batch extract acoustic features for all audio samples.
Compare with WavLM embeddings.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from audio_augmented_llm.src.emotion_encoder.acoustic_features import AcousticFeatureExtractor

def extract_acoustic_features_batch(
    data_dir: str,
    output_path: str = None
):
    """
    Extract acoustic features for all audio files in dataset.

    Args:
        data_dir: Directory containing metadata.json and audio/
        output_path: Path to save acoustic embeddings (default: data_dir/acoustic_embeddings.npz)
    """
    data_path = Path(data_dir)

    if output_path is None:
        output_path = data_path / "acoustic_embeddings.npz"

    print("=" * 80)
    print("Acoustic Feature Extraction (Batch)")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Output: {output_path}")

    # Load metadata
    metadata_path = data_path / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"\n📊 Dataset: {len(metadata)} samples")

    # Initialize feature extractor
    print("\n🔬 Initializing acoustic feature extractor...")
    extractor = AcousticFeatureExtractor(sample_rate=16000)

    # Extract features for all samples
    print(f"\n🎵 Extracting acoustic features...")
    all_embeddings = {}
    successful = 0
    failed = 0

    for item in tqdm(metadata, desc="Extracting"):
        sample_id = item['id']
        # audio_path in metadata is already a full path, use it directly
        audio_path = item['audio_path']

        try:
            # Extract acoustic features
            features = extractor.extract_from_file(audio_path)
            all_embeddings[sample_id] = features
            successful += 1
        except Exception as e:
            print(f"\n⚠ Failed to extract {sample_id}: {e}")
            failed += 1
            # Use zero vector as fallback
            all_embeddings[sample_id] = np.zeros(46, dtype=np.float32)

    # Save embeddings
    print(f"\n💾 Saving acoustic embeddings...")
    np.savez_compressed(output_path, **all_embeddings)

    # Statistics
    print("\n" + "=" * 80)
    print("✅ Acoustic feature extraction complete!")
    print("=" * 80)
    print(f"\n📊 Statistics:")
    print(f"   Total samples: {len(metadata)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {successful / len(metadata) * 100:.1f}%")

    # Feature statistics
    if successful > 0:
        sample_features = list(all_embeddings.values())[0]
        print(f"\n🔬 Feature details:")
        print(f"   Dimension: {len(sample_features)}")
        print(f"   Dtype: {sample_features.dtype}")

        # Compute statistics across all samples
        all_features_array = np.array(list(all_embeddings.values()))
        print(f"\n   Overall statistics:")
        print(f"   Mean range: [{all_features_array.mean(axis=0).min():.2f}, {all_features_array.mean(axis=0).max():.2f}]")
        print(f"   Std range: [{all_features_array.std(axis=0).min():.2f}, {all_features_array.std(axis=0).max():.2f}]")

    print(f"\n📁 Saved to: {output_path}")
    print()

    return all_embeddings

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract acoustic features in batch")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output", type=str, default=None, help="Output path (optional)")

    args = parser.parse_args()

    extract_acoustic_features_batch(args.data_dir, args.output)

if __name__ == '__main__':
    main()
