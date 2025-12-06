#!/usr/bin/env python3
"""
Batch extract relative (speaker-invariant) acoustic features.

This replaces absolute acoustic features with ratio-based relative features
that should generalize better across speakers.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from audio_augmented_llm.src.emotion_encoder.relative_features import RelativeAcousticFeatures


def extract_relative_features_batch(
    data_dir: str,
    output_path: str = None
):
    """
    Extract relative acoustic features for all samples in a dataset.

    Args:
        data_dir: Directory containing metadata.json and audio files
        output_path: Where to save features (default: data_dir/relative_embeddings.npz)
    """
    print("=" * 80)
    print("Relative Acoustic Feature Extraction (Batch)")
    print("=" * 80)
    print(f"Data directory: {data_dir}")

    data_path = Path(data_dir)

    # Set output path
    if output_path is None:
        output_path = data_path / "relative_embeddings.npz"
    else:
        output_path = Path(output_path)

    print(f"Output: {output_path}")
    print()

    # Load metadata
    metadata_path = data_path / "metadata.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"📊 Dataset: {len(metadata)} samples")
    print()

    # Initialize feature extractor
    print("🔬 Initializing relative feature extractor...")
    extractor = RelativeAcousticFeatures(sample_rate=16000)
    print()

    # Extract features
    print("🎵 Extracting relative acoustic features...")
    all_embeddings = {}
    successful = 0
    failed = 0

    for item in tqdm(metadata, desc="Extracting"):
        sample_id = item['id']
        audio_path = item['audio_path']

        try:
            # Extract relative features
            features = extractor.extract_from_file(audio_path)
            all_embeddings[sample_id] = features
            successful += 1

        except Exception as e:
            # Fill with zeros on failure
            print(f"\n⚠ Failed {sample_id}: {e}")
            all_embeddings[sample_id] = np.zeros(44, dtype=np.float32)
            failed += 1

    # Save embeddings
    print()
    print("💾 Saving relative embeddings...")
    np.savez_compressed(output_path, **all_embeddings)

    # Compute statistics
    all_vecs = np.array(list(all_embeddings.values()))
    feature_means = np.mean(all_vecs, axis=0)
    feature_stds = np.std(all_vecs, axis=0)

    print()
    print("=" * 80)
    print("✅ Relative feature extraction complete!")
    print("=" * 80)
    print(f"\n📊 Statistics:")
    print(f"   Total samples: {len(metadata)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {successful / len(metadata) * 100:.1f}%")
    print()
    print(f"🔬 Feature details:")
    print(f"   Dimension: {all_vecs.shape[1]}D")
    print(f"   Dtype: {all_vecs.dtype}")
    print()
    print(f"   Overall statistics:")
    print(f"   Mean range: [{feature_means.min():.2f}, {feature_means.max():.2f}]")
    print(f"   Std range: [{feature_stds.min():.2f}, {feature_stds.max():.2f}]")
    print()
    print(f"📁 Saved to: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract relative acoustic features (batch)"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing metadata.json and audio")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for embeddings (default: data_dir/relative_embeddings.npz)")

    args = parser.parse_args()

    extract_relative_features_batch(
        args.data_dir,
        args.output
    )


if __name__ == '__main__':
    main()
