#!/usr/bin/env python3
"""
Extract feature subsets for ablation studies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
from pathlib import Path

# Feature indices
# Prosodic: 0-12 (13D)
# Spectral: 13-42 (30D: 13 MFCCs + 13 deltas + 4 spectral stats)
# Formants: 43-45 (3D)

FEATURE_GROUPS = {
    'prosodic': list(range(0, 13)),      # 13D
    'spectral': list(range(13, 43)),     # 30D
    'formants': list(range(43, 46)),     # 3D
}

def extract_subset(input_npz: str, output_npz: str, subset_type: str):
    """
    Extract feature subset from full acoustic features.

    Args:
        input_npz: Path to full acoustic_embeddings.npz
        output_npz: Path to save subset
        subset_type: One of:
            - 'prosodic_only': Only prosodic features (13D)
            - 'spectral_only': Only spectral features (30D)
            - 'formants_only': Only formant features (3D)
            - 'no_prosodic': All except prosodic (33D)
            - 'no_spectral': All except spectral (16D)
            - 'no_formants': All except formants (43D)
    """
    print(f"Loading from {input_npz}")
    data = np.load(input_npz)

    # Determine indices to keep
    if subset_type == 'prosodic_only':
        indices = FEATURE_GROUPS['prosodic']
        dim = 13
    elif subset_type == 'spectral_only':
        indices = FEATURE_GROUPS['spectral']
        dim = 30
    elif subset_type == 'formants_only':
        indices = FEATURE_GROUPS['formants']
        dim = 3
    elif subset_type == 'no_prosodic':
        indices = FEATURE_GROUPS['spectral'] + FEATURE_GROUPS['formants']
        dim = 33
    elif subset_type == 'no_spectral':
        indices = FEATURE_GROUPS['prosodic'] + FEATURE_GROUPS['formants']
        dim = 16
    elif subset_type == 'no_formants':
        indices = FEATURE_GROUPS['prosodic'] + FEATURE_GROUPS['spectral']
        dim = 43
    else:
        raise ValueError(f"Unknown subset_type: {subset_type}")

    print(f"Subset: {subset_type}")
    print(f"Extracting indices: {indices[:5]}... ({len(indices)} total)")
    print(f"Output dimension: {dim}D")

    # Extract subset for all samples
    subset_data = {}
    for key in data.files:
        full_features = data[key]
        if len(full_features.shape) == 2:
            full_features = full_features[0]  # Remove batch dim if present

        subset_features = full_features[indices]
        subset_data[key] = subset_features

    # Save
    print(f"Saving {len(subset_data)} samples to {output_npz}")
    np.savez_compressed(output_npz, **subset_data)

    # Verify
    saved = np.load(output_npz)
    sample_key = list(saved.files)[0]
    sample_shape = saved[sample_key].shape
    print(f"✓ Saved successfully. Sample shape: {sample_shape}")
    assert sample_shape[0] == dim, f"Expected {dim}D, got {sample_shape}"

    return dim

def main():
    parser = argparse.ArgumentParser(description="Extract acoustic feature subsets")
    parser.add_argument("--input", type=str, required=True,
                        help="Input acoustic_embeddings.npz file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .npz file")
    parser.add_argument("--subset", type=str, required=True,
                        choices=['prosodic_only', 'spectral_only', 'formants_only',
                                 'no_prosodic', 'no_spectral', 'no_formants'],
                        help="Feature subset to extract")

    args = parser.parse_args()

    extract_subset(args.input, args.output, args.subset)
    print("\n✅ Feature subset extraction complete!")

if __name__ == '__main__':
    main()
