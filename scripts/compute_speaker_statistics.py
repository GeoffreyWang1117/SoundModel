#!/usr/bin/env python3
"""
Compute per-speaker acoustic statistics for normalization.

This script computes mean and std for all acoustic features across
all samples from a single speaker. Used for z-score normalization.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
import librosa

from audio_augmented_llm.src.emotion_encoder.acoustic_features import AcousticFeatureExtractor


def compute_speaker_statistics(
    data_dir: str,
    speaker_name: str,
    output_path: str = None
):
    """
    Compute statistics for all acoustic features from one speaker.

    Args:
        data_dir: Directory containing audio files
        speaker_name: Name of the speaker (for documentation)
        output_path: Where to save statistics JSON
    """
    print("=" * 80)
    print(f"Computing Speaker Statistics")
    print("=" * 80)
    print(f"Data dir: {data_dir}")
    print(f"Speaker: {speaker_name}")
    print()

    data_path = Path(data_dir)

    # Load metadata to get audio files
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        audio_files = [item['audio_path'] for item in metadata]
        print(f"Found {len(audio_files)} audio files from metadata")
    else:
        # Fallback: glob audio directory
        audio_files = glob(str(data_path / "audio" / "*.wav"))
        print(f"Found {len(audio_files)} audio files from directory")

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {data_dir}")

    # Initialize feature extractor
    extractor = AcousticFeatureExtractor(sample_rate=16000)

    # Collect raw features for all samples
    all_features = {
        'pitch': [],
        'energy': [],
        'formants': {'f1': [], 'f2': [], 'f3': []},
        'mfcc': [],
        'spectral_centroid': [],
        'spectral_rolloff': [],
        'spectral_flux': [],
        'spectral_bandwidth': [],
        'zcr': []
    }

    print(f"Extracting features from {len(audio_files)} files...")
    for audio_path in tqdm(audio_files, desc="Processing"):
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)

            # Extract F0
            f0 = librosa.yin(audio, fmin=65, fmax=2093, sr=sr)
            f0_valid = f0[f0 < 10000]
            if len(f0_valid) > 0:
                all_features['pitch'].extend(f0_valid)

            # Extract energy (RMS)
            rms = librosa.feature.rms(y=audio)[0]
            all_features['energy'].extend(rms)

            # Extract formants (skip if fails - not critical for normalization)
            try:
                formants = extractor.extract_formant_features(audio)
                if 'f1' in formants and 'f2' in formants and 'f3' in formants:
                    all_features['formants']['f1'].append(formants['f1'])
                    all_features['formants']['f2'].append(formants['f2'])
                    all_features['formants']['f3'].append(formants['f3'])
            except:
                pass  # Skip formants for this sample

            # Extract MFCCs (just for statistics, we'll normalize individual coeffs)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
            all_features['mfcc'].append(mfccs)

            # Extract spectral features
            spectral_features = extractor.extract_spectral_features(audio)
            all_features['spectral_centroid'].append(spectral_features['spectral_centroid'])
            all_features['spectral_rolloff'].append(spectral_features['spectral_rolloff'])
            all_features['spectral_flux'].append(spectral_features['spectral_flux'])
            all_features['spectral_bandwidth'].append(spectral_features['spectral_bandwidth'])

            # ZCR
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            all_features['zcr'].extend(zcr)

        except Exception as e:
            print(f"\n⚠ Warning: Failed to process {audio_path}: {e}")
            continue

    # Compute statistics
    print("\nComputing statistics...")
    statistics = {
        'speaker': speaker_name,
        'n_samples': len(audio_files),
        'features': {}
    }

    # Pitch statistics
    pitch_data = np.array(all_features['pitch'])
    statistics['features']['pitch'] = {
        'mean': float(np.mean(pitch_data)),
        'std': float(np.std(pitch_data)),
        'median': float(np.median(pitch_data)),
        'min': float(np.min(pitch_data)),
        'max': float(np.max(pitch_data)),
        'p5': float(np.percentile(pitch_data, 5)),
        'p95': float(np.percentile(pitch_data, 95))
    }

    # Energy statistics
    energy_data = np.array(all_features['energy'])
    statistics['features']['energy'] = {
        'mean': float(np.mean(energy_data)),
        'std': float(np.std(energy_data)),
        'median': float(np.median(energy_data)),
        'min': float(np.min(energy_data)),
        'max': float(np.max(energy_data))
    }

    # Formant statistics (if we have data)
    statistics['features']['formants'] = {}
    for formant_name in ['f1', 'f2', 'f3']:
        formant_data = all_features['formants'][formant_name]
        if len(formant_data) > 0:
            formant_data = np.array(formant_data)
            statistics['features']['formants'][formant_name] = {
                'mean': float(np.mean(formant_data)),
                'std': float(np.std(formant_data)),
                'n_samples': len(formant_data)
            }
        else:
            # No formant data - use defaults
            statistics['features']['formants'][formant_name] = {
                'mean': 0.0,
                'std': 1.0,
                'n_samples': 0
            }

    # MFCC statistics (per coefficient)
    mfcc_concat = np.concatenate([m.T for m in all_features['mfcc']], axis=0)  # [n_frames, 13]
    statistics['features']['mfcc'] = {
        f'mfcc_{i}': {
            'mean': float(np.mean(mfcc_concat[:, i])),
            'std': float(np.std(mfcc_concat[:, i]))
        }
        for i in range(13)
    }

    # Spectral statistics
    for feature_name in ['spectral_centroid', 'spectral_rolloff', 'spectral_flux', 'spectral_bandwidth']:
        data = np.array(all_features[feature_name])
        statistics['features'][feature_name] = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data))
        }

    # ZCR statistics
    zcr_data = np.array(all_features['zcr'])
    statistics['features']['zcr'] = {
        'mean': float(np.mean(zcr_data)),
        'std': float(np.std(zcr_data))
    }

    # Save statistics
    if output_path is None:
        output_path = f"speaker_stats_{speaker_name.replace(' ', '_').lower()}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print("✅ Statistics computed successfully!")
    print("=" * 80)
    print(f"\n📊 Speaker: {speaker_name}")
    print(f"   Samples: {len(audio_files)}")
    print(f"\n   Pitch:")
    print(f"      Mean: {statistics['features']['pitch']['mean']:.2f} Hz")
    print(f"      Std:  {statistics['features']['pitch']['std']:.2f} Hz")
    print(f"      Range: [{statistics['features']['pitch']['p5']:.2f}, {statistics['features']['pitch']['p95']:.2f}] Hz (5-95 percentile)")
    print(f"\n   Energy:")
    print(f"      Mean: {statistics['features']['energy']['mean']:.4f}")
    print(f"      Std:  {statistics['features']['energy']['std']:.4f}")
    print(f"\n   Formants:")
    for fname in ['f1', 'f2', 'f3']:
        print(f"      {fname.upper()}: {statistics['features']['formants'][fname]['mean']:.2f} ± {statistics['features']['formants'][fname]['std']:.2f} Hz")
    print(f"\n📁 Saved to: {output_path}")
    print()

    return statistics


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-speaker acoustic statistics for normalization"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--speaker", type=str, required=True,
                        help="Speaker name (e.g., 'Claribel Dervla')")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for statistics JSON")

    args = parser.parse_args()

    compute_speaker_statistics(
        args.data_dir,
        args.speaker,
        args.output
    )


if __name__ == '__main__':
    main()
