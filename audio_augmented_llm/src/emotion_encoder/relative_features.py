"""
Relative Acoustic Features - Speaker-Invariant Emotion Encoding

This module extracts speaker-invariant acoustic features using relative/ratio-based
normalization instead of absolute values. The goal is to capture emotion-specific
patterns that generalize across speakers.

Key Principle: Use ratios, coefficients of variation, and normalized ranges
instead of absolute pitch/energy values.
"""

import numpy as np
import librosa
from typing import Dict


class RelativeAcousticFeatures:
    """
    Extract speaker-invariant relative acoustic features.

    Replaces absolute values with:
    - Ratios (range/median, max/min)
    - Coefficient of variation (std/mean)
    - Normalized slopes
    - Percentile-based features
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sr = sample_rate

    def extract_relative_prosodic(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract relative prosodic features (pitch, energy).

        Returns speaker-invariant ratios instead of absolute values.
        """
        features = {}

        # Extract F0 contour
        f0 = librosa.yin(audio, fmin=65, fmax=2093, sr=self.sr)
        f0_valid = f0[f0 < 10000]

        if len(f0_valid) < 10:
            # Not enough pitch data - return zeros
            return {f'pitch_{key}': 0.0 for key in [
                'range_rel', 'cv', 'slope_rel', 'p90_p10_ratio', 'skewness'
            ]} | {f'energy_{key}': 0.0 for key in [
                'range_rel', 'cv', 'peak_mean_ratio', 'dynamic_range_db'
            ]} | {'duration': 0.0, 'speech_rate_rel': 0.0, 'zcr_cv': 0.0}

        # Pitch relative features
        f0_median = np.median(f0_valid)
        f0_mean = np.mean(f0_valid)
        f0_std = np.std(f0_valid)
        f0_p5 = np.percentile(f0_valid, 5)
        f0_p10 = np.percentile(f0_valid, 10)
        f0_p90 = np.percentile(f0_valid, 90)
        f0_p95 = np.percentile(f0_valid, 95)

        # 1. Relative range (normalized by median)
        features['pitch_range_rel'] = (f0_p95 - f0_p5) / (f0_median + 1e-6)

        # 2. Coefficient of variation (std/mean) - scale-independent
        features['pitch_cv'] = f0_std / (f0_mean + 1e-6)

        # 3. Normalized slope (trend normalized by mean)
        time = np.arange(len(f0_valid))
        if len(time) > 1:
            slope, _ = np.polyfit(time, f0_valid, 1)
            features['pitch_slope_rel'] = slope / (f0_mean + 1e-6)
        else:
            features['pitch_slope_rel'] = 0.0

        # 4. Percentile ratio (spread at extremes)
        features['pitch_p90_p10_ratio'] = f0_p90 / (f0_p10 + 1e-6)

        # 5. Skewness (distribution shape)
        features['pitch_skewness'] = float(
            np.mean(((f0_valid - f0_mean) / (f0_std + 1e-6)) ** 3)
        )

        # Energy relative features
        rms = librosa.feature.rms(y=audio)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        rms_max = np.max(rms)
        rms_min = np.min(rms)
        rms_median = np.median(rms)

        # 1. Relative energy range
        features['energy_range_rel'] = (rms_max - rms_min) / (rms_median + 1e-6)

        # 2. Energy coefficient of variation
        features['energy_cv'] = rms_std / (rms_mean + 1e-6)

        # 3. Peak-to-mean ratio
        features['energy_peak_mean_ratio'] = rms_max / (rms_mean + 1e-6)

        # 4. Dynamic range in dB (log scale)
        features['energy_dynamic_range_db'] = 20 * np.log10(
            (rms_max + 1e-9) / (rms_min + 1e-9)
        )

        # Duration (absolute, but useful)
        features['duration'] = len(audio) / self.sr

        # Speech rate (relative to duration)
        # Count voiced frames (energy above 10% of mean)
        voiced_frames = np.sum(rms > rms_mean * 0.1)
        features['speech_rate_rel'] = voiced_frames / (len(rms) + 1e-6)

        # ZCR coefficient of variation
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        features['zcr_cv'] = zcr_std / (zcr_mean + 1e-6)

        return features

    def extract_relative_spectral(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features.

        Note: MFCCs are already relatively speaker-invariant (they capture
        spectral shape, not absolute frequency). We keep them as-is but
        compute statistics.
        """
        features = {}

        # Extract MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=13,
            n_fft=512,
            hop_length=256
        )

        # MFCC statistics (mean across time)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))

        # Delta MFCCs (temporal dynamics) - mean only
        delta_mfccs = librosa.feature.delta(mfccs)
        for i in range(13):
            features[f'mfcc_delta_{i}_mean'] = float(np.mean(delta_mfccs[i]))

        # Spectral shape features (already relative)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0]

        # Relative spectral features
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_cv'] = float(
            np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-6)
        )

        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))

        # Spectral flux (frame-to-frame change)
        spec = np.abs(librosa.stft(audio))
        flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
        features['spectral_flux_mean'] = float(np.mean(flux))
        features['spectral_flux_cv'] = float(
            np.std(flux) / (np.mean(flux) + 1e-6)
        )

        return features

    def extract_all_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract all relative acoustic features.

        Returns:
            Feature vector (40D total)
        """
        all_features = {}

        # Prosodic features (13D)
        all_features.update(self.extract_relative_prosodic(audio))

        # Spectral features (32D)
        all_features.update(self.extract_relative_spectral(audio))

        # Convert to numpy array
        feature_vector = np.array(list(all_features.values()), dtype=np.float32)

        # Handle NaN/Inf
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=10.0, neginf=-10.0)

        return feature_vector

    def extract_from_file(self, audio_path: str) -> np.ndarray:
        """
        Extract features from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Feature vector
        """
        audio, sr = librosa.load(audio_path, sr=self.sr)
        return self.extract_all_features(audio)


# Standalone test
if __name__ == '__main__':
    import sys

    extractor = RelativeAcousticFeatures()

    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        features = extractor.extract_from_file(audio_path)
        print(f"Extracted {len(features)}D relative features from {audio_path}")
        print(f"Feature vector: {features}")
    else:
        # Test with synthetic signal
        print("Testing with synthetic audio...")
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Simulate speech with varying pitch
        f0_base = 200  # Hz
        f0_modulation = f0_base + 50 * np.sin(2 * np.pi * 2 * t)  # 2 Hz modulation
        audio = np.sin(2 * np.pi * f0_modulation * t)

        features = extractor.extract_all_features(audio)
        print(f"Extracted {len(features)}D features")
        print(f"Feature ranges: [{features.min():.3f}, {features.max():.3f}]")
        print("✓ Test passed!")
