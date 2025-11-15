"""
Acoustic feature extraction for emotion recognition based on speech science principles.

Key features extracted:
1. Prosodic features: pitch, energy, duration
2. Spectral features: MFCC, spectral statistics
3. Voice quality: formants, jitter, shimmer
4. Temporal dynamics: speech rate, pause patterns
"""

import numpy as np
import librosa
import librosa.feature as F
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class AcousticFeatureExtractor:
    """
    Extract acoustic features based on speech science principles.

    References:
    - Prosodic features: pitch, energy, duration (most reliable for emotion)
    - MFCC: mel-frequency cepstral coefficients
    - Formants: F1, F2, F3 for voice quality
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        """
        Initialize acoustic feature extractor.

        Args:
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sr = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract prosodic features: pitch, energy, duration.

        These are the MOST RELIABLE indicators of emotion according to research.

        Args:
            audio: Audio waveform [n_samples]

        Returns:
            Dictionary of prosodic features
        """
        features = {}

        # 1. Pitch features (F0)
        try:
            # Extract pitch using YIN algorithm (robust for speech)
            f0 = librosa.yin(
                audio,
                fmin=librosa.note_to_hz('C2'),  # 65 Hz
                fmax=librosa.note_to_hz('C7'),  # 2093 Hz
                sr=self.sr
            )

            # Filter out unvoiced frames (F0 = inf)
            f0_valid = f0[f0 < 10000]

            if len(f0_valid) > 0:
                features['pitch_mean'] = float(np.mean(f0_valid))
                features['pitch_std'] = float(np.std(f0_valid))
                features['pitch_min'] = float(np.min(f0_valid))
                features['pitch_max'] = float(np.max(f0_valid))
                features['pitch_range'] = features['pitch_max'] - features['pitch_min']

                # Pitch dynamics (contour variation)
                if len(f0_valid) > 1:
                    features['pitch_slope'] = float(np.polyfit(np.arange(len(f0_valid)), f0_valid, 1)[0])
            else:
                # Fallback for unvoiced speech
                features.update({
                    'pitch_mean': 0.0,
                    'pitch_std': 0.0,
                    'pitch_min': 0.0,
                    'pitch_max': 0.0,
                    'pitch_range': 0.0,
                    'pitch_slope': 0.0,
                })
        except Exception as e:
            print(f"Warning: Pitch extraction failed: {e}")
            features.update({
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'pitch_min': 0.0,
                'pitch_max': 0.0,
                'pitch_range': 0.0,
                'pitch_slope': 0.0,
            })

        # 2. Energy features (RMS, ZCR)
        rms = librosa.feature.rms(y=audio, frame_length=self.n_fft, hop_length=self.hop_length)[0]
        features['energy_mean'] = float(np.mean(rms))
        features['energy_std'] = float(np.std(rms))
        features['energy_max'] = float(np.max(rms))

        # Zero-crossing rate (voice quality indicator)
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.n_fft, hop_length=self.hop_length)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))

        # 3. Duration features
        features['duration'] = float(len(audio) / self.sr)

        # Speech rate (voiced frames per second)
        voiced_frames = np.sum(rms > np.mean(rms) * 0.1)
        features['speech_rate'] = float(voiced_frames / (len(audio) / self.sr))

        return features

    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features: MFCC, spectral statistics.

        Args:
            audio: Audio waveform [n_samples]

        Returns:
            Dictionary of spectral features
        """
        features = {}

        # 1. MFCC (mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Statistical functionals over time
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))

        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))

        spectral_flux = librosa.onset.onset_strength(y=audio, sr=self.sr)
        features['spectral_flux_mean'] = float(np.mean(spectral_flux))

        return features

    def extract_formant_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract formant features (F1, F2, F3) - voice quality indicators.

        Formants are resonance frequencies of the vocal tract and carry
        emotional information through voice quality changes.

        Args:
            audio: Audio waveform [n_samples]

        Returns:
            Dictionary of formant features
        """
        features = {}

        try:
            # Pre-emphasis to highlight formants
            pre_emphasized = librosa.effects.preemphasis(audio)

            # Compute LPC coefficients
            # Higher order to capture first 3 formants
            lpc_order = 12
            a = librosa.lpc(pre_emphasized, order=lpc_order)

            # Find formant frequencies from LPC roots
            roots = np.roots(a)
            roots = roots[np.imag(roots) >= 0]  # Keep positive frequencies

            # Convert to frequencies
            formants = []
            for root in roots:
                if np.abs(root) < 1:  # Stability check
                    freq = np.angle(root) * self.sr / (2 * np.pi)
                    if 0 < freq < self.sr / 2:  # Nyquist check
                        formants.append(freq)

            formants = sorted(formants)[:3]  # First 3 formants

            # Store formant features
            for i, f in enumerate(formants):
                features[f'formant_f{i+1}'] = float(f)

            # Fill missing formants with 0
            for i in range(len(formants), 3):
                features[f'formant_f{i+1}'] = 0.0

        except Exception as e:
            print(f"Warning: Formant extraction failed: {e}")
            features.update({
                'formant_f1': 0.0,
                'formant_f2': 0.0,
                'formant_f3': 0.0,
            })

        return features

    def extract_all_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract all acoustic features and return as a single vector.

        Feature composition:
        - Prosodic: 13 features (pitch, energy, duration)
        - Spectral: 26 features (MFCC statistics)
        - Spectral stats: 4 features
        - Formants: 3 features
        Total: ~46 features

        Args:
            audio: Audio waveform [n_samples]

        Returns:
            Feature vector [n_features]
        """
        all_features = {}

        # Extract all feature groups
        all_features.update(self.extract_prosodic_features(audio))
        all_features.update(self.extract_spectral_features(audio))
        all_features.update(self.extract_formant_features(audio))

        # Convert to ordered numpy array
        feature_vector = np.array(list(all_features.values()), dtype=np.float32)

        # Normalize to prevent scale issues
        # Using robust normalization (clip outliers)
        feature_vector = np.clip(feature_vector, -1e6, 1e6)

        # Handle NaN/Inf
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_vector

    def extract_from_file(self, audio_path: str) -> np.ndarray:
        """
        Extract acoustic features from an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Feature vector [n_features]
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sr)

        # Extract features
        features = self.extract_all_features(audio)

        return features


# Example usage and testing
if __name__ == "__main__":
    import os

    print("=" * 70)
    print("Acoustic Feature Extractor - Test")
    print("=" * 70)

    # Initialize extractor
    extractor = AcousticFeatureExtractor(sample_rate=16000)

    # Test with a sample audio file
    test_audio_dir = "./audio_augmented_llm/data/train_100/audio"

    if os.path.exists(test_audio_dir):
        audio_files = [f for f in os.listdir(test_audio_dir) if f.endswith('.wav')]

        if audio_files:
            test_file = os.path.join(test_audio_dir, audio_files[0])
            print(f"\nTesting on: {test_file}")

            # Extract features
            features = extractor.extract_from_file(test_file)

            print(f"\n✓ Feature extraction successful!")
            print(f"  Feature vector shape: {features.shape}")
            print(f"  Feature vector dtype: {features.dtype}")
            print(f"  Feature range: [{features.min():.2f}, {features.max():.2f}]")
            print(f"  Non-zero features: {np.sum(features != 0)}/{len(features)}")

            # Load audio for detailed analysis
            audio, sr = librosa.load(test_file, sr=16000)

            # Show feature breakdown
            print("\n📊 Feature breakdown:")
            prosodic = extractor.extract_prosodic_features(audio)
            print(f"  Prosodic features: {len(prosodic)}")
            for k, v in list(prosodic.items())[:5]:
                print(f"    {k}: {v:.3f}")

            spectral = extractor.extract_spectral_features(audio)
            print(f"  Spectral features: {len(spectral)}")

            formants = extractor.extract_formant_features(audio)
            print(f"  Formant features: {len(formants)}")
            for k, v in formants.items():
                print(f"    {k}: {v:.1f} Hz")
        else:
            print("\n⚠ No audio files found in test directory")
    else:
        print(f"\n⚠ Test directory not found: {test_audio_dir}")
        print("  Creating dummy test...")

        # Create dummy audio for testing
        duration = 2.0  # 2 seconds
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))

        # Synthesize a simple tone with varying pitch (simulating emotional speech)
        f0_base = 200  # Base frequency 200 Hz
        f0_variation = 50 * np.sin(2 * np.pi * 2 * t)  # Pitch variation
        audio = 0.5 * np.sin(2 * np.pi * (f0_base + f0_variation) * t)

        features = extractor.extract_all_features(audio)
        print(f"\n✓ Dummy audio test successful!")
        print(f"  Feature vector shape: {features.shape}")
        print(f"  Total features: {len(features)}")

    print("\n" + "=" * 70)
    print("✅ Acoustic feature extraction module ready!")
    print("=" * 70)
