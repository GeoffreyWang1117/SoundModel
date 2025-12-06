"""
TTS Engine for generating emotion-rich audio from teacher LLM text outputs.
Supports XTTS v2 and CosyVoice models.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import soundfile as sf


class TTSEngine:
    """
    Text-to-Speech engine wrapper supporting multiple TTS backends.
    """

    def __init__(
        self,
        model_type: str = "xtts",
        model_path: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_rate: int = 22050,
        language: str = "en",
    ):
        """
        Initialize TTS engine.

        Args:
            model_type: Type of TTS model ("xtts" or "cosyvoice")
            model_path: Path or name of the TTS model
            device: Device to run inference on
            sample_rate: Target sample rate for generated audio
            language: Target language code
        """
        self.model_type = model_type
        self.model_path = model_path
        self.device = device
        self.sample_rate = sample_rate
        self.language = language

        self._load_model()

    def _load_model(self):
        """Load the TTS model."""
        if self.model_type == "xtts":
            self._load_xtts()
        elif self.model_type == "cosyvoice":
            self._load_cosyvoice()
        else:
            raise ValueError(f"Unsupported TTS model type: {self.model_type}")

    def _load_xtts(self):
        """Load XTTS v2 model."""
        try:
            import os
            # Set environment variable to agree to XTTS license automatically
            os.environ['COQUI_TOS_AGREED'] = '1'

            from TTS.api import TTS
            self.model = TTS(self.model_path, gpu=(self.device == "cuda")).to(self.device)
            print(f"XTTS model loaded on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load XTTS model: {e}")

    def _load_cosyvoice(self):
        """Load CosyVoice model."""
        # Placeholder for CosyVoice implementation
        raise NotImplementedError("CosyVoice support coming soon!")

    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        speaker: Optional[str] = None,
        emotion: Optional[str] = None,
        speed: float = 1.0,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            speaker_wav: Path to reference speaker audio (for voice cloning)
            speaker: Speaker name for built-in speakers (XTTS only)
            emotion: Emotion label to guide synthesis (if supported)
            speed: Speech speed multiplier
            output_path: Optional path to save audio file

        Returns:
            Audio waveform as numpy array
        """
        if self.model_type == "xtts":
            return self._synthesize_xtts(text, speaker_wav, speaker, speed, output_path)
        elif self.model_type == "cosyvoice":
            return self._synthesize_cosyvoice(text, emotion, speed, output_path)

    def _synthesize_xtts(
        self,
        text: str,
        speaker_wav: Optional[str],
        speaker: Optional[str],
        speed: float,
        output_path: Optional[str],
    ) -> np.ndarray:
        """Synthesize using XTTS."""
        try:
            # Generate audio
            if speaker_wav:
                # Use provided speaker wav for voice cloning
                wav = self.model.tts(
                    text=text,
                    speaker_wav=speaker_wav,
                    language=self.language,
                )
            elif speaker:
                # Use specified built-in speaker
                wav = self.model.tts(
                    text=text,
                    speaker=speaker,
                    language=self.language,
                )
            else:
                # Use default built-in speaker for reproducibility
                # XTTS v2 has built-in speakers, we use "Claribel Dervla" as default
                wav = self.model.tts(
                    text=text,
                    speaker="Claribel Dervla",  # Use consistent built-in speaker
                    language=self.language,
                )

            # Convert to numpy array
            wav_np = np.array(wav)

            # Save if output path specified
            if output_path:
                sf.write(output_path, wav_np, self.sample_rate)

            return wav_np

        except Exception as e:
            raise RuntimeError(f"XTTS synthesis failed: {e}")

    def _synthesize_cosyvoice(
        self,
        text: str,
        emotion: Optional[str],
        speed: float,
        output_path: Optional[str],
    ) -> np.ndarray:
        """Synthesize using CosyVoice."""
        raise NotImplementedError("CosyVoice support coming soon!")

    def batch_synthesize(
        self,
        texts: list[str],
        output_dir: str,
        **kwargs,
    ) -> list[str]:
        """
        Batch synthesize multiple texts.

        Args:
            texts: List of input texts
            output_dir: Directory to save audio files
            **kwargs: Additional arguments for synthesize()

        Returns:
            List of output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []
        for i, text in enumerate(texts):
            output_path = output_dir / f"audio_{i:05d}.wav"
            self.synthesize(text, output_path=str(output_path), **kwargs)
            output_paths.append(str(output_path))

        return output_paths


# Example usage
if __name__ == "__main__":
    # Initialize TTS engine
    tts = TTSEngine(model_type="xtts")

    # Synthesize single text
    text = "Hello, I am feeling very happy today!"
    audio = tts.synthesize(text, output_path="test_output.wav")

    print(f"Generated audio shape: {audio.shape}")
    print(f"Sample rate: {tts.sample_rate}")
