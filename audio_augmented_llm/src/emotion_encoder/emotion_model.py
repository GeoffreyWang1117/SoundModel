"""
Emotion/Prosody Encoder for extracting emotion embeddings from audio.
Supports WavLM and Whisper-based encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2FeatureExtractor, WhisperModel, WhisperFeatureExtractor
import librosa
import numpy as np
from typing import Optional, Tuple


class EmotionEncoder(nn.Module):
    """
    Emotion encoder that extracts continuous emotion embeddings from audio.
    """

    def __init__(
        self,
        model_type: str = "wavlm",
        model_name: str = "microsoft/wavlm-large",
        embedding_dim: int = 256,
        num_emotion_classes: int = 7,
        freeze_encoder: bool = False,
        target_sample_rate: int = 16000,
    ):
        """
        Initialize emotion encoder.

        Args:
            model_type: Type of encoder ("wavlm" or "whisper")
            model_name: Pre-trained model name
            embedding_dim: Output emotion embedding dimension
            num_emotion_classes: Number of emotion classes for classification head
            freeze_encoder: Whether to freeze pre-trained weights
            target_sample_rate: Target sample rate for audio input
        """
        super().__init__()

        self.model_type = model_type
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_emotion_classes = num_emotion_classes
        self.freeze_encoder = freeze_encoder
        self.target_sample_rate = target_sample_rate

        # Load encoder and feature extractor
        self._load_encoder()

        # Get encoder hidden size
        if model_type == "wavlm":
            self.encoder_hidden_size = self.encoder.config.hidden_size
        elif model_type == "whisper":
            self.encoder_hidden_size = self.encoder.config.d_model

        # Projection layers
        self.emotion_projection = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
        )

        # Emotion classification head (for supervised training)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_emotion_classes),
        )

    def _load_encoder(self):
        """Load pre-trained encoder model."""
        if self.model_type == "wavlm":
            self.encoder = AutoModel.from_pretrained(self.model_name)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_name
            )
        elif self.model_type == "whisper":
            full_model = WhisperModel.from_pretrained(self.model_name)
            self.encoder = full_model.encoder
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                self.model_name
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Freeze encoder if specified
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def preprocess_audio(
        self, audio_path: str, max_length: float = 10.0
    ) -> torch.Tensor:
        """
        Load and preprocess audio file.

        Args:
            audio_path: Path to audio file
            max_length: Maximum audio length in seconds

        Returns:
            Preprocessed audio tensor
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.target_sample_rate)

        # Truncate if too long
        max_samples = int(max_length * self.target_sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        return audio

    def forward(
        self,
        audio_input: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass to extract emotion embeddings.

        Args:
            audio_input: Raw audio waveform [batch_size, audio_length]
            return_logits: Whether to return emotion classification logits

        Returns:
            emotion_embedding: [batch_size, embedding_dim]
            emotion_logits: [batch_size, num_emotion_classes] (if return_logits=True)
        """
        # Extract features from encoder
        if self.model_type == "wavlm":
            encoder_outputs = self.encoder(audio_input)
            hidden_states = encoder_outputs.last_hidden_state  # [B, T, H]
        elif self.model_type == "whisper":
            encoder_outputs = self.encoder(audio_input)
            hidden_states = encoder_outputs.last_hidden_state  # [B, T, H]

        # Temporal pooling (mean over time)
        pooled_output = hidden_states.mean(dim=1)  # [B, H]

        # Project to emotion embedding space
        emotion_embedding = self.emotion_projection(pooled_output)  # [B, embedding_dim]

        # Normalize embedding
        emotion_embedding = F.normalize(emotion_embedding, p=2, dim=-1)

        if return_logits:
            emotion_logits = self.emotion_classifier(emotion_embedding)
            return emotion_embedding, emotion_logits
        else:
            return emotion_embedding, None

    def encode_from_file(
        self, audio_path: str, device: str = "cuda"
    ) -> torch.Tensor:
        """
        Encode audio file to emotion embedding.

        Args:
            audio_path: Path to audio file
            device: Device to run inference on

        Returns:
            Emotion embedding tensor [1, embedding_dim]
        """
        # Preprocess audio
        audio = self.preprocess_audio(audio_path)

        # Extract features using feature extractor
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
        )

        # Move to device
        input_values = inputs.input_values.to(device)

        # Forward pass
        with torch.no_grad():
            emotion_embedding, _ = self.forward(input_values)

        return emotion_embedding

    def batch_encode_from_files(
        self, audio_paths: list[str], device: str = "cuda", batch_size: int = 8
    ) -> torch.Tensor:
        """
        Batch encode multiple audio files.

        Args:
            audio_paths: List of audio file paths
            device: Device to run inference on
            batch_size: Batch size for inference

        Returns:
            Emotion embeddings [num_files, embedding_dim]
        """
        self.eval()
        self.to(device)

        all_embeddings = []

        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i : i + batch_size]

            # Load and preprocess batch
            batch_audios = [self.preprocess_audio(path) for path in batch_paths]

            # Extract features
            inputs = self.feature_extractor(
                batch_audios,
                sampling_rate=self.target_sample_rate,
                return_tensors="pt",
                padding=True,
            )

            input_values = inputs.input_values.to(device)

            # Forward pass
            with torch.no_grad():
                emotion_embeddings, _ = self.forward(input_values)

            all_embeddings.append(emotion_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)


# Example usage
if __name__ == "__main__":
    # Initialize emotion encoder
    emotion_encoder = EmotionEncoder(
        model_type="wavlm",
        model_name="microsoft/wavlm-large",
        embedding_dim=256,
        num_emotion_classes=7,
    )

    # Create dummy audio input
    dummy_audio = torch.randn(2, 16000 * 3)  # 2 samples, 3 seconds each

    # Forward pass
    emotion_emb, emotion_logits = emotion_encoder(dummy_audio, return_logits=True)

    print(f"Emotion embedding shape: {emotion_emb.shape}")
    print(f"Emotion logits shape: {emotion_logits.shape}")
