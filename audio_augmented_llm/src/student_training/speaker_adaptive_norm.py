"""
Speaker Adaptive Normalization for Cross-Speaker Generalization.

This module implements speaker-conditioned normalization to disentangle
speaker identity from emotion information.

Key idea:
- Training: Use speaker-specific normalization parameters
- Inference: Remove speaker conditioning to force speaker-invariant representations
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class SpeakerAdaptiveNorm(nn.Module):
    """
    Speaker-conditioned normalization layer that learns speaker-specific
    scale and bias parameters.

    During training: Uses speaker ID to select speaker-specific parameters
    During inference: Uses averaged parameters or zero conditioning
    """

    def __init__(
        self,
        num_features: int,
        num_speakers: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        track_running_stats: bool = True
    ):
        """
        Initialize Speaker Adaptive Normalization.

        Args:
            num_features: Number of features to normalize (emotion_dim)
            num_speakers: Number of speakers in training set
            eps: Epsilon for numerical stability
            momentum: Momentum for running statistics
            track_running_stats: Whether to track running mean/var
        """
        super().__init__()

        self.num_features = num_features
        self.num_speakers = num_speakers
        self.eps = eps
        self.momentum = momentum

        # Speaker-specific affine parameters
        # Shape: [num_speakers, num_features]
        self.speaker_scales = nn.Parameter(torch.ones(num_speakers, num_features))
        self.speaker_biases = nn.Parameter(torch.zeros(num_speakers, num_features))

        # Global normalization statistics (optional)
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

        self.track_running_stats = track_running_stats

    def forward(
        self,
        x: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
        inference_mode: str = "average"  # "average", "zero", or "neutral"
    ) -> torch.Tensor:
        """
        Apply speaker-adaptive normalization.

        Args:
            x: Input tensor [batch_size, num_features]
            speaker_ids: Speaker IDs [batch_size] (0 to num_speakers-1)
                        If None, use inference mode
            inference_mode: How to handle missing speaker_ids
                - "average": Use average of all speaker parameters
                - "zero": Use no conditioning (scale=1, bias=0)
                - "neutral": Use neutral speaker (speaker_id=0)

        Returns:
            Normalized tensor [batch_size, num_features]
        """
        # Compute batch statistics
        if self.training and self.track_running_stats:
            # Update running statistics
            batch_mean = x.mean(dim=0, keepdim=True)
            batch_var = x.var(dim=0, keepdim=True, unbiased=False)

            n = self.num_batches_tracked
            if n == 0:
                self.running_mean = batch_mean.squeeze(0)
                self.running_var = batch_var.squeeze(0)
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                  self.momentum * batch_mean.squeeze(0)
                self.running_var = (1 - self.momentum) * self.running_var + \
                                 self.momentum * batch_var.squeeze(0)

            self.num_batches_tracked += 1
            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics
            if self.track_running_stats:
                mean = self.running_mean.unsqueeze(0)
                var = self.running_var.unsqueeze(0)
            else:
                mean = x.mean(dim=0, keepdim=True)
                var = x.var(dim=0, keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply speaker-specific affine transformation
        if speaker_ids is not None:
            # Training mode: use speaker-specific parameters
            scales = self.speaker_scales[speaker_ids]  # [batch, num_features]
            biases = self.speaker_biases[speaker_ids]  # [batch, num_features]
        else:
            # Inference mode: remove speaker conditioning
            if inference_mode == "average":
                # Use average of all speaker parameters
                scales = self.speaker_scales.mean(dim=0, keepdim=True).expand(x.size(0), -1)
                biases = self.speaker_biases.mean(dim=0, keepdim=True).expand(x.size(0), -1)
            elif inference_mode == "zero":
                # No conditioning (identity transformation)
                scales = torch.ones(x.size(0), self.num_features, device=x.device)
                biases = torch.zeros(x.size(0), self.num_features, device=x.device)
            elif inference_mode == "neutral":
                # Use first speaker as neutral
                scales = self.speaker_scales[0].unsqueeze(0).expand(x.size(0), -1)
                biases = self.speaker_biases[0].unsqueeze(0).expand(x.size(0), -1)
            else:
                raise ValueError(f"Unknown inference_mode: {inference_mode}")

        # Apply affine transformation
        out = scales * x_norm + biases

        return out


class SpeakerConditionedProjection(nn.Module):
    """
    Speaker-conditioned emotion projection with adaptive normalization.

    This replaces the standard EmotionProjection with speaker-aware version.
    """

    def __init__(
        self,
        emotion_dim: int,
        hidden_size: int,
        num_speakers: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_adaptive_norm: bool = True,
        inference_mode: str = "average"
    ):
        """
        Initialize speaker-conditioned projection.

        Args:
            emotion_dim: Input emotion dimension
            hidden_size: Output hidden size (LLM hidden size)
            num_speakers: Number of speakers
            num_layers: Number of projection layers
            dropout: Dropout rate
            use_adaptive_norm: Whether to use speaker adaptive normalization
            inference_mode: Inference mode for adaptive norm
        """
        super().__init__()

        self.emotion_dim = emotion_dim
        self.hidden_size = hidden_size
        self.num_speakers = num_speakers
        self.use_adaptive_norm = use_adaptive_norm
        self.inference_mode = inference_mode

        # Speaker adaptive normalization (applied first)
        if use_adaptive_norm:
            self.adaptive_norm = SpeakerAdaptiveNorm(
                num_features=emotion_dim,
                num_speakers=num_speakers
            )
        else:
            self.adaptive_norm = None

        # Standard projection layers
        layers = []
        current_dim = emotion_dim

        for i in range(num_layers):
            next_dim = hidden_size if i == num_layers - 1 else (emotion_dim + hidden_size) // 2
            layers.append(nn.Linear(current_dim, next_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            current_dim = next_dim

        self.projection = nn.Sequential(*layers)

    def forward(
        self,
        emotion_embeddings: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Project emotion embeddings with speaker conditioning.

        Args:
            emotion_embeddings: [batch_size, emotion_dim]
            speaker_ids: [batch_size] speaker IDs (optional)

        Returns:
            Projected embeddings: [batch_size, hidden_size]
        """
        x = emotion_embeddings

        # Apply speaker adaptive normalization
        if self.adaptive_norm is not None:
            x = self.adaptive_norm(x, speaker_ids, inference_mode=self.inference_mode)

        # Project to hidden size
        x = self.projection(x)

        return x


# Helper function to create speaker ID mapping
def create_speaker_id_mapping(speaker_names: list) -> Dict[str, int]:
    """
    Create mapping from speaker names to IDs.

    Args:
        speaker_names: List of unique speaker names

    Returns:
        Dictionary mapping speaker name to integer ID
    """
    return {name: idx for idx, name in enumerate(sorted(speaker_names))}


# Example usage
if __name__ == "__main__":
    # Test Speaker Adaptive Normalization
    batch_size = 4
    num_features = 256
    num_speakers = 4

    # Create layer
    san = SpeakerAdaptiveNorm(num_features=num_features, num_speakers=num_speakers)

    # Test with speaker IDs (training)
    x_train = torch.randn(batch_size, num_features)
    speaker_ids = torch.tensor([0, 1, 2, 3])  # Different speakers

    san.train()
    out_train = san(x_train, speaker_ids=speaker_ids)
    print(f"Training output shape: {out_train.shape}")
    print(f"Training output mean: {out_train.mean(dim=1)}")
    print(f"Training output std: {out_train.std(dim=1)}")

    # Test without speaker IDs (inference)
    x_test = torch.randn(batch_size, num_features)

    san.eval()
    out_test_avg = san(x_test, speaker_ids=None, inference_mode="average")
    out_test_zero = san(x_test, speaker_ids=None, inference_mode="zero")

    print(f"\nInference (average) output mean: {out_test_avg.mean(dim=1)}")
    print(f"Inference (zero) output mean: {out_test_zero.mean(dim=1)}")

    # Test Speaker-Conditioned Projection
    print("\n" + "="*50)
    print("Testing SpeakerConditionedProjection")
    print("="*50)

    proj = SpeakerConditionedProjection(
        emotion_dim=256,
        hidden_size=1536,
        num_speakers=4,
        use_adaptive_norm=True
    )

    # Training mode
    proj.train()
    emotion_embeds = torch.randn(batch_size, 256)
    speaker_ids = torch.tensor([0, 1, 2, 3])

    out_proj_train = proj(emotion_embeds, speaker_ids)
    print(f"Projection output shape: {out_proj_train.shape}")

    # Inference mode
    proj.eval()
    out_proj_test = proj(emotion_embeds, speaker_ids=None)
    print(f"Projection output (inference) shape: {out_proj_test.shape}")

    # Check that outputs differ between training and inference
    diff = (out_proj_train - out_proj_test).abs().mean()
    print(f"Difference between training and inference: {diff.item():.4f}")
