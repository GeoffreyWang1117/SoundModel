#!/usr/bin/env python3
"""
Extract WavLM embeddings from RAVDESS audio files.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from tqdm import tqdm


def load_wav_lm_model(device="cuda"):
    """Load WavLM model and feature extractor."""
    print("Loading WavLM model...")
    model_name = "microsoft/wavlm-base-plus"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name).to(device)
    model.eval()
    return feature_extractor, model


def extract_emotion_embedding(
    audio_path: str,
    feature_extractor,
    model,
    target_length: int = 16000 * 3,
    device="cuda"
) -> np.ndarray:
    """
    Extract emotion embedding from audio file using WavLM.

    Args:
        audio_path: Path to audio file
        feature_extractor: WavLM feature extractor
        model: WavLM model
        target_length: Target audio length in samples (default: 3 seconds at 16kHz)
        device: Device to use

    Returns:
        256-dimensional emotion embedding
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Pad or trim to target length
    if waveform.shape[1] < target_length:
        waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))
    else:
        waveform = waveform[:, :target_length]

    # Extract features
    inputs = feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )

    # Get embeddings
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        # Use mean pooling of last hidden state
        embeddings = outputs.last_hidden_state.mean(dim=1)  # [1, 768]

        # Project to 256D (using learned projection would be better, but for now we'll use PCA later)
        # For now, just return the full 768D embedding
        return embeddings.cpu().numpy()[0]  # [768]


def process_split(split_dir: Path, feature_extractor, model, device="cuda"):
    """Process all audio files in a split directory."""
    # Load metadata
    metadata_file = split_dir / "metadata.json"
    with open(metadata_file, 'r') as f:
        samples = json.load(f)

    print(f"\nProcessing {split_dir.name} split ({len(samples)} samples)...")

    embeddings_list = []
    texts_list = []

    for sample in tqdm(samples, desc=f"Extracting {split_dir.name}"):
        audio_path = split_dir / sample["audio_path"]

        # Extract embedding
        embedding = extract_emotion_embedding(
            str(audio_path),
            feature_extractor,
            model,
            device=device
        )

        # Get text with emotion annotation
        emotion = sample["emotion"]
        statement = sample["statement"]
        text = f"[{emotion.upper()}] {statement}"

        embeddings_list.append(embedding)
        texts_list.append(text)

    # Convert to numpy arrays
    embeddings = np.array(embeddings_list)  # [N, 768]
    texts = np.array(texts_list)

    # Save embeddings and texts
    np.savez(
        split_dir / "emotion_embeddings.npz",
        embeddings=embeddings
    )

    np.save(
        split_dir / "texts.npy",
        texts
    )

    print(f"✅ Saved {split_dir.name}: {embeddings.shape[0]} samples, embedding dim: {embeddings.shape[1]}")

    return embeddings, texts


def main():
    parser = argparse.ArgumentParser(description="Extract WavLM embeddings from RAVDESS")
    parser.add_argument(
        "--ravdess_processed_dir",
        type=str,
        required=True,
        help="Path to processed RAVDESS dataset directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    ravdess_dir = Path(args.ravdess_processed_dir)

    if not ravdess_dir.exists():
        raise ValueError(f"RAVDESS processed directory not found: {ravdess_dir}")

    # Load WavLM model
    feature_extractor, model = load_wav_lm_model(device=args.device)

    # Process each split
    for split_name in ["train", "val", "test"]:
        split_dir = ravdess_dir / split_name
        if split_dir.exists():
            process_split(split_dir, feature_extractor, model, device=args.device)

    print("\n✅ WavLM embedding extraction complete!")


if __name__ == "__main__":
    main()
