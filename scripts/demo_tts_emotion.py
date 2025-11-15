#!/usr/bin/env python3
"""
Demo script to test TTS synthesis and emotion encoding.
This is a proof-of-concept to verify the basic pipeline works.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from audio_augmented_llm.src.tts_pipeline.tts_engine import TTSEngine
from audio_augmented_llm.src.emotion_encoder.emotion_model import EmotionEncoder


def create_demo_samples():
    """Create demo text samples with different emotions."""
    samples = [
        {
            "text": "I'm so happy and excited to help you with this wonderful project!",
            "emotion": "joy",
            "id": "sample_001"
        },
        {
            "text": "This is really frustrating and concerning. I'm not sure what to do.",
            "emotion": "anger",
            "id": "sample_002"
        },
        {
            "text": "I'm deeply sorry for your loss. My heart goes out to you.",
            "emotion": "sadness",
            "id": "sample_003"
        },
        {
            "text": "Let me explain the solution step by step in a calm manner.",
            "emotion": "neutral",
            "id": "sample_004"
        },
        {
            "text": "Oh no! That's terrifying! What should we do?",
            "emotion": "fear",
            "id": "sample_005"
        },
    ]
    return samples


def test_tts_synthesis(output_dir="./audio_augmented_llm/outputs/samples"):
    """Test TTS synthesis on demo samples."""
    print("=" * 60)
    print("Step 1: Testing TTS Synthesis")
    print("=" * 60)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TTS engine
    print("\nInitializing TTS engine (XTTS v2)...")
    try:
        tts = TTSEngine(
            model_type="xtts",
            model_path="tts_models/multilingual/multi-dataset/xtts_v2",
            language="en"
        )
        print("✓ TTS engine loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load TTS: {e}")
        return None, None

    # Get demo samples
    samples = create_demo_samples()

    # Synthesize audio for each sample
    audio_paths = []
    print(f"\nSynthesizing {len(samples)} audio samples...")

    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample['emotion']}: {sample['text'][:50]}...")

        audio_path = output_dir / f"{sample['id']}_{sample['emotion']}.wav"

        try:
            audio = tts.synthesize(
                text=sample['text'],
                output_path=str(audio_path)
            )
            audio_paths.append(str(audio_path))
            print(f"      ✓ Saved to: {audio_path}")
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            audio_paths.append(None)

    print(f"\n✓ Synthesized {len([p for p in audio_paths if p])} audio files")

    return samples, audio_paths


def test_emotion_encoder(samples, audio_paths):
    """Test emotion encoder on synthesized audio."""
    print("\n" + "=" * 60)
    print("Step 2: Testing Emotion Encoder")
    print("=" * 60)

    # Initialize emotion encoder
    print("\nInitializing emotion encoder (WavLM)...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        emotion_encoder = EmotionEncoder(
            model_type="wavlm",
            model_name="microsoft/wavlm-base-plus",  # Use base for faster loading
            embedding_dim=256,
            num_emotion_classes=7,
            freeze_encoder=True  # Freeze for demo
        )
        emotion_encoder = emotion_encoder.to(device)
        emotion_encoder.eval()

        print("✓ Emotion encoder loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load emotion encoder: {e}")
        return None

    # Extract emotion embeddings
    print(f"\nExtracting emotion embeddings...")
    embeddings = []

    for i, (sample, audio_path) in enumerate(zip(samples, audio_paths)):
        if audio_path is None:
            embeddings.append(None)
            continue

        print(f"  [{i+1}/{len(samples)}] {sample['emotion']}: Encoding...")

        try:
            with torch.no_grad():
                embedding = emotion_encoder.encode_from_file(audio_path, device=device)
            embeddings.append(embedding.cpu().numpy())
            print(f"      ✓ Embedding shape: {embedding.shape}")
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            embeddings.append(None)

    print(f"\n✓ Extracted {len([e for e in embeddings if e is not None])} embeddings")

    return embeddings


def visualize_embeddings(samples, embeddings, output_path="./audio_augmented_llm/outputs/emotion_embeddings.png"):
    """Visualize emotion embeddings."""
    print("\n" + "=" * 60)
    print("Step 3: Visualizing Emotion Embeddings")
    print("=" * 60)

    # Filter valid embeddings
    valid_samples = []
    valid_embeddings = []

    for sample, emb in zip(samples, embeddings):
        if emb is not None:
            valid_samples.append(sample)
            valid_embeddings.append(emb[0])  # Remove batch dimension

    if len(valid_embeddings) == 0:
        print("✗ No valid embeddings to visualize")
        return

    # Convert to numpy array
    embeddings_array = np.array(valid_embeddings)  # [N, 256]

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. First 20 dimensions
    ax = axes[0, 0]
    for i, (sample, emb) in enumerate(zip(valid_samples, valid_embeddings)):
        ax.plot(emb[:20], marker='o', label=f"{sample['emotion']}", alpha=0.7)
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Value')
    ax.set_title('Emotion Embeddings (First 20 Dimensions)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Heatmap of all embeddings
    ax = axes[0, 1]
    im = ax.imshow(embeddings_array.T, aspect='auto', cmap='viridis')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Embedding Dimension')
    ax.set_title('Emotion Embeddings Heatmap')
    ax.set_yticks([0, 64, 128, 192, 255])
    ax.set_xticks(range(len(valid_samples)))
    ax.set_xticklabels([s['emotion'] for s in valid_samples], rotation=45)
    plt.colorbar(im, ax=ax)

    # 3. Pairwise cosine similarity
    ax = axes[1, 0]
    # Normalize embeddings
    embeddings_norm = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    # Compute similarity matrix
    similarity = np.dot(embeddings_norm, embeddings_norm.T)

    im = ax.imshow(similarity, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title('Pairwise Cosine Similarity')
    ax.set_xticks(range(len(valid_samples)))
    ax.set_yticks(range(len(valid_samples)))
    ax.set_xticklabels([s['emotion'] for s in valid_samples], rotation=45)
    ax.set_yticklabels([s['emotion'] for s in valid_samples])

    # Add text annotations
    for i in range(len(valid_samples)):
        for j in range(len(valid_samples)):
            text = ax.text(j, i, f'{similarity[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax)

    # 4. Embedding statistics
    ax = axes[1, 1]
    means = embeddings_array.mean(axis=1)
    stds = embeddings_array.std(axis=1)

    x = range(len(valid_samples))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Mean ± Std')
    ax.set_title('Embedding Statistics')
    ax.set_xticks(x)
    ax.set_xticklabels([s['emotion'] for s in valid_samples], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")

    # Print similarity analysis
    print("\nEmotion Similarity Analysis:")
    print("-" * 60)
    for i, sample_i in enumerate(valid_samples):
        print(f"\n{sample_i['emotion'].upper()}:")
        similarities = [(valid_samples[j]['emotion'], similarity[i, j])
                       for j in range(len(valid_samples)) if i != j]
        similarities.sort(key=lambda x: x[1], reverse=True)
        for emotion, sim in similarities:
            print(f"  vs {emotion:10s}: {sim:.3f}")


def main():
    """Run the demo pipeline."""
    print("\n" + "=" * 60)
    print("Audio-Augmented LLM: Proof of Concept Demo")
    print("=" * 60)

    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Step 1: Test TTS
    samples, audio_paths = test_tts_synthesis()
    if samples is None:
        print("\n✗ TTS synthesis failed. Exiting.")
        return 1

    # Step 2: Test emotion encoder
    embeddings = test_emotion_encoder(samples, audio_paths)
    if embeddings is None:
        print("\n✗ Emotion encoding failed. Exiting.")
        return 1

    # Step 3: Visualize
    visualize_embeddings(samples, embeddings)

    print("\n" + "=" * 60)
    print("✓ Demo completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - Audio samples: ./audio_augmented_llm/outputs/samples/")
    print("  - Visualization: ./audio_augmented_llm/outputs/emotion_embeddings.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
