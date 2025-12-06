#!/usr/bin/env python3
"""
Generate test data for Andrew Chipper speaker.
This creates a held-out speaker test set to validate cross-speaker generalization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

from audio_augmented_llm.src.tts_pipeline.tts_engine import TTSEngine
from audio_augmented_llm.src.emotion_encoder.emotion_model import EmotionEncoder
import torch

def generate_andrew_test_set(num_samples=100):
    """
    Generate Andrew Chipper test set from existing text data.

    Strategy:
    - Sample 100 items from train_1000 metadata
    - Regenerate audio using Andrew Chipper voice
    - Extract WavLM embeddings
    - Save as test set for cross-speaker evaluation
    """

    print("=" * 80)
    print("Generating Andrew Chipper Test Set (Held-Out Speaker)")
    print("=" * 80)

    random.seed(42)

    # Load source data
    print("\n1. Loading source data...")
    source_dir = Path("audio_augmented_llm/data/train_1000")
    with open(source_dir / "metadata.json", 'r') as f:
        source_metadata = json.load(f)

    # Sample 100 items
    sampled_indices = random.sample(range(len(source_metadata)), num_samples)
    print(f"   ✓ Sampled {num_samples} items from train_1000")

    # Create output directory
    output_dir = Path("audio_augmented_llm/data/test_cross_speaker_andrew")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Initialize TTS engine
    print("\n2. Initializing TTS engine...")
    tts_engine = TTSEngine(model_type="xtts", model_path="tts_models/multilingual/multi-dataset/xtts_v2")
    print("   ✓ TTS engine loaded")

    # Initialize emotion encoder
    print("\n3. Initializing WavLM encoder...")
    encoder = EmotionEncoder(model_type="wavlm")
    encoder.eval()
    print("   ✓ WavLM encoder loaded")

    # Generate audio and extract embeddings
    print("\n4. Generating audio and extracting embeddings...")
    test_samples = []
    embeddings = {}

    with torch.no_grad():
        for idx in tqdm(sampled_indices, desc="   Processing"):
            source_sample = source_metadata[idx]
            sample_id = f"andrew_{idx:04d}"

            # Prepare text
            context = source_sample.get('context', '')
            response = source_sample.get('teacher_response', '')

            if context:
                full_text = f"{context} {response}"
            else:
                full_text = response

            # Generate audio with Andrew Chipper voice
            audio_path = audio_dir / f"{sample_id}.wav"

            try:
                audio = tts_engine.synthesize(
                    text=full_text,
                    speaker="Andrew Chipper",  # Use Andrew's voice
                    output_path=str(audio_path)
                )

                # Extract WavLM embedding
                import librosa
                audio_reloaded, sr = librosa.load(audio_path, sr=16000)
                audio_tensor = torch.from_numpy(audio_reloaded).float().unsqueeze(0)
                embedding, _ = encoder.forward(audio_tensor)
                embedding_np = embedding.detach().cpu().numpy()[0]

                # Save metadata
                test_samples.append({
                    "id": sample_id,
                    "text": full_text,
                    "context": context,
                    "teacher_response": response,
                    "emotion": source_sample['emotion'],
                    "dataset": "test_cross_speaker_andrew",
                    "speaker": "Andrew Chipper",
                    "speaker_id": 2,  # Different from Claribel (0) and Damien (1)
                    "audio_path": str(audio_path),
                    "has_embedding": True
                })

                embeddings[sample_id] = embedding_np

            except Exception as e:
                print(f"\n   ⚠ Failed to process {sample_id}: {e}")
                continue

    # Save metadata
    print("\n5. Saving metadata...")
    with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved {len(test_samples)} samples")

    # Save embeddings
    print("\n6. Saving WavLM embeddings...")
    np.savez_compressed(output_dir / "emotion_embeddings.npz", **embeddings)
    print(f"   ✓ Saved {len(embeddings)} embeddings")

    print("\n" + "=" * 80)
    print("✅ Andrew Chipper Test Set Generated Successfully!")
    print("=" * 80)
    print(f"\n📊 Dataset Statistics:")
    print(f"   Samples: {len(test_samples)}")
    print(f"   Speaker: Andrew Chipper (speaker_id=2)")
    print(f"   Purpose: Held-out speaker for cross-speaker generalization test")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n🎯 Next Step:")
    print(f"   Test Exp 11 model (trained on Claribel + Damien)")
    print(f"   on Andrew Chipper (completely unseen speaker)")

    return output_dir


if __name__ == "__main__":
    generate_andrew_test_set(num_samples=100)
