#!/usr/bin/env python3
"""
Generate teacher training data with audio and emotion embeddings.
Creates a dataset with: input text → teacher response → TTS audio → emotion embedding
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Optional
from datasets import load_dataset

# Import our modules
from audio_augmented_llm.src.tts_pipeline.tts_engine import TTSEngine
from audio_augmented_llm.src.emotion_encoder.emotion_model import EmotionEncoder


def create_synthetic_samples(num_samples: int) -> List[Dict]:
    """
    Create synthetic dialogue samples for prototyping when datasets are unavailable.

    Args:
        num_samples: Number of samples to create

    Returns:
        List of synthetic dialogue samples
    """
    emotions = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]

    # Template dialogues for different emotions
    templates = {
        "joy": [
            ("I just got accepted to my dream university!", "Congratulations! That's absolutely wonderful news! I'm so happy for you!"),
            ("My team won the championship!", "That's amazing! You must be thrilled! All your hard work paid off!"),
            ("I finally finished my big project!", "Fantastic! You should be so proud of yourself!"),
        ],
        "sadness": [
            ("I'm feeling really down today.", "I'm sorry to hear that. Do you want to talk about what's bothering you?"),
            ("I miss my family so much.", "That must be really hard. It's completely normal to feel that way."),
            ("Things haven't been going well lately.", "I understand. Sometimes life can be overwhelming."),
        ],
        "anger": [
            ("This is so unfair! I can't believe this happened!", "I can see why you're upset. That does sound frustrating."),
            ("I'm really annoyed with this situation.", "That's understandable. It sounds like a difficult situation."),
            ("Why does this always happen to me?", "I hear your frustration. Let's think about how to address this."),
        ],
        "fear": [
            ("I'm really worried about the upcoming exam.", "It's natural to feel nervous. Have you been preparing well?"),
            ("What if something goes wrong?", "I understand your concerns. Let's think through this together."),
            ("I'm scared about the future.", "Those feelings are valid. Remember that you're not alone in this."),
        ],
        "surprise": [
            ("I can't believe what just happened!", "Really? Tell me more! What happened?"),
            ("You won't believe what I just found out!", "I'm all ears! This sounds exciting!"),
            ("Guess what just happened?", "I'm intrigued! Please tell me!"),
        ],
        "neutral": [
            ("What do you think about this topic?", "That's an interesting question. Let me share my perspective on this."),
            ("Can you explain how this works?", "Of course. Let me break it down step by step for you."),
            ("I need some information about this.", "I'd be happy to help. What specifically would you like to know?"),
        ],
    }

    samples = []
    for i in range(num_samples):
        emotion = emotions[i % len(emotions)]
        template_idx = (i // len(emotions)) % len(templates[emotion])
        context, response = templates[emotion][template_idx]

        samples.append({
            "id": f"synthetic_{i:05d}",
            "context": context,
            "response": response,
            "emotion": emotion,
            "dataset": "synthetic"
        })

    print(f"✓ Created {len(samples)} synthetic samples")
    return samples


def load_dialogue_samples(dataset_name: str = "empathetic_dialogues", num_samples: int = 100, split: str = "train"):
    """
    Load dialogue samples from HuggingFace datasets.

    Args:
        dataset_name: Name of the dataset to load
        num_samples: Number of samples to generate
        split: Dataset split to use

    Returns:
        List of dialogue samples with context and response
    """
    print(f"\nLoading {num_samples} samples from {dataset_name}...")

    try:
        if dataset_name == "empathetic_dialogues":
            # Try loading with trust_remote_code or use alternative approach
            try:
                dataset = load_dataset("empathetic_dialogues", split=split, trust_remote_code=True)
            except Exception:
                # Alternative: Use a pre-processed version or create synthetic data
                print("⚠ Could not load empathetic_dialogues, creating synthetic samples...")
                return create_synthetic_samples(num_samples)

            samples = []
            for i, item in enumerate(dataset):
                if len(samples) >= num_samples:
                    break

                # Extract context and response
                context = item.get("prompt", "") or item.get("context", "")
                response = item.get("utterance", "") or item.get("response", "")
                emotion = item.get("context", "neutral")  # emotion label if available

                if context and response:
                    samples.append({
                        "id": f"{dataset_name}_{i:05d}",
                        "context": context,
                        "response": response,
                        "emotion": emotion if isinstance(emotion, str) else "neutral",
                        "dataset": dataset_name
                    })

            print(f"✓ Loaded {len(samples)} samples from {dataset_name}")
            return samples

        elif dataset_name == "daily_dialog":
            # DailyDialog has multi-turn conversations
            dataset = load_dataset("daily_dialog", split=split)

            samples = []
            for conv_idx, conversation in enumerate(dataset):
                if len(samples) >= num_samples:
                    break

                dialog = conversation["dialog"]
                emotions = conversation.get("emotion", [0] * len(dialog))  # 0 = no emotion

                # Extract context-response pairs
                for i in range(1, len(dialog)):
                    if len(samples) >= num_samples:
                        break

                    context = dialog[i-1]
                    response = dialog[i]
                    emotion_label = emotions[i] if i < len(emotions) else 0

                    # Map emotion index to name
                    emotion_map = {0: "neutral", 1: "anger", 2: "disgust", 3: "fear",
                                   4: "joy", 5: "sadness", 6: "surprise"}
                    emotion = emotion_map.get(emotion_label, "neutral")

                    samples.append({
                        "id": f"{dataset_name}_{conv_idx:05d}_{i:03d}",
                        "context": context,
                        "response": response,
                        "emotion": emotion,
                        "dataset": dataset_name
                    })

            print(f"✓ Loaded {len(samples)} samples from {dataset_name}")
            return samples

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return []


def generate_teacher_responses(samples: List[Dict], use_teacher_model: bool = False):
    """
    Generate teacher responses (or use existing responses from dataset).

    For prototyping, we'll use the existing responses from the dataset.
    For production, you can load a teacher model here.

    Args:
        samples: List of dialogue samples
        use_teacher_model: Whether to generate responses with a teacher LLM

    Returns:
        Samples with teacher responses
    """
    print(f"\n{'Generating' if use_teacher_model else 'Using existing'} teacher responses...")

    if not use_teacher_model:
        # Use existing responses from dataset
        for sample in samples:
            sample["teacher_response"] = sample["response"]
        print(f"✓ Using existing responses from dataset")
        return samples

    else:
        # TODO: Load teacher model and generate responses
        # This would be for more advanced experiments
        print("⚠ Teacher model generation not implemented yet. Using existing responses.")
        for sample in samples:
            sample["teacher_response"] = sample["response"]
        return samples


def synthesize_audio(samples: List[Dict], tts_engine: TTSEngine, output_dir: Path):
    """
    Synthesize audio for teacher responses using TTS.

    Args:
        samples: List of samples with teacher responses
        tts_engine: TTS engine instance
        output_dir: Directory to save audio files

    Returns:
        Samples with audio file paths
    """
    print(f"\nSynthesizing audio for {len(samples)} samples...")

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    for sample in tqdm(samples, desc="TTS synthesis"):
        audio_path = audio_dir / f"{sample['id']}.wav"

        try:
            tts_engine.synthesize(
                text=sample["teacher_response"],
                output_path=str(audio_path)
            )
            sample["audio_path"] = str(audio_path)
            successful += 1
        except Exception as e:
            print(f"\n✗ Failed to synthesize {sample['id']}: {e}")
            sample["audio_path"] = None

    print(f"✓ Successfully synthesized {successful}/{len(samples)} audio files")
    return samples


def extract_emotion_embeddings(samples: List[Dict], emotion_encoder: EmotionEncoder, device: str):
    """
    Extract emotion embeddings from synthesized audio.

    Args:
        samples: List of samples with audio paths
        emotion_encoder: Emotion encoder instance
        device: Device to run inference on

    Returns:
        Samples with emotion embeddings
    """
    print(f"\nExtracting emotion embeddings...")

    emotion_encoder.eval()
    successful = 0

    for sample in tqdm(samples, desc="Emotion encoding"):
        if sample.get("audio_path") is None:
            sample["emotion_embedding"] = None
            continue

        try:
            with torch.no_grad():
                embedding = emotion_encoder.encode_from_file(
                    sample["audio_path"],
                    device=device
                )
            sample["emotion_embedding"] = embedding.cpu().numpy().tolist()
            successful += 1
        except Exception as e:
            print(f"\n✗ Failed to encode {sample['id']}: {e}")
            sample["emotion_embedding"] = None

    print(f"✓ Successfully extracted {successful}/{len(samples)} emotion embeddings")
    return samples


def save_dataset(samples: List[Dict], output_dir: Path):
    """
    Save the generated dataset to disk.

    Args:
        samples: List of samples with all fields
        output_dir: Directory to save dataset
    """
    print(f"\nSaving dataset to {output_dir}...")

    # Save metadata (without embeddings, which are large)
    metadata = []
    for sample in samples:
        metadata.append({
            "id": sample["id"],
            "context": sample["context"],
            "teacher_response": sample["teacher_response"],
            "emotion": sample["emotion"],
            "dataset": sample["dataset"],
            "audio_path": sample.get("audio_path"),
            "has_embedding": sample.get("emotion_embedding") is not None
        })

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved metadata to {metadata_path}")

    # Save emotion embeddings separately (as numpy arrays for efficiency)
    embeddings = {}
    for sample in samples:
        if sample.get("emotion_embedding") is not None:
            embeddings[sample["id"]] = sample["emotion_embedding"]

    embeddings_path = output_dir / "emotion_embeddings.npz"
    np.savez_compressed(embeddings_path, **embeddings)
    print(f"✓ Saved emotion embeddings to {embeddings_path}")

    # Save full dataset as JSON (for inspection)
    full_data_path = output_dir / "full_dataset.json"
    with open(full_data_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved full dataset to {full_data_path}")

    print(f"\n✅ Dataset generation complete!")
    print(f"   Total samples: {len(samples)}")
    print(f"   With audio: {sum(1 for s in samples if s.get('audio_path') is not None)}")
    print(f"   With embeddings: {len(embeddings)}")


def main():
    parser = argparse.ArgumentParser(description="Generate teacher training data")
    parser.add_argument("--dataset", type=str, default="empathetic_dialogues",
                        choices=["empathetic_dialogues", "daily_dialog"],
                        help="Source dialogue dataset")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="./audio_augmented_llm/data/train_100",
                        help="Output directory for generated data")
    parser.add_argument("--use_teacher_model", action="store_true",
                        help="Generate responses with teacher model (not implemented yet)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (default: cuda if available)")

    args = parser.parse_args()

    # Set device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Audio-Augmented LLM: Teacher Data Generation")
    print("=" * 70)

    # Step 1: Load dialogue samples
    samples = load_dialogue_samples(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        split="train"
    )

    if not samples:
        print("✗ No samples loaded. Exiting.")
        return 1

    # Step 2: Generate/use teacher responses
    samples = generate_teacher_responses(samples, args.use_teacher_model)

    # Step 3: Initialize TTS engine
    print("\nInitializing TTS engine...")
    try:
        tts_engine = TTSEngine(
            model_type="xtts",
            model_path="tts_models/multilingual/multi-dataset/xtts_v2",
            language="en"
        )
        print("✓ TTS engine loaded")
    except Exception as e:
        print(f"✗ Failed to load TTS engine: {e}")
        return 1

    # Step 4: Synthesize audio
    samples = synthesize_audio(samples, tts_engine, output_dir)

    # Step 5: Initialize emotion encoder
    print("\nInitializing emotion encoder...")
    try:
        emotion_encoder = EmotionEncoder(
            model_type="wavlm",
            model_name="microsoft/wavlm-base-plus",
            embedding_dim=256,
            num_emotion_classes=7,
            freeze_encoder=True
        )
        emotion_encoder = emotion_encoder.to(device)
        print("✓ Emotion encoder loaded")
    except Exception as e:
        print(f"✗ Failed to load emotion encoder: {e}")
        return 1

    # Step 6: Extract emotion embeddings
    samples = extract_emotion_embeddings(samples, emotion_encoder, device)

    # Step 7: Save dataset
    save_dataset(samples, output_dir)

    print("\n" + "=" * 70)
    print("✅ Teacher data generation completed successfully!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
