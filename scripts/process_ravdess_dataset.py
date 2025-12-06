#!/usr/bin/env python3
"""
Process RAVDESS dataset for emotion-aware LLM training.

RAVDESS file naming convention:
    Modality-VocalChannel-Emotion-EmotionIntensity-Statement-Repetition-Actor.wav

Example: 03-01-06-01-02-01-12.wav
- Modality: 01=Audio-only, 02=Video-only, 03=Audio-Video
- VocalChannel: 01=Speech, 02=Song
- Emotion: 01=Neutral, 02=Calm, 03=Happy, 04=Sad, 05=Angry, 06=Fearful, 07=Disgust, 08=Surprised
- EmotionIntensity: 01=Normal, 02=Strong
- Statement: 01="Kids are talking by the door", 02="Dogs are sitting by the door"
- Repetition: 01=1st, 02=2nd
- Actor: 01-24 (odd=male, even=female)
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import random

# Emotion mapping
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Statement text mapping
STATEMENT_MAP = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door"
}


def parse_ravdess_filename(filename: str) -> Dict:
    """Parse RAVDESS filename to extract metadata."""
    stem = Path(filename).stem
    parts = stem.split('-')

    if len(parts) != 7:
        raise ValueError(f"Invalid RAVDESS filename: {filename}")

    modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts

    return {
        "modality": modality,
        "vocal_channel": vocal_channel,
        "emotion": EMOTION_MAP[emotion],
        "emotion_code": emotion,
        "intensity": "normal" if intensity == "01" else "strong",
        "statement": STATEMENT_MAP[statement],
        "statement_code": statement,
        "repetition": int(repetition),
        "actor": int(actor),
        "gender": "male" if int(actor) % 2 == 1 else "female",
        "audio_path": filename
    }


def organize_dataset(
    ravdess_dir: Path,
    output_dir: Path,
    train_actors: List[int],
    val_actors: List[int],
    test_actors: List[int]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Organize RAVDESS dataset into train/val/test splits based on actors.

    Args:
        ravdess_dir: Path to extracted RAVDESS dataset
        output_dir: Output directory for organized dataset
        train_actors: List of actor IDs for training
        val_actors: List of actor IDs for validation
        test_actors: List of actor IDs for test

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    # Find all audio files
    audio_files = []
    for actor_dir in ravdess_dir.glob("Actor_*"):
        if actor_dir.is_dir():
            audio_files.extend(actor_dir.glob("*.wav"))

    print(f"Found {len(audio_files)} audio files")

    # Parse and organize
    train_samples = []
    val_samples = []
    test_samples = []

    for audio_file in audio_files:
        try:
            metadata = parse_ravdess_filename(audio_file.name)
            metadata["audio_path"] = str(audio_file.absolute())

            actor = metadata["actor"]

            if actor in train_actors:
                train_samples.append(metadata)
            elif actor in val_actors:
                val_samples.append(metadata)
            elif actor in test_actors:
                test_samples.append(metadata)

        except Exception as e:
            print(f"Warning: Failed to parse {audio_file.name}: {e}")

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy files to organized structure
    def copy_samples(samples, target_dir):
        for idx, sample in enumerate(samples):
            src = Path(sample["audio_path"])
            dst = target_dir / f"{idx:04d}_{sample['emotion']}_{sample['actor']:02d}.wav"
            shutil.copy2(src, dst)
            sample["audio_path"] = str(dst.relative_to(target_dir))

    copy_samples(train_samples, train_dir)
    copy_samples(val_samples, val_dir)
    copy_samples(test_samples, test_dir)

    # Save metadata
    def save_metadata(samples, split_dir, split_name):
        metadata_file = split_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(samples, f, indent=2)

        # Create summary
        emotion_counts = {}
        for sample in samples:
            emotion = sample["emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        actor_ids = sorted(set(s["actor"] for s in samples))

        summary = {
            "split": split_name,
            "num_samples": len(samples),
            "num_actors": len(actor_ids),
            "actor_ids": actor_ids,
            "emotion_distribution": emotion_counts
        }

        summary_file = split_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{split_name.upper()} Split:")
        print(f"  Samples: {len(samples)}")
        print(f"  Actors: {len(actor_ids)} - {actor_ids}")
        print(f"  Emotions: {emotion_counts}")

    save_metadata(train_samples, train_dir, "train")
    save_metadata(val_samples, val_dir, "val")
    save_metadata(test_samples, test_dir, "test")

    return train_samples, val_samples, test_samples


def main():
    parser = argparse.ArgumentParser(description="Process RAVDESS dataset")
    parser.add_argument(
        "--ravdess_dir",
        type=str,
        required=True,
        help="Path to extracted RAVDESS dataset (Audio_Speech_Actors_01-24)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./audio_augmented_llm/data/ravdess_processed",
        help="Output directory for organized dataset"
    )
    parser.add_argument(
        "--train_actors",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16",
        help="Comma-separated actor IDs for training (default: 1-16)"
    )
    parser.add_argument(
        "--val_actors",
        type=str,
        default="17,18,19,20",
        help="Comma-separated actor IDs for validation (default: 17-20)"
    )
    parser.add_argument(
        "--test_actors",
        type=str,
        default="21,22,23,24",
        help="Comma-separated actor IDs for test (default: 21-24)"
    )

    args = parser.parse_args()

    ravdess_dir = Path(args.ravdess_dir)
    output_dir = Path(args.output_dir)

    if not ravdess_dir.exists():
        raise ValueError(f"RAVDESS directory not found: {ravdess_dir}")

    # Parse actor lists
    train_actors = [int(x.strip()) for x in args.train_actors.split(',')]
    val_actors = [int(x.strip()) for x in args.val_actors.split(',')]
    test_actors = [int(x.strip()) for x in args.test_actors.split(',')]

    print(f"Processing RAVDESS dataset...")
    print(f"  Source: {ravdess_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Train actors: {train_actors}")
    print(f"  Val actors: {val_actors}")
    print(f"  Test actors: {test_actors}")

    # Organize dataset
    train, val, test = organize_dataset(
        ravdess_dir,
        output_dir,
        train_actors,
        val_actors,
        test_actors
    )

    print(f"\n✅ Dataset processing complete!")
    print(f"Total samples: {len(train) + len(val) + len(test)}")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")


if __name__ == "__main__":
    main()
