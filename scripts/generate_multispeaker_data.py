#!/usr/bin/env python3
"""
Generate multi-speaker training data for Experiment 11.
Creates a dataset with multiple speakers to enable speaker-invariant emotion learning.
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
from typing import List, Dict

from audio_augmented_llm.src.tts_pipeline.tts_engine import TTSEngine
from audio_augmented_llm.src.emotion_encoder.emotion_model import EmotionEncoder


def create_synthetic_samples(num_samples: int) -> List[Dict]:
    """
    Create synthetic dialogue samples.

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
            ("I got a promotion at work!", "That's excellent news! Your dedication really paid off!"),
            ("My best friend is coming to visit!", "How wonderful! You must be so excited to see them!"),
        ],
        "sadness": [
            ("I'm feeling really down today.", "I'm sorry to hear that. Do you want to talk about what's bothering you?"),
            ("I miss my family so much.", "That must be really hard. It's completely normal to feel that way."),
            ("Things haven't been going well lately.", "I understand. Sometimes life can be overwhelming."),
            ("I lost something very important to me.", "I'm so sorry. That must be difficult for you."),
            ("I feel lonely sometimes.", "Those feelings are valid. Remember you're not alone in this."),
        ],
        "anger": [
            ("This is so unfair! I can't believe this happened!", "I can see why you're upset. That does sound frustrating."),
            ("I'm really annoyed with this situation.", "That's understandable. It sounds like a difficult situation."),
            ("Why does this always happen to me?", "I hear your frustration. Let's think about how to address this."),
            ("This is completely unacceptable!", "I understand your anger. Let's work through this together."),
            ("I'm so fed up with everything!", "That sounds really frustrating. What can I do to help?"),
        ],
        "fear": [
            ("I'm really worried about the upcoming exam.", "It's natural to feel nervous. Have you been preparing well?"),
            ("What if something goes wrong?", "I understand your concerns. Let's think through this together."),
            ("I'm scared about the future.", "Those feelings are valid. Remember that you're not alone in this."),
            ("This is making me really anxious.", "I can understand that. Let's take it one step at a time."),
            ("I don't know if I can handle this.", "You're stronger than you think. We'll figure this out together."),
        ],
        "surprise": [
            ("I can't believe what just happened!", "Really? Tell me more! What happened?"),
            ("You won't believe this news!", "Oh wow! What is it? I'm so curious!"),
            ("Something unexpected just occurred!", "That's surprising! What was it?"),
            ("I never saw that coming!", "How unexpected! Tell me all about it!"),
            ("This is so unexpected!", "Wow! What a surprise! Share the details!"),
        ],
        "neutral": [
            ("How's your day going?", "It's going well, thanks for asking. How about yours?"),
            ("What are you working on today?", "I'm helping with various tasks. What can I do for you?"),
            ("Tell me about your plans.", "I'd be happy to discuss that. What would you like to know?"),
            ("I need some information.", "Of course, I'm here to help. What do you need to know?"),
            ("Can you help me with something?", "Absolutely. What do you need assistance with?"),
        ],
    }

    samples = []
    for i in range(num_samples):
        emotion = emotions[i % len(emotions)]
        template_list = templates[emotion]
        context, response = template_list[i % len(template_list)]

        samples.append({
            "id": f"synthetic_{i:05d}",
            "context": context,
            "teacher_response": response,
            "emotion": emotion,
            "dataset": "synthetic"
        })

    return samples


def generate_multispeaker_dataset(
    output_base_dir: str,
    speakers: List[str],
    samples_per_speaker: int,
    extract_embeddings: bool = True
):
    """
    Generate multi-speaker dataset.

    Args:
        output_base_dir: Base directory for output
        speakers: List of speaker names to use
        samples_per_speaker: Number of samples to generate per speaker
        extract_embeddings: Whether to extract emotion embeddings
    """
    print("=" * 80)
    print("Multi-Speaker Dataset Generation")
    print("=" * 80)
    print(f"Speakers: {', '.join(speakers)}")
    print(f"Samples per speaker: {samples_per_speaker}")
    print(f"Total samples: {len(speakers) * samples_per_speaker}")
    print()

    # Initialize TTS engine
    print("Initializing TTS engine...")
    tts_engine = TTSEngine(model_type="xtts")
    print("✓ TTS engine ready")

    # Initialize emotion encoder if needed
    emotion_encoder = None
    if extract_embeddings:
        print("\nInitializing emotion encoder...")
        emotion_encoder = EmotionEncoder(model_type="wavlm")
        print("✓ Emotion encoder ready")

    # Process each speaker
    all_samples = []

    for speaker in speakers:
        print(f"\n{'=' * 80}")
        print(f"Processing speaker: {speaker}")
        print(f"{'=' * 80}")

        # Create output directory for this speaker
        speaker_dir = Path(output_base_dir) / f"speaker_{speaker.replace(' ', '_').lower()}"
        audio_dir = speaker_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Generate synthetic samples
        print(f"\nGenerating {samples_per_speaker} synthetic samples...")
        samples = create_synthetic_samples(samples_per_speaker)

        # Generate audio for each sample
        print(f"\nGenerating audio with speaker: {speaker}")
        speaker_samples = []
        embeddings_dict = {}

        for sample in tqdm(samples, desc=f"Synthesizing"):
            text = sample["teacher_response"]
            sample_id = f"{speaker.replace(' ', '_').lower()}_{sample['id']}"
            audio_path = audio_dir / f"{sample_id}.wav"

            try:
                # Synthesize speech with specific speaker
                audio = tts_engine.synthesize(
                    text=text,
                    speaker=speaker,  # Use specific speaker name
                    output_path=str(audio_path)
                )

                # Extract emotion embedding if needed
                if extract_embeddings and emotion_encoder:
                    # Convert audio to torch tensor and add batch dimension
                    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                    # Get embedding
                    embedding, _ = emotion_encoder.forward(audio_tensor)
                    # Convert to numpy and remove batch dimension
                    embedding_np = embedding.detach().cpu().numpy()[0]
                    embeddings_dict[sample_id] = embedding_np

                # Create sample entry
                speaker_sample = {
                    "id": sample_id,
                    "context": sample["context"],
                    "teacher_response": sample["teacher_response"],
                    "emotion": sample["emotion"],
                    "dataset": "synthetic_multispeaker",
                    "speaker": speaker,  # Add speaker information
                    "audio_path": str(audio_path),
                    "has_embedding": extract_embeddings
                }
                speaker_samples.append(speaker_sample)
                all_samples.append(speaker_sample)

            except Exception as e:
                print(f"\n⚠ Failed to process sample {sample_id}: {e}")
                continue

        # Save speaker-specific metadata
        metadata_path = speaker_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(speaker_samples, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved metadata for {speaker}: {len(speaker_samples)} samples")

        # Save embeddings if extracted
        if embeddings_dict:
            embeddings_path = speaker_dir / "emotion_embeddings.npz"
            np.savez_compressed(embeddings_path, **embeddings_dict)
            print(f"✓ Saved embeddings for {speaker}: {len(embeddings_dict)} embeddings")

    print(f"\n{'=' * 80}")
    print("✅ Multi-speaker dataset generation complete!")
    print(f"{'=' * 80}")
    print(f"Total speakers: {len(speakers)}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Saved to: {output_base_dir}")

    return all_samples


def main():
    parser = argparse.ArgumentParser(description="Generate multi-speaker training data")
    parser.add_argument("--output_dir", type=str, default="./audio_augmented_llm/data/multispeaker",
                        help="Output directory for multi-speaker data")
    parser.add_argument("--speakers", type=str, nargs="+",
                        default=["Claribel Dervla", "Andrew Chipper", "Gracie Wise"],
                        help="List of speakers to use")
    parser.add_argument("--samples_per_speaker", type=int, default=250,
                        help="Number of samples to generate per speaker")
    parser.add_argument("--skip_embeddings", action="store_true",
                        help="Skip emotion embedding extraction")

    args = parser.parse_args()

    generate_multispeaker_dataset(
        output_base_dir=args.output_dir,
        speakers=args.speakers,
        samples_per_speaker=args.samples_per_speaker,
        extract_embeddings=not args.skip_embeddings
    )


if __name__ == "__main__":
    main()
