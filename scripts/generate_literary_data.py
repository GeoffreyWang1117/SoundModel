#!/usr/bin/env python3
"""
Generate training data from literary/dramatic texts with rich emotional content.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from audio_augmented_llm.src.tts_pipeline.tts_engine import TTSEngine
from audio_augmented_llm.src.emotion_encoder.emotion_model import EmotionEncoder
import argparse

# Emotion-rich literary excerpts (as fallback if datasets don't work)
LITERARY_EXAMPLES = [
    {
        "text": "She burst into tears of joy, her heart overflowing with happiness she thought she'd never feel again.",
        "emotion": "joy",
        "source": "novel_excerpt"
    },
    {
        "text": "The darkness pressed in around him, suffocating, terrifying. His hands trembled as he fumbled for the light switch.",
        "emotion": "fear",
        "source": "thriller_excerpt"
    },
    {
        "text": "How dare they! His face flushed red with rage, fists clenched so tight his knuckles turned white.",
        "emotion": "anger",
        "source": "drama_excerpt"
    },
    {
        "text": "She sat alone in the empty house, tears streaming down her face, mourning all that was lost.",
        "emotion": "sadness",
        "source": "literary_fiction"
    },
    {
        "text": "His eyes widened in amazement. Never in his wildest dreams had he imagined such beauty could exist.",
        "emotion": "surprise",
        "source": "adventure_novel"
    },
    {
        "text": "The old man smiled gently, his weathered face peaceful as he watched the sunset one last time.",
        "emotion": "neutral",
        "source": "literary_fiction"
    },
]

def load_tinystories(num_samples=100):
    """Load TinyStories dataset - short stories with emotional content."""
    print("\n📚 Loading TinyStories dataset...")
    try:
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

        samples = []
        for i, example in enumerate(tqdm(dataset, total=num_samples, desc="Loading stories")):
            if i >= num_samples:
                break

            text = example['text']
            # Take first few sentences
            sentences = text.split('. ')[:3]
            snippet = '. '.join(sentences) + '.'

            if len(snippet) > 50 and len(snippet) < 300:
                samples.append({
                    'text': snippet,
                    'source': 'tinystories',
                    'emotion': 'inferred'  # Will be inferred from audio
                })

        print(f"✓ Loaded {len(samples)} story snippets")
        return samples

    except Exception as e:
        print(f"❌ Error loading TinyStories: {e}")
        print("📝 Using fallback literary examples...")
        return LITERARY_EXAMPLES * (num_samples // len(LITERARY_EXAMPLES) + 1)

def load_daily_dialog(num_samples=100):
    """Load DailyDialog dataset - conversational dialogues."""
    print("\n💬 Loading DailyDialog dataset...")
    try:
        dataset = load_dataset("daily_dialog", split="train")

        samples = []
        for i, example in enumerate(tqdm(dataset, total=min(num_samples, len(dataset)), desc="Loading dialogs")):
            if len(samples) >= num_samples:
                break

            dialog = example['dialog']
            for utterance in dialog[:2]:  # Take first 2 turns
                if len(utterance) > 20 and len(utterance) < 200:
                    samples.append({
                        'text': utterance,
                        'source': 'daily_dialog',
                        'emotion': 'conversational'
                    })
                    if len(samples) >= num_samples:
                        break

        print(f"✓ Loaded {len(samples)} dialog utterances")
        return samples

    except Exception as e:
        print(f"❌ Error loading DailyDialog: {e}")
        print("📝 Using fallback literary examples...")
        return LITERARY_EXAMPLES * (num_samples // len(LITERARY_EXAMPLES) + 1)

def generate_literary_dataset(num_samples=500, output_dir="./audio_augmented_llm/data/literary_500", dataset_type="tinystories"):
    """Generate dataset from literary texts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    audio_dir = output_path / "audio"
    audio_dir.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print(f"Generating Literary Dataset ({num_samples} samples)")
    print("="*80)

    # Load literary texts
    if dataset_type == "tinystories":
        texts = load_tinystories(num_samples)
    elif dataset_type == "daily_dialog":
        texts = load_daily_dialog(num_samples)
    else:
        print(f"📝 Using curated literary examples...")
        texts = LITERARY_EXAMPLES * (num_samples // len(LITERARY_EXAMPLES) + 1)

    texts = texts[:num_samples]

    # Initialize TTS and emotion encoder
    print("\n🎤 Initializing TTS engine...")
    os.environ['COQUI_TOS_AGREED'] = '1'
    tts = TTSEngine(model_path="tts_models/multilingual/multi-dataset/xtts_v2")

    print("\n🎭 Initializing emotion encoder...")
    emotion_encoder = EmotionEncoder()

    # Generate data
    metadata = []
    all_embeddings = {}

    print(f"\n🎵 Synthesizing audio for {len(texts)} texts...")
    for i, item in enumerate(tqdm(texts, desc="TTS synthesis")):
        text = item['text']
        sample_id = f"literary_{i:04d}"
        audio_path = audio_dir / f"{sample_id}.wav"

        try:
            # Synthesize audio
            tts.synthesize(text, output_path=str(audio_path))

            metadata.append({
                "id": sample_id,
                "text": text,
                "source": item.get('source', 'literary'),
                "emotion": item.get('emotion', 'unknown'),
                "audio_path": str(audio_path.relative_to(output_path)),
                "has_audio": True,
                "has_embedding": False
            })
        except Exception as e:
            print(f"\n⚠ Failed to synthesize sample {i}: {e}")
            continue

    # Extract emotion embeddings
    print(f"\n🎭 Extracting emotion embeddings...")
    for item in tqdm(metadata, desc="Emotion encoding"):
        audio_path = output_path / item["audio_path"]
        try:
            embedding = emotion_encoder.encode_from_file(str(audio_path))
            all_embeddings[item["id"]] = embedding.cpu().numpy().flatten()
            item["has_embedding"] = True
        except Exception as e:
            print(f"\n⚠ Failed to encode {item['id']}: {e}")
            continue

    # Save metadata
    print(f"\n💾 Saving metadata...")
    with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save emotion embeddings
    print(f"\n💾 Saving emotion embeddings...")
    np.savez_compressed(output_path / "emotion_embeddings.npz", **all_embeddings)

    # Create full dataset file
    print(f"\n💾 Creating full dataset...")
    full_dataset = []
    for item in metadata:
        if item["has_embedding"]:
            full_dataset.append({
                "id": item["id"],
                "text": item["text"],
                "source": item["source"],
                "emotion": item["emotion"],
                "audio_path": item["audio_path"],
                "emotion_embedding": all_embeddings[item["id"]].tolist() if item["id"] in all_embeddings else []
            })

    with open(output_path / "full_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, indent=2, ensure_ascii=False)

    # Summary
    successful = len([m for m in metadata if m["has_embedding"]])

    print("\n" + "="*80)
    print("✅ Literary dataset generation complete!")
    print("="*80)
    print(f"\n📊 Statistics:")
    print(f"   Total texts: {len(texts)}")
    print(f"   Successful samples: {successful}")
    print(f"   With audio: {len([m for m in metadata if m['has_audio']])}")
    print(f"   With embeddings: {successful}")
    print(f"\n📁 Output directory: {output_path}")
    print(f"   • metadata.json")
    print(f"   • emotion_embeddings.npz")
    print(f"   • full_dataset.json")
    print(f"   • audio/ ({len(list(audio_dir.glob('*.wav')))} WAV files)")
    print()

def main():
    parser = argparse.ArgumentParser(description="Generate literary dataset")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="./audio_augmented_llm/data/literary_500", help="Output directory")
    parser.add_argument("--dataset_type", type=str, default="tinystories", choices=["tinystories", "daily_dialog", "curated"], help="Dataset type")

    args = parser.parse_args()

    generate_literary_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type
    )

if __name__ == '__main__':
    main()
