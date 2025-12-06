#!/usr/bin/env python3
"""
Generate cross-speaker test set for generalization testing.
Use different TTS speakers to synthesize the same text.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from pathlib import Path
from TTS.api import TTS
from tqdm import tqdm

# Available XTTS speakers (different from training speaker "Claribel Dervla")
TEST_SPEAKERS = [
    "Damien Black",      # Male voice 1
    "Gilberto Mathias",  # Male voice 2
    "Royston Min",       # Male voice 3
]

def generate_cross_speaker_test(
    source_metadata: str,
    output_dir: str,
    num_samples: int = 100,
    speaker: str = "Damien Black"
):
    """
    Generate test set with different TTS speaker.

    Args:
        source_metadata: Path to original metadata.json
        output_dir: Output directory for cross-speaker test
        num_samples: Number of samples to generate
        speaker: TTS speaker name
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Cross-Speaker Test Generation")
    print("=" * 80)
    print(f"Source: {source_metadata}")
    print(f"Output: {output_dir}")
    print(f"Speaker: {speaker}")
    print(f"Samples: {num_samples}")
    print()

    # Load source metadata
    with open(source_metadata, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    # Randomly sample
    random.seed(42)  # Reproducible
    sampled = random.sample(source_data, min(num_samples, len(source_data)))

    print(f"Sampled {len(sampled)} entries from source")

    # Initialize TTS
    print("\nInitializing XTTS v2...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    print("✓ XTTS loaded")

    # Create audio directory
    audio_dir = output_path / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Generate audio with new speaker
    new_metadata = []
    successful = 0
    failed = 0

    for item in tqdm(sampled, desc=f"Generating ({speaker})"):
        sample_id = item['id']

        # Reconstruct text from context and response
        if 'context' in item:
            text = f"{item['context']} {item['teacher_response']}"
        else:
            text = item.get('text', '')

        if not text:
            print(f"\n⚠ Skipping {sample_id}: no text")
            failed += 1
            continue

        # Generate audio with new speaker
        audio_path = audio_dir / f"{sample_id}.wav"

        try:
            tts.tts_to_file(
                text=text,
                speaker=speaker,
                language="en",
                file_path=str(audio_path)
            )

            # Create metadata entry
            new_item = {
                'id': sample_id,
                'audio_path': str(audio_path),
                'text': text,
                'emotion': item.get('emotion', 'neutral'),
                'speaker': speaker,
                'source_speaker': 'Claribel Dervla'
            }

            new_metadata.append(new_item)
            successful += 1

        except Exception as e:
            print(f"\n⚠ Failed {sample_id}: {e}")
            failed += 1

    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(new_metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✅ Cross-speaker test generation complete!")
    print("=" * 80)
    print(f"\n📊 Statistics:")
    print(f"   Requested: {num_samples}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {successful / num_samples * 100:.1f}%")
    print(f"\n📁 Output: {output_dir}")
    print(f"   Metadata: {metadata_path}")
    print(f"   Audio: {audio_dir}/ ({successful} files)")
    print()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate cross-speaker test set")
    parser.add_argument("--source", type=str,
                        default="./audio_augmented_llm/data/train_1000/metadata.json",
                        help="Source metadata.json")
    parser.add_argument("--output", type=str,
                        default="./audio_augmented_llm/data/test_cross_speaker",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--speaker", type=str, default="Damien Black",
                        choices=TEST_SPEAKERS,
                        help="TTS speaker name")

    args = parser.parse_args()

    generate_cross_speaker_test(
        args.source,
        args.output,
        args.num_samples,
        args.speaker
    )

if __name__ == '__main__':
    main()
