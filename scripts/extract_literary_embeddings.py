#!/usr/bin/env python3
"""
Extract emotion embeddings for already-generated literary audio files.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from audio_augmented_llm.src.emotion_encoder.emotion_model import EmotionEncoder

def main():
    data_dir = Path("./audio_augmented_llm/data/literary_500")

    # Load existing metadata
    with open(data_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)

    print(f"\n🎭 Initializing emotion encoder...")
    emotion_encoder = EmotionEncoder(
        model_type="wavlm",
        model_name="microsoft/wavlm-base-plus",
        embedding_dim=256,
    )
    emotion_encoder.eval()
    emotion_encoder.to("cuda")

    # Extract embeddings
    print(f"\n🎭 Extracting emotion embeddings for {len(metadata)} audio files...")
    all_embeddings = {}
    successful = 0

    for item in tqdm(metadata, desc="Emotion encoding"):
        audio_path = data_dir / item["audio_path"]
        try:
            embedding = emotion_encoder.encode_from_file(str(audio_path), device="cuda")
            all_embeddings[item["id"]] = embedding.cpu().numpy().flatten()
            item["has_embedding"] = True
            successful += 1
        except Exception as e:
            print(f"\n⚠ Failed to encode {item['id']}: {e}")
            item["has_embedding"] = False
            continue

    # Save updated metadata
    print(f"\n💾 Saving updated metadata...")
    with open(data_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save emotion embeddings
    print(f"\n💾 Saving emotion embeddings...")
    np.savez_compressed(data_dir / "emotion_embeddings.npz", **all_embeddings)

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
                "emotion_embedding": all_embeddings[item["id"]].tolist()
            })

    with open(data_dir / "full_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "="*80)
    print("✅ Emotion embedding extraction complete!")
    print("="*80)
    print(f"\n📊 Statistics:")
    print(f"   Total audio files: {len(metadata)}")
    print(f"   Successful embeddings: {successful}")
    print(f"   Failed: {len(metadata) - successful}")
    print(f"\n📁 Output files:")
    print(f"   • metadata.json (updated)")
    print(f"   • emotion_embeddings.npz ({successful} embeddings)")
    print(f"   • full_dataset.json ({successful} samples)")
    print()

if __name__ == '__main__':
    main()
