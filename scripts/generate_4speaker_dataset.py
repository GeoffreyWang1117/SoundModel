#!/usr/bin/env python3
"""
Generate balanced 4-speaker dataset for Exp 13.
"""
import sys
import os

# Change to project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import librosa

from audio_augmented_llm.src.tts_pipeline.tts_engine import TTSEngine
from audio_augmented_llm.src.emotion_encoder.emotion_model import EmotionEncoder


def generate_4speaker_dataset(
    output_dir: str,
    samples_per_speaker: int = 200,
    val_samples_per_speaker: int = 20,
    seed: int = 42
):
    """
    Generate balanced 4-speaker dataset.
    
    Args:
        output_dir: Output directory
        samples_per_speaker: Samples per speaker for training
        val_samples_per_speaker: Samples per speaker for validation
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Speakers (gender balanced)
    speakers = [
        "Claribel Dervla",  # Female
        "Damien Black",      # Male
        "Andrew Chipper",    # Male
        "Gracie Wise"        # Female
    ]
    
    print("=" * 80)
    print("Exp 13: 4-Speaker Balanced Dataset Generation")
    print("=" * 80)
    print(f"Speakers: {speakers}")
    print(f"Samples per speaker (train): {samples_per_speaker}")
    print(f"Samples per speaker (val): {val_samples_per_speaker}")
    print(f"Total samples: {len(speakers) * (samples_per_speaker + val_samples_per_speaker)}")
    print()
    
    # Create directories
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    for dir_path in [train_dir, val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "audio").mkdir(exist_ok=True)
    
    # Initialize TTS and encoder
    print("Initializing TTS and emotion encoder...")
    tts_engine = TTSEngine(model_type="xtts", model_path="tts_models/multilingual/multi-dataset/xtts_v2")
    encoder = EmotionEncoder(model_type="wavlm")
    print("✓ Initialization complete")
    print()
    
    # Load context data
    context_file = Path("audio_augmented_llm/data/train_1000/metadata.json")
    with open(context_file, 'r') as f:
        original_data = json.load(f)
    
    print(f"Loaded {len(original_data)} context samples")
    print()
    
    # Process each speaker
    for speaker_id, speaker in enumerate(speakers):
        print(f"\n{'='*80}")
        print(f"Processing Speaker {speaker_id + 1}/4: {speaker}")
        print(f"{'='*80}\n")
        
        # Sample contexts for this speaker
        total_samples = samples_per_speaker + val_samples_per_speaker
        sampled_indices = random.sample(range(len(original_data)), total_samples)
        
        # Generate audio and embeddings
        train_metadata = []
        val_metadata = []
        train_embeddings = {}
        val_embeddings = {}
        
        for idx, sample_idx in enumerate(tqdm(sampled_indices, desc=f"Generating {speaker}")):
            sample = original_data[sample_idx]
            
            # Determine if train or val
            is_val = idx >= samples_per_speaker
            sample_id = f"{speaker.lower().replace(' ', '_')}_{idx:04d}"
            
            # Prepare text
            context = sample.get('context', '')
            response = sample.get('teacher_response', sample.get('text', ''))
            full_text = f"{context} {response}".strip()
            
            # Select output directory
            if is_val:
                current_dir = val_dir
                current_metadata = val_metadata
                current_embeddings = val_embeddings
            else:
                current_dir = train_dir
                current_metadata = train_metadata
                current_embeddings = train_embeddings
            
            # Audio path
            audio_path = current_dir / "audio" / f"{sample_id}.wav"
            
            # Generate audio
            try:
                tts_engine.synthesize(
                    text=full_text,
                    speaker=speaker,
                    output_path=str(audio_path)
                )
                
                # Extract WavLM embedding
                audio, sr = librosa.load(str(audio_path), sr=16000)
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                
                with torch.no_grad():
                    embedding, _ = encoder.forward(audio_tensor)
                    current_embeddings[sample_id] = embedding.detach().cpu().numpy()[0]
                
                # Add to metadata
                current_metadata.append({
                    "id": sample_id,
                    "context": context,
                    "teacher_response": response,
                    "emotion": sample.get('emotion', 'neutral'),
                    "speaker": speaker,
                    "speaker_id": speaker_id,
                    "audio_path": str(audio_path.relative_to(current_dir)),
                    "has_embedding": True
                })
                
            except Exception as e:
                print(f"Error processing {sample_id}: {e}")
                continue
        
        # Save metadata and embeddings for this speaker
        print(f"Saving data for {speaker}...")
        
        # Append to train metadata and embeddings
        if train_metadata:
            train_meta_file = train_dir / "metadata.json"
            if train_meta_file.exists():
                with open(train_meta_file, 'r') as f:
                    existing_train = json.load(f)
                existing_train.extend(train_metadata)
                with open(train_meta_file, 'w') as f:
                    json.dump(existing_train, f, indent=2)
            else:
                with open(train_meta_file, 'w') as f:
                    json.dump(train_metadata, f, indent=2)
            
            # Merge embeddings
            emb_file = train_dir / "embeddings.npz"
            if emb_file.exists():
                existing_emb = dict(np.load(emb_file))
                existing_emb.update(train_embeddings)
                np.savez_compressed(emb_file, **existing_emb)
            else:
                np.savez_compressed(emb_file, **train_embeddings)
        
        # Append to val metadata and embeddings
        if val_metadata:
            val_meta_file = val_dir / "metadata.json"
            if val_meta_file.exists():
                with open(val_meta_file, 'r') as f:
                    existing_val = json.load(f)
                existing_val.extend(val_metadata)
                with open(val_meta_file, 'w') as f:
                    json.dump(existing_val, f, indent=2)
            else:
                with open(val_meta_file, 'w') as f:
                    json.dump(val_metadata, f, indent=2)
            
            # Merge embeddings
            emb_file = val_dir / "embeddings.npz"
            if emb_file.exists():
                existing_emb = dict(np.load(emb_file))
                existing_emb.update(val_embeddings)
                np.savez_compressed(emb_file, **existing_emb)
            else:
                np.savez_compressed(emb_file, **val_embeddings)
        
        print(f"✓ {speaker}: {len(train_metadata)} train, {len(val_metadata)} val samples")
    
    print("\n" + "=" * 80)
    print("Dataset generation complete!")
    print("=" * 80)
    
    # Final counts
    with open(train_dir / "metadata.json", 'r') as f:
        final_train = json.load(f)
    with open(val_dir / "metadata.json", 'r') as f:
        final_val = json.load(f)
    
    print(f"Total train samples: {len(final_train)}")
    print(f"Total val samples: {len(final_val)}")
    print(f"Train directory: {train_dir}")
    print(f"Val directory: {val_dir}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate 4-speaker balanced dataset")
    parser.add_argument("--output_dir", type=str, default="./audio_augmented_llm/data/exp13_4speaker",
                        help="Output directory")
    parser.add_argument("--samples_per_speaker", type=int, default=200,
                        help="Samples per speaker for training")
    parser.add_argument("--val_samples", type=int, default=20,
                        help="Samples per speaker for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    generate_4speaker_dataset(
        output_dir=args.output_dir,
        samples_per_speaker=args.samples_per_speaker,
        val_samples_per_speaker=args.val_samples,
        seed=args.seed
    )
