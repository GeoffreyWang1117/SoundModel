#!/usr/bin/env python3
"""
Generate Viktor Eka test set for Exp 13 (held-out speaker).
"""
import sys
import os

# Change to project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

import json
import random
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import librosa

from audio_augmented_llm.src.tts_pipeline.tts_engine import TTSEngine
from audio_augmented_llm.src.emotion_encoder.emotion_model import EmotionEncoder


def main():
    random.seed(43)  # Different seed from training
    np.random.seed(43)
    torch.manual_seed(43)
    
    print("=" * 80)
    print("Generating Viktor Eka Test Set (Exp 13 Held-Out Speaker)")
    print("=" * 80)
    print("Speaker: Viktor Eka (male, completely unseen)")
    print("Samples: 100")
    print()
    
    # Setup
    output_dir = Path("audio_augmented_llm/data/test_cross_speaker_viktor")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    # Initialize
    print("Initializing TTS and emotion encoder...")
    tts_engine = TTSEngine(model_type="xtts", model_path="tts_models/multilingual/multi-dataset/xtts_v2")
    encoder = EmotionEncoder(model_type="wavlm")
    print("✓ Initialization complete\n")
    
    # Load contexts
    context_file = Path("audio_augmented_llm/data/train_1000/metadata.json")
    with open(context_file, 'r') as f:
        original_data = json.load(f)
    
    # Sample 100 contexts
    sampled_indices = random.sample(range(len(original_data)), 100)
    
    # Generate
    metadata = []
    embeddings = {}
    
    for idx, sample_idx in enumerate(tqdm(sampled_indices, desc="Generating Viktor test set")):
        sample = original_data[sample_idx]
        sample_id = f"viktor_{idx:04d}"
        
        # Prepare text
        context = sample.get('context', '')
        response = sample.get('teacher_response', sample.get('text', ''))
        full_text = f"{context} {response}".strip()
        
        audio_path = audio_dir / f"{sample_id}.wav"
        
        try:
            # Generate audio
            tts_engine.synthesize(
                text=full_text,
                speaker="Viktor Eka",
                output_path=str(audio_path)
            )
            
            # Extract embedding
            audio, sr = librosa.load(str(audio_path), sr=16000)
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            
            with torch.no_grad():
                embedding, _ = encoder.forward(audio_tensor)
                embeddings[sample_id] = embedding.detach().cpu().numpy()[0]
            
            # Metadata
            metadata.append({
                "id": sample_id,
                "context": context,
                "teacher_response": response,
                "emotion": sample.get('emotion', 'neutral'),
                "speaker": "Viktor Eka",
                "speaker_id": 99,  # Unseen ID
                "audio_path": str(audio_path.relative_to(output_dir)),
                "has_embedding": True
            })
            
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            continue
    
    # Save
    print("\nSaving metadata and embeddings...")
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    np.savez_compressed(output_dir / "embeddings.npz", **embeddings)
    
    print("=" * 80)
    print("✅ Viktor test set generation complete!")
    print("=" * 80)
    print(f"Samples: {len(metadata)}")
    print(f"Directory: {output_dir}")
    print()


if __name__ == '__main__':
    main()
