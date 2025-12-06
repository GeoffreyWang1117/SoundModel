#!/usr/bin/env python3
"""
Evaluate a trained student model on a test set.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer

from audio_augmented_llm.src.student_training.student_model import StudentModelWithEmotion
from audio_augmented_llm.src.student_training.student_model_san import StudentModelWithSAN
from audio_augmented_llm.src.student_training.dataset import EmotionAugmentedDataset, collate_fn


def evaluate_model(
    model_dir: str,
    data_dir: str,
    embedding_type: str = "acoustic",
    batch_size: int = 4
):
    """
    Evaluate a trained model on a test set.

    Args:
        model_dir: Directory containing trained model
        data_dir: Directory containing test data
        embedding_type: Type of embedding ('acoustic', 'wavlm', 'fusion')
        batch_size: Batch size for evaluation
    """
    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    print(f"Model: {model_dir}")
    print(f"Test data: {data_dir}")
    print(f"Embedding type: {embedding_type}")
    print()

    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    import json
    from pathlib import Path
    model_path = Path(model_dir)
    with open(model_path / "training_config.json", 'r') as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if this is a SAN model
    use_adaptive_norm = config.get('use_adaptive_norm', False)

    # Reconstruct model
    if use_adaptive_norm:
        print(f"Loading base model: {config['model_name']}...")
        print("Applying LoRA...")
        model = StudentModelWithSAN(
            model_name=config['model_name'],
            emotion_dim=config['emotion_dim'],
            num_speakers=config.get('num_speakers', 4),
            integration_mode=config.get('integration_mode', 'concat'),
            use_lora=True,
            load_in_4bit=True,
            use_adaptive_norm=True,
            inference_mode=config.get('inference_mode', 'average')
        )
        print(f"✓ Student model initialized with concat integration")
    else:
        model = StudentModelWithEmotion(
            model_name=config['model_name'],
            emotion_dim=config['emotion_dim'],
            integration_mode=config.get('integration_mode', 'concat')
        )

    # Load emotion projection weights
    emotion_proj_path = model_path / "emotion_projection.pt"
    if emotion_proj_path.exists():
        model.emotion_projection.load_state_dict(torch.load(emotion_proj_path, map_location=device))

    model.to(device)
    model.eval()
    print(f"✓ Model loaded on {device}")
    print()

    # Create dataset
    print("Loading test dataset...")
    test_dataset = EmotionAugmentedDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=512,
        use_emotion=True,
        embedding_type=embedding_type
    )
    print(f"✓ Test samples: {len(test_dataset)}")
    print()

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Evaluate
    print("Evaluating...")
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            emotion_embeddings = batch['emotion_embeddings'].to(device) if 'emotion_embeddings' in batch else None

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                emotion_embeddings=emotion_embeddings
            )

            loss = outputs.loss

            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

    avg_loss = total_loss / total_samples

    print()
    print("=" * 80)
    print("✅ Evaluation complete!")
    print("=" * 80)
    print(f"\n📊 Results:")
    print(f"   Test loss: {avg_loss:.4f}")
    print(f"   Samples evaluated: {total_samples}")
    print()

    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing trained model")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing test data")
    parser.add_argument("--embedding_type", type=str, default="acoustic",
                        choices=['acoustic', 'wavlm', 'relative', 'fusion'],
                        help="Type of embedding")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")

    args = parser.parse_args()

    evaluate_model(
        args.model_dir,
        args.data_dir,
        args.embedding_type,
        args.batch_size
    )


if __name__ == '__main__':
    main()
