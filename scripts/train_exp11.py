#!/usr/bin/env python3
"""
Training script for Experiment 11: Multi-Speaker Training.
Uses separate train and validation directories.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from audio_augmented_llm.src.student_training.student_model import StudentModelWithEmotion
from audio_augmented_llm.src.student_training.dataset import EmotionAugmentedDataset, collate_fn


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        emotion_embeddings = batch.get("emotion_embeddings")

        if emotion_embeddings is not None:
            emotion_embeddings = emotion_embeddings.to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            emotion_embeddings=emotion_embeddings,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            emotion_embeddings = batch.get("emotion_embeddings")

            if emotion_embeddings is not None:
                emotion_embeddings = emotion_embeddings.to(device)

            outputs = model(
                input_ids=input_ids,
                emotion_embeddings=emotion_embeddings,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train Experiment 11: Multi-Speaker Model")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Training data directory")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Validation data directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--emotion_dim", type=int, default=256)
    parser.add_argument("--embedding_type", type=str, default="wavlm")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="./audio_augmented_llm/models/exp11_2speaker")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Experiment 11: Multi-Speaker Training")
    print("=" * 80)
    print(f"Train dir: {args.train_dir}")
    print(f"Val dir: {args.val_dir}")
    print(f"Device: {device}")
    print(f"Embedding: {args.embedding_type} ({args.emotion_dim}D)")
    print(f"Epochs: {args.num_epochs}, Batch: {args.batch_size}, LR: {args.learning_rate}")
    print("=" * 80)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    print("\nLoading training dataset...")
    train_dataset = EmotionAugmentedDataset(
        data_dir=args.train_dir,
        tokenizer=tokenizer,
        max_length=512,
        use_emotion=True,
        embedding_type=args.embedding_type
    )
    print(f"✓ Train samples: {len(train_dataset)}")

    print("\nLoading validation dataset...")
    val_dataset = EmotionAugmentedDataset(
        data_dir=args.val_dir,
        tokenizer=tokenizer,
        max_length=512,
        use_emotion=True,
        embedding_type=args.embedding_type
    )
    print(f"✓ Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize model
    print("\nInitializing model...")
    model = StudentModelWithEmotion(
        model_name=args.model_name,
        emotion_dim=args.emotion_dim,
        integration_mode="concat",
        use_lora=True,
        load_in_4bit=True
    )
    model = model.to(device)
    print("✓ Model initialized")

    # Optimizer
    print("\nSetting up optimizer...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        print(f"Val loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best_model"
            print(f"✓ Saving best model to {checkpoint_path}")

            model.base_model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            torch.save(model.emotion_projection.state_dict(),
                      checkpoint_path / "emotion_projection.pt")

            import json
            config = {
                "model_name": args.model_name,
                "emotion_dim": args.emotion_dim,
                "integration_mode": "concat",
                "best_val_loss": best_val_loss,
                "epoch": epoch + 1,
                "experiment": "exp11_multispeaker"
            }
            with open(checkpoint_path / "training_config.json", "w") as f:
                json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
