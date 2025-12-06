#!/usr/bin/env python3
"""
Training script for Experiment 15: Speaker Adaptive Normalization (SAN).
Tests whether speaker-conditioned normalization improves cross-speaker generalization.
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

from audio_augmented_llm.src.student_training.student_model_san import StudentModelWithSAN
from audio_augmented_llm.src.student_training.dataset import EmotionAugmentedDataset, collate_fn


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch with speaker conditioning."""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        emotion_embeddings = batch.get("emotion_embeddings")
        speaker_ids = batch.get("speaker_ids")  # Get speaker IDs for training

        if emotion_embeddings is not None:
            emotion_embeddings = emotion_embeddings.to(device)

        if speaker_ids is not None:
            speaker_ids = speaker_ids.to(device)

        # Forward pass with speaker conditioning
        outputs = model(
            input_ids=input_ids,
            emotion_embeddings=emotion_embeddings,
            speaker_ids=speaker_ids,  # Pass speaker IDs during training
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


def evaluate(model, dataloader, device, use_speaker_ids=True):
    """
    Evaluate the model.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run on
        use_speaker_ids: If False, use inference mode (no speaker conditioning)
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            emotion_embeddings = batch.get("emotion_embeddings")
            speaker_ids = batch.get("speaker_ids") if use_speaker_ids else None

            if emotion_embeddings is not None:
                emotion_embeddings = emotion_embeddings.to(device)

            if speaker_ids is not None:
                speaker_ids = speaker_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                emotion_embeddings=emotion_embeddings,
                speaker_ids=speaker_ids,  # None for inference mode
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train Experiment 15: Speaker Adaptive Normalization")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Training data directory")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Validation data directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--emotion_dim", type=int, default=256)
    parser.add_argument("--num_speakers", type=int, default=4,
                        help="Number of speakers in training set")
    parser.add_argument("--embedding_type", type=str, default="wavlm")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--inference_mode", type=str, default="average",
                        choices=["average", "zero", "neutral"],
                        help="Inference mode for speaker conditioning")
    parser.add_argument("--output_dir", type=str, default="./audio_augmented_llm/models/exp15_san")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Experiment 15: Speaker Adaptive Normalization (SAN)")
    print("=" * 80)
    print(f"Train dir: {args.train_dir}")
    print(f"Val dir: {args.val_dir}")
    print(f"Device: {device}")
    print(f"Embedding: {args.embedding_type} ({args.emotion_dim}D)")
    print(f"Num speakers: {args.num_speakers}")
    print(f"Inference mode: {args.inference_mode}")
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

    # Initialize model with Speaker Adaptive Normalization
    print("\nInitializing model with SAN...")
    model = StudentModelWithSAN(
        model_name=args.model_name,
        emotion_dim=args.emotion_dim,
        num_speakers=args.num_speakers,
        integration_mode="concat",
        use_lora=True,
        load_in_4bit=True,
        use_adaptive_norm=True,
        inference_mode=args.inference_mode
    )
    model = model.to(device)
    print("✓ Model initialized with Speaker Adaptive Normalization")

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

        # Train (with speaker IDs)
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate on validation set
        # Option 1: With speaker IDs (same-speaker performance)
        val_loss_with_speaker = evaluate(model, val_loader, device, use_speaker_ids=True)
        print(f"Val loss (with speaker IDs): {val_loss_with_speaker:.4f}")

        # Option 2: Without speaker IDs (simulates cross-speaker)
        val_loss_no_speaker = evaluate(model, val_loader, device, use_speaker_ids=False)
        print(f"Val loss (no speaker IDs): {val_loss_no_speaker:.4f}")

        # Print the gap
        gap = val_loss_no_speaker / val_loss_with_speaker
        print(f"  → Gap: {gap:.2f}× (lower is better)")

        # Use loss without speaker IDs for model selection (more realistic)
        val_loss = val_loss_no_speaker

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
                "num_speakers": args.num_speakers,
                "integration_mode": "concat",
                "use_adaptive_norm": True,
                "inference_mode": args.inference_mode,
                "best_val_loss": best_val_loss,
                "best_val_loss_with_speaker": val_loss_with_speaker,
                "epoch": epoch + 1,
                "experiment": "exp15_speaker_adaptive_norm"
            }
            with open(checkpoint_path / "training_config.json", "w") as f:
                json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print("=" * 80)
    print(f"Best validation loss (no speaker IDs): {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
