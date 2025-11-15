#!/usr/bin/env python3
"""
Simple training script for student model with emotion embeddings.
This is a minimal version for initial experiments.
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

# Import our modules
from audio_augmented_llm.src.student_training.student_model import StudentModelWithEmotion
from audio_augmented_llm.src.student_training.dataset import EmotionAugmentedDataset, collate_fn


def train_epoch(model, dataloader, optimizer, device, use_emotion=True):
    """
    Train for one epoch.

    Args:
        model: Student model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        use_emotion: Whether to use emotion embeddings

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        emotion_embeddings = None
        if use_emotion and "emotion_embeddings" in batch:
            emotion_embeddings = batch["emotion_embeddings"].to(device)

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

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, device, use_emotion=True):
    """
    Evaluate the model.

    Args:
        model: Student model
        dataloader: Evaluation dataloader
        device: Device to evaluate on
        use_emotion: Whether to use emotion embeddings

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            emotion_embeddings = None
            if use_emotion and "emotion_embeddings" in batch:
                emotion_embeddings = batch["emotion_embeddings"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                emotion_embeddings=emotion_embeddings,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train student model with emotion embeddings")
    parser.add_argument("--data_dir", type=str, default="./audio_augmented_llm/data/train_100",
                        help="Directory containing training data")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="Base model name or path")
    parser.add_argument("--emotion_dim", type=int, default=None,
                        help="Dimension of emotion embeddings (auto-set based on embedding_type if not specified)")
    parser.add_argument("--embedding_type", type=str, default="wavlm",
                        choices=["wavlm", "acoustic", "fusion"],
                        help="Type of embedding: wavlm (256D), acoustic (46D), fusion (302D)")
    parser.add_argument("--integration_mode", type=str, default="concat",
                        choices=["concat", "attention"],
                        help="Emotion integration mode")
    parser.add_argument("--use_emotion", action="store_true",
                        help="Use emotion embeddings (disable for text-only baseline)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="./outputs/student_model",
                        help="Output directory for checkpoints")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on")

    args = parser.parse_args()

    # Auto-set emotion_dim based on embedding_type if not specified
    if args.emotion_dim is None:
        if args.embedding_type == "wavlm":
            args.emotion_dim = 256
        elif args.embedding_type == "acoustic":
            args.emotion_dim = 46
        elif args.embedding_type == "fusion":
            args.emotion_dim = 302

    # Set device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Student Model Training")
    print("=" * 70)
    print(f"Base model: {args.model_name}")
    print(f"Integration mode: {args.integration_mode}")
    print(f"Use emotion: {args.use_emotion}")
    if args.use_emotion:
        print(f"Embedding type: {args.embedding_type} ({args.emotion_dim}D)")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")

    # Create dataset
    print("\nCreating dataset...")
    dataset = EmotionAugmentedDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_emotion=args.use_emotion,
        embedding_type=args.embedding_type if args.use_emotion else "wavlm"
    )

    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize model
    print("\nInitializing student model...")
    model = StudentModelWithEmotion(
        model_name=args.model_name,
        emotion_dim=args.emotion_dim,
        integration_mode=args.integration_mode,
        use_lora=True,
        load_in_4bit=True
    )
    model = model.to(device)
    print("✓ Model initialized")

    # Initialize optimizer
    print("\nSetting up optimizer...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )
    print("✓ Optimizer ready")

    # Training loop
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 70)

        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, device, args.use_emotion)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate
        val_loss = evaluate(model, val_dataloader, device, args.use_emotion)
        print(f"Val loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best_model"
            print(f"✓ Saving best model to {checkpoint_path}")

            # Save model and tokenizer
            model.base_model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

            # Save emotion projection separately
            torch.save(model.emotion_projection.state_dict(),
                      checkpoint_path / "emotion_projection.pt")

            # Save config
            config = {
                "model_name": args.model_name,
                "emotion_dim": args.emotion_dim,
                "integration_mode": args.integration_mode,
                "best_val_loss": best_val_loss,
                "epoch": epoch + 1
            }
            import json
            with open(checkpoint_path / "training_config.json", "w") as f:
                json.dump(config, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ Training completed!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
