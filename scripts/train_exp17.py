#!/usr/bin/env python3
"""
Experiment 17B: Joint Training with Contrastive Learning

Train student model with supervised contrastive loss to learn speaker-invariant
emotion representations while maintaining language modeling capability.

Architecture:
    Audio → WavLM (frozen) → Emotion Projection → L2 Normalize → Contrastive Loss
                                                ↓
                                           LLM Input → Language Modeling Loss

Combined Loss:
    Total = λ_contrastive * Loss_contrastive + λ_lm * Loss_LM
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from audio_augmented_llm.src.student_training.contrastive_loss import SupConLossWithStats


# ============================================================================
# Dataset with Contrastive Learning Support
# ============================================================================

class EmotionDatasetWithLabels(Dataset):
    """
    Dataset that returns emotion_label and speaker_id for contrastive learning.
    """

    def __init__(self, data_dir: str, tokenizer, max_length: int = 512, embedding_type: str = "wavlm"):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.embedding_type = embedding_type

        # Load metadata
        with open(self.data_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)

        # Load embeddings
        # Check for different naming conventions
        if embedding_type == "wavlm":
            # Try emotion_embeddings.npz first (exp13 format), then wavlm_embeddings.npz
            embeddings_file = self.data_dir / "emotion_embeddings.npz"
            if not embeddings_file.exists():
                embeddings_file = self.data_dir / "wavlm_embeddings.npz"
        else:
            embeddings_file = self.data_dir / f"{embedding_type}_embeddings.npz"

        self.embeddings_data = np.load(embeddings_file)

        # Create emotion and speaker mappings
        self.emotion_to_id = {
            'anger': 0, 'fear': 1, 'joy': 2, 'neutral': 3, 'sadness': 4, 'surprise': 5
        }

        # Extract unique speakers and create mapping
        speakers = sorted(set(item['speaker'] for item in self.metadata))
        self.speaker_to_id = {speaker: idx for idx, speaker in enumerate(speakers)}

        print(f"Dataset: {data_dir}")
        print(f"  Samples: {len(self.metadata)}")
        print(f"  Emotions: {len(self.emotion_to_id)} - {list(self.emotion_to_id.keys())}")
        print(f"  Speakers: {len(self.speaker_to_id)} - {list(self.speaker_to_id.keys())}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        sample_id = item['id']

        # Get emotion embedding
        emotion_embedding = self.embeddings_data[sample_id].astype(np.float32)

        # Get emotion label and speaker ID
        emotion_label = self.emotion_to_id[item['emotion']]
        speaker_id = item['speaker_id']

        # Create conversation
        context = item['context']
        response = item['teacher_response']

        # Format as conversation
        conversation_text = f"Context: {context}\nResponse: {response}"

        # Tokenize
        encoded = self.tokenizer(
            conversation_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # Labels (for language modeling)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'emotion_embedding': torch.from_numpy(emotion_embedding),
            'emotion_label': torch.tensor(emotion_label, dtype=torch.long),
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long),
            'sample_id': sample_id
        }


def stratified_batch_sampler(dataset: Dataset, batch_size: int, num_speakers_per_batch: int = 4):
    """
    Create batches that contain samples from multiple speakers for each emotion.

    Args:
        dataset: EmotionDatasetWithLabels
        batch_size: Total batch size (should be divisible by num_speakers_per_batch)
        num_speakers_per_batch: Number of different speakers to include per batch

    Yields:
        List of sample indices for each batch
    """
    # Group samples by (speaker, emotion)
    speaker_emotion_groups = {}
    for idx, item in enumerate(dataset.metadata):
        speaker = item['speaker']
        emotion = item['emotion']
        key = (speaker, emotion)
        if key not in speaker_emotion_groups:
            speaker_emotion_groups[key] = []
        speaker_emotion_groups[key].append(idx)

    # Shuffle each group
    for key in speaker_emotion_groups:
        np.random.shuffle(speaker_emotion_groups[key])

    speakers = list(dataset.speaker_to_id.keys())
    emotions = list(dataset.emotion_to_id.keys())

    samples_per_speaker = batch_size // num_speakers_per_batch

    while True:
        # Sample speakers
        sampled_speakers = np.random.choice(speakers, size=num_speakers_per_batch, replace=False)

        batch_indices = []

        for speaker in sampled_speakers:
            # Sample emotions for this speaker
            speaker_samples = []
            for emotion in emotions:
                key = (speaker, emotion)
                if key in speaker_emotion_groups and len(speaker_emotion_groups[key]) > 0:
                    speaker_samples.append((key, emotion))

            # Sample from this speaker
            for _ in range(samples_per_speaker):
                if not speaker_samples:
                    break

                # Randomly pick an emotion for this speaker
                key, emotion = speaker_samples[np.random.randint(len(speaker_samples))]

                if len(speaker_emotion_groups[key]) > 0:
                    idx = speaker_emotion_groups[key].pop()
                    batch_indices.append(idx)

                    # Refill if empty
                    if len(speaker_emotion_groups[key]) == 0:
                        speaker_emotion_groups[key] = [
                            i for i, item in enumerate(dataset.metadata)
                            if item['speaker'] == key[0] and item['emotion'] == key[1]
                        ]
                        np.random.shuffle(speaker_emotion_groups[key])

        if len(batch_indices) >= batch_size:
            yield batch_indices[:batch_size]
        else:
            # Not enough samples, skip this batch
            continue


# ============================================================================
# Student Model with Contrastive Learning
# ============================================================================

class StudentModelWithContrastive(nn.Module):
    """
    Student model that supports both language modeling and contrastive learning.
    """

    def __init__(self, base_model, emotion_dim: int, hidden_size: int, normalize_emotions: bool = True):
        super().__init__()
        self.base_model = base_model
        self.emotion_dim = emotion_dim
        self.hidden_size = hidden_size
        self.normalize_emotions = normalize_emotions

        # Emotion projection layer
        self.emotion_projection = nn.Linear(emotion_dim, hidden_size)

    def forward(self, input_ids, attention_mask, emotion_embedding, labels=None):
        """
        Forward pass for language modeling.

        Returns:
            outputs with .loss and .logits
        """
        batch_size = input_ids.shape[0]

        # Project emotion embeddings
        emotion_hidden = self.emotion_projection(emotion_embedding)  # (batch_size, hidden_size)
        emotion_hidden = emotion_hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)

        # Get embeddings from base model
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        # Match dtype of emotion_hidden to inputs_embeds (for quantized models)
        emotion_hidden = emotion_hidden.to(inputs_embeds.dtype)

        # Prepend emotion embedding
        inputs_embeds = torch.cat([emotion_hidden, inputs_embeds], dim=1)

        # Adjust attention mask
        emotion_attention = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([emotion_attention, attention_mask], dim=1)

        # Adjust labels
        if labels is not None:
            emotion_labels = torch.full((batch_size, 1), -100, device=labels.device, dtype=labels.dtype)
            labels = torch.cat([emotion_labels, labels], dim=1)

        # Forward through model
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs

    def get_emotion_representations(self, emotion_embedding):
        """
        Get L2-normalized emotion representations for contrastive learning.

        Args:
            emotion_embedding: (batch_size, emotion_dim)

        Returns:
            (batch_size, hidden_size) L2-normalized embeddings
        """
        emotion_hidden = self.emotion_projection(emotion_embedding)

        if self.normalize_emotions:
            emotion_hidden = F.normalize(emotion_hidden, dim=1)

        return emotion_hidden


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: StudentModelWithContrastive,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    contrastive_criterion: SupConLossWithStats,
    lambda_lm: float,
    lambda_contrastive: float,
    device: torch.device,
    epoch: int
):
    """
    Train for one epoch with combined loss.
    """
    model.train()

    total_loss = 0.0
    total_lm_loss = 0.0
    total_contrastive_loss = 0.0
    total_positives = 0
    total_negatives = 0
    avg_positive_sim = 0.0
    avg_negative_sim = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        emotion_embedding = batch['emotion_embedding'].to(device)
        emotion_label = batch['emotion_label'].to(device)
        speaker_id = batch['speaker_id'].to(device)

        # Forward pass for language modeling
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            emotion_embedding=emotion_embedding,
            labels=labels
        )

        lm_loss = outputs.loss

        # Get emotion representations for contrastive learning
        emotion_repr = model.get_emotion_representations(emotion_embedding)

        # Compute contrastive loss
        contrastive_loss, stats = contrastive_criterion.forward_with_stats(
            emotion_repr,
            emotion_label,
            speaker_id
        )

        # Combined loss
        loss = lambda_lm * lm_loss + lambda_contrastive * contrastive_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate statistics
        total_loss += loss.item()
        total_lm_loss += lm_loss.item()
        total_contrastive_loss += contrastive_loss.item()
        total_positives += stats['num_positives']
        total_negatives += stats['num_negatives']
        avg_positive_sim += stats['avg_positive_sim']
        avg_negative_sim += stats['avg_negative_sim']
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lm': f"{lm_loss.item():.4f}",
            'con': f"{contrastive_loss.item():.4f}",
            'pos_sim': f"{stats['avg_positive_sim']:.3f}",
            'neg_sim': f"{stats['avg_negative_sim']:.3f}"
        })

    # Compute averages
    avg_loss = total_loss / num_batches
    avg_lm_loss = total_lm_loss / num_batches
    avg_contrastive_loss = total_contrastive_loss / num_batches
    avg_positives = total_positives / num_batches
    avg_negatives = total_negatives / num_batches
    avg_positive_sim = avg_positive_sim / num_batches
    avg_negative_sim = avg_negative_sim / num_batches

    return {
        'loss': avg_loss,
        'lm_loss': avg_lm_loss,
        'contrastive_loss': avg_contrastive_loss,
        'avg_positives': avg_positives,
        'avg_negatives': avg_negatives,
        'avg_positive_sim': avg_positive_sim,
        'avg_negative_sim': avg_negative_sim
    }


def evaluate(
    model: StudentModelWithContrastive,
    dataloader: DataLoader,
    device: torch.device
):
    """
    Evaluate language modeling loss on validation set.
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            emotion_embedding = batch['emotion_embedding'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                emotion_embedding=emotion_embedding,
                labels=labels
            )

            total_loss += outputs.loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Experiment 17B with contrastive learning")

    # Data arguments
    parser.add_argument('--train_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Validation data directory')
    parser.add_argument('--embedding_type', type=str, default='wavlm', choices=['wavlm', 'acoustic'])

    # Model arguments
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--emotion_dim', type=int, default=256, help='Emotion embedding dimension')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=512)

    # Contrastive learning arguments
    parser.add_argument('--lambda_lm', type=float, default=1.0, help='Weight for LM loss')
    parser.add_argument('--lambda_contrastive', type=float, default=0.3, help='Weight for contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.07, help='Contrastive loss temperature')

    # LoRA arguments
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./audio_augmented_llm/models/exp17b_contrastive')
    parser.add_argument('--log_file', type=str, default='./outputs/exp17b_training_log.txt')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    # Save configuration
    config = vars(args)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 80)
    print("Experiment 17B: Joint Training with Contrastive Learning")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = EmotionDatasetWithLabels(
        args.train_dir,
        tokenizer,
        max_length=args.max_length,
        embedding_type=args.embedding_type
    )

    val_dataset = EmotionDatasetWithLabels(
        args.val_dir,
        tokenizer,
        max_length=args.max_length,
        embedding_type=args.embedding_type
    )

    # Create dataloaders with stratified sampling
    print("\nCreating dataloaders with stratified sampling...")
    train_sampler = stratified_batch_sampler(train_dataset, args.batch_size, num_speakers_per_batch=4)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=[[train_sampler.__next__() for _ in range(len(train_dataset) // args.batch_size)]],
        num_workers=0
    )

    # Actually, let's use a simpler approach for now
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Load base model with 4-bit quantization
    print("\nLoading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # Use single GPU to avoid device mismatch with emotion_projection
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},  # Force single GPU
        trust_remote_code=True
    )

    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # Create student model
    print("\nCreating student model with contrastive learning support...")
    hidden_size = base_model.config.hidden_size
    model = StudentModelWithContrastive(
        base_model=base_model,
        emotion_dim=args.emotion_dim,
        hidden_size=hidden_size,
        normalize_emotions=True
    )
    model = model.to(device)

    # Create contrastive criterion
    contrastive_criterion = SupConLossWithStats(
        temperature=args.temperature,
        base_temperature=args.temperature
    )

    # Create optimizer (include emotion projection parameters)
    optimizer = torch.optim.AdamW(
        [
            {'params': model.emotion_projection.parameters()},
            {'params': model.base_model.parameters()}
        ],
        lr=args.learning_rate
    )

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    best_val_loss = float('inf')
    training_history = []

    with open(args.log_file, 'w') as log_f:
        log_f.write("Experiment 17B Training Log\n")
        log_f.write("=" * 80 + "\n\n")
        log_f.write(f"Configuration:\n{json.dumps(config, indent=2)}\n\n")
        log_f.write("=" * 80 + "\n\n")

        for epoch in range(1, args.num_epochs + 1):
            print(f"\nEpoch {epoch}/{args.num_epochs}")
            print("-" * 80)

            # Train
            train_stats = train_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                contrastive_criterion=contrastive_criterion,
                lambda_lm=args.lambda_lm,
                lambda_contrastive=args.lambda_contrastive,
                device=device,
                epoch=epoch
            )

            # Evaluate
            val_loss = evaluate(model, val_dataloader, device)

            # Log results
            log_msg = f"\nEpoch {epoch} Results:\n"
            log_msg += f"  Train Loss: {train_stats['loss']:.4f}\n"
            log_msg += f"    - LM Loss: {train_stats['lm_loss']:.4f}\n"
            log_msg += f"    - Contrastive Loss: {train_stats['contrastive_loss']:.4f}\n"
            log_msg += f"  Validation Loss: {val_loss:.4f}\n"
            log_msg += f"  Contrastive Stats:\n"
            log_msg += f"    - Avg Positives/Batch: {train_stats['avg_positives']:.1f}\n"
            log_msg += f"    - Avg Negatives/Batch: {train_stats['avg_negatives']:.1f}\n"
            log_msg += f"    - Avg Positive Similarity: {train_stats['avg_positive_sim']:.4f}\n"
            log_msg += f"    - Avg Negative Similarity: {train_stats['avg_negative_sim']:.4f}\n"

            print(log_msg)
            log_f.write(log_msg + "\n")
            log_f.flush()

            # Save statistics
            training_history.append({
                'epoch': epoch,
                'train_loss': train_stats['loss'],
                'train_lm_loss': train_stats['lm_loss'],
                'train_contrastive_loss': train_stats['contrastive_loss'],
                'val_loss': val_loss,
                'avg_positives': train_stats['avg_positives'],
                'avg_negatives': train_stats['avg_negatives'],
                'avg_positive_sim': train_stats['avg_positive_sim'],
                'avg_negative_sim': train_stats['avg_negative_sim']
            })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(args.output_dir, 'best_model')
                os.makedirs(best_model_path, exist_ok=True)

                model.base_model.save_pretrained(best_model_path)
                torch.save(model.emotion_projection.state_dict(),
                          os.path.join(best_model_path, 'emotion_projection.pt'))

                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
                log_f.write(f"  ✓ Saved best model (val_loss: {val_loss:.4f})\n\n")
                log_f.flush()

        # Save training history
        with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)

        # Final summary
        summary = f"\n{'=' * 80}\n"
        summary += "Training Complete!\n"
        summary += f"{'=' * 80}\n"
        summary += f"Best validation loss: {best_val_loss:.4f}\n"
        summary += f"Model saved to: {args.output_dir}/best_model\n"

        print(summary)
        log_f.write(summary)

    print(f"\nTraining log saved to: {args.log_file}")


if __name__ == '__main__':
    main()
