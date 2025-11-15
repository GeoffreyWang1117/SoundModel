"""
Dataset loader for student model training with emotion embeddings.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import Dataset


class EmotionAugmentedDataset(Dataset):
    """
    Dataset that loads context-response pairs with emotion embeddings.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: int = 512,
        use_emotion: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing metadata.json and emotion_embeddings.npz
            tokenizer: Tokenizer for the student model
            max_length: Maximum sequence length
            use_emotion: Whether to include emotion embeddings
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_emotion = use_emotion

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

        # Load emotion embeddings
        if use_emotion:
            embeddings_path = self.data_dir / "emotion_embeddings.npz"
            self.emotion_embeddings = np.load(embeddings_path)
        else:
            self.emotion_embeddings = None

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.

        Returns:
            Dictionary with:
                - input_ids: Tokenized input
                - attention_mask: Attention mask
                - labels: Target labels
                - emotion_embedding: Emotion embedding (if use_emotion=True)
        """
        sample = self.samples[idx]

        # Support two data formats:
        # 1. Dialogue format (synthetic data): context + teacher_response
        # 2. Text format (literary data): text only
        if "context" in sample and "teacher_response" in sample:
            # Dialogue format
            context = sample["context"]
            response = sample["teacher_response"]
            input_text = f"User: {context}\nAssistant: {response}"
        elif "text" in sample:
            # Literary text format
            input_text = sample["text"]
        else:
            raise ValueError(f"Sample {idx} has invalid format. Expected 'context'+'teacher_response' or 'text'")

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels (same as input_ids for causal LM)
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Add emotion embedding if available
        if self.use_emotion and sample.get("has_embedding", False):
            sample_id = sample["id"]
            if sample_id in self.emotion_embeddings:
                emotion_emb = self.emotion_embeddings[sample_id]
                # Convert to tensor and ensure correct shape
                if isinstance(emotion_emb, list):
                    emotion_emb = np.array(emotion_emb)
                if len(emotion_emb.shape) == 2:
                    emotion_emb = emotion_emb[0]  # Remove batch dimension if present
                result["emotion_embedding"] = torch.tensor(emotion_emb, dtype=torch.float32)
            else:
                # Use zero embedding as fallback
                result["emotion_embedding"] = torch.zeros(256, dtype=torch.float32)
        elif self.use_emotion:
            # Use zero embedding as fallback
            result["emotion_embedding"] = torch.zeros(256, dtype=torch.float32)

        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched tensors
    """
    # Stack all tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    # Add emotion embeddings if present
    if "emotion_embedding" in batch[0]:
        emotion_embeddings = torch.stack([item["emotion_embedding"] for item in batch])
        result["emotion_embeddings"] = emotion_embeddings

    return result


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    dataset = EmotionAugmentedDataset(
        data_dir="./audio_augmented_llm/data/train_100",
        tokenizer=tokenizer,
        max_length=512,
        use_emotion=True
    )

    # Test loading
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print("\nSample 0:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    # Test batching
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    batch = next(iter(dataloader))
    print("\nBatch:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
