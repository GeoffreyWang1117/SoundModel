#!/usr/bin/env python3
"""
Compare generation quality of text-only vs text+audio models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer
from peft import PeftModel

# Import our modules
from audio_augmented_llm.src.student_training.student_model import StudentModelWithEmotion
import numpy as np


def load_model(model_dir, use_emotion=False, device="cuda"):
    """Load a trained model."""
    print(f"\nLoading model from {model_dir}...")

    # Load config
    config_path = Path(model_dir) / "training_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = StudentModelWithEmotion(
        model_name=config["model_name"],
        emotion_dim=config["emotion_dim"],
        integration_mode=config["integration_mode"],
        use_lora=True,
        load_in_4bit=True
    )

    # Load trained weights
    model.base_model = PeftModel.from_pretrained(
        model.base_model,
        model_dir,
        is_trainable=False
    )

    # Load emotion projection if available
    if use_emotion:
        emotion_proj_path = Path(model_dir) / "emotion_projection.pt"
        if emotion_proj_path.exists():
            model.emotion_projection.load_state_dict(
                torch.load(emotion_proj_path, map_location=device)
            )

    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded (best val loss: {config['best_val_loss']:.4f})")

    return model, tokenizer


def generate_response(model, tokenizer, context, emotion_embedding=None, device="cuda"):
    """Generate a response for a given context."""
    # Format input
    input_text = f"User: {context}\nAssistant:"

    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=256
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Convert emotion embedding to tensor if provided
    if emotion_embedding is not None:
        if isinstance(emotion_embedding, list):
            emotion_embedding = np.array(emotion_embedding)
        emotion_embedding = torch.tensor(emotion_embedding, dtype=torch.float32).unsqueeze(0).to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            emotion_embeddings=emotion_embedding,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()

    return response


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=" * 70)
    print("Generation Quality Comparison")
    print("=" * 70)

    # Load models
    text_only_model, text_only_tokenizer = load_model(
        "./outputs/baseline_text_only/best_model",
        use_emotion=False,
        device=device
    )

    text_audio_model, text_audio_tokenizer = load_model(
        "./outputs/baseline_text_audio/best_model",
        use_emotion=True,
        device=device
    )

    # Load test samples with emotion embeddings
    data_dir = Path("./audio_augmented_llm/data/train_100")

    with open(data_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)

    emotion_embeddings = np.load(data_dir / "emotion_embeddings.npz")

    # Select 5 test samples
    test_indices = [0, 1, 2, 3, 4]  # joy, sadness, anger, fear, surprise

    print("\n" + "=" * 70)
    print("Comparing Generations on Test Samples")
    print("=" * 70)

    results = []

    for i in test_indices:
        sample = metadata[i]
        context = sample["context"]
        ground_truth = sample["teacher_response"]
        emotion = sample["emotion"]
        sample_id = sample["id"]

        # Get emotion embedding
        emotion_emb = emotion_embeddings[sample_id]

        print(f"\n{'=' * 70}")
        print(f"Sample {i+1} - Emotion: {emotion.upper()}")
        print(f"{'=' * 70}")
        print(f"\n📝 Context:\n   \"{context}\"")
        print(f"\n🎯 Ground Truth:\n   \"{ground_truth}\"")

        # Generate with text-only model
        print(f"\n🤖 Text-Only Model:")
        text_only_response = generate_response(
            text_only_model,
            text_only_tokenizer,
            context,
            emotion_embedding=None,
            device=device
        )
        print(f"   \"{text_only_response}\"")

        # Generate with text+audio model
        print(f"\n🎵 Text+Audio Model:")
        text_audio_response = generate_response(
            text_audio_model,
            text_audio_tokenizer,
            context,
            emotion_embedding=emotion_emb,
            device=device
        )
        print(f"   \"{text_audio_response}\"")

        results.append({
            "sample_id": i,
            "emotion": emotion,
            "context": context,
            "ground_truth": ground_truth,
            "text_only": text_only_response,
            "text_audio": text_audio_response
        })

    # Save results
    output_dir = Path("./outputs/generation_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "comparison_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("✅ Generation comparison complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
