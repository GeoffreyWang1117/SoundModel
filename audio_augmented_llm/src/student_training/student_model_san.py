"""
Student model with Speaker Adaptive Normalization (SAN).

This is an enhanced version of student_model.py that incorporates
speaker-conditioned normalization for better cross-speaker generalization.

Key differences from student_model.py:
- Uses SpeakerConditionedProjection instead of EmotionProjection
- Accepts speaker_ids during training
- Automatically removes speaker conditioning during inference
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional, Dict, Any

from .speaker_adaptive_norm import SpeakerConditionedProjection


class StudentModelWithSAN(nn.Module):
    """
    Student LLM model with Speaker Adaptive Normalization.

    Extends StudentModelWithEmotion with speaker-conditioned normalization
    to disentangle speaker identity from emotion information.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        emotion_dim: int = 256,
        num_speakers: int = 4,
        integration_mode: str = "concat",
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        use_adaptive_norm: bool = True,
        inference_mode: str = "average",  # "average", "zero", or "neutral"
    ):
        """
        Initialize student model with Speaker Adaptive Normalization.

        Args:
            model_name: Name or path of the base LLM
            emotion_dim: Dimension of emotion embeddings
            num_speakers: Number of speakers in training set
            integration_mode: How to integrate emotion embeddings (only "concat" supported for now)
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_config: LoRA configuration dict
            load_in_4bit: Whether to load model in 4-bit quantization
            load_in_8bit: Whether to load model in 8-bit quantization
            use_adaptive_norm: Whether to use speaker adaptive normalization
            inference_mode: How to handle speaker conditioning during inference
                - "average": Use average of all speaker parameters
                - "zero": Use no conditioning (scale=1, bias=0)
                - "neutral": Use first speaker as neutral
        """
        super().__init__()

        self.model_name = model_name
        self.emotion_dim = emotion_dim
        self.num_speakers = num_speakers
        self.integration_mode = integration_mode
        self.use_adaptive_norm = use_adaptive_norm
        self.inference_mode = inference_mode

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        print(f"Loading base model: {model_name}...")
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Get hidden size
        self.hidden_size = self.base_model.config.hidden_size

        # Speaker-conditioned emotion projection layer
        self.emotion_projection = SpeakerConditionedProjection(
            emotion_dim=emotion_dim,
            hidden_size=self.hidden_size,
            num_speakers=num_speakers,
            num_layers=2,
            use_adaptive_norm=use_adaptive_norm,
            inference_mode=inference_mode
        )

        # Apply LoRA if specified
        if use_lora:
            if lora_config is None:
                lora_config = {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "task_type": "CAUSAL_LM"
                }

            print("Applying LoRA...")
            self.base_model = prepare_model_for_kbit_training(self.base_model)
            lora_config_obj = LoraConfig(**lora_config)
            self.base_model = get_peft_model(self.base_model, lora_config_obj)

        print(f"✓ Student model with SAN initialized")
        print(f"  - Integration mode: {integration_mode}")
        print(f"  - Adaptive normalization: {use_adaptive_norm}")
        print(f"  - Inference mode: {inference_mode}")
        print(f"  - Number of speakers: {num_speakers}")

    def integrate_emotion_concat(
        self,
        input_embeds: torch.Tensor,
        emotion_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Integrate emotion embeddings via concatenation.

        Args:
            input_embeds: [batch_size, seq_len, hidden_size]
            emotion_embeds: [batch_size, hidden_size] (already projected)
            attention_mask: [batch_size, seq_len]

        Returns:
            Combined embeddings and updated attention mask
        """
        batch_size, seq_len, _ = input_embeds.shape

        # Emotion embeddings are already projected, just add dimension
        projected_emotion = emotion_embeds.unsqueeze(1)  # [batch, 1, hidden]

        # Concatenate at the beginning of sequence
        combined_embeds = torch.cat([projected_emotion, input_embeds], dim=1)

        # Update attention mask
        if attention_mask is not None:
            emotion_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            combined_mask = torch.cat([emotion_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        return combined_embeds, combined_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        emotion_embeddings: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with emotion embedding integration and speaker conditioning.

        Args:
            input_ids: [batch_size, seq_len]
            emotion_embeddings: [batch_size, emotion_dim] (optional)
            speaker_ids: [batch_size] speaker IDs for training (optional)
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] for training
            **kwargs: Additional arguments for the base model

        Returns:
            Model outputs with loss if labels are provided
        """
        # Get input embeddings from base model
        if hasattr(self.base_model, 'get_input_embeddings'):
            embed_layer = self.base_model.get_input_embeddings()
        else:
            # For PEFT models
            embed_layer = self.base_model.base_model.get_input_embeddings()

        input_embeds = embed_layer(input_ids)

        # Integrate emotion embeddings if provided
        if emotion_embeddings is not None:
            # Project emotion embeddings with speaker conditioning
            # During training: uses speaker_ids
            # During inference: speaker_ids=None triggers inference_mode
            projected_emotion = self.emotion_projection(
                emotion_embeddings,
                speaker_ids=speaker_ids
            )

            if self.integration_mode == "concat":
                input_embeds, attention_mask = self.integrate_emotion_concat(
                    input_embeds, projected_emotion, attention_mask
                )
                # Update labels for concatenation
                if labels is not None:
                    batch_size = labels.shape[0]
                    emotion_labels = torch.full(
                        (batch_size, 1),
                        fill_value=-100,  # Ignore emotion token in loss
                        dtype=labels.dtype,
                        device=labels.device
                    )
                    labels = torch.cat([emotion_labels, labels], dim=1)
            else:
                raise NotImplementedError(f"Integration mode {self.integration_mode} not supported with SAN")

        # Forward through base model
        outputs = self.base_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        emotion_embeddings: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        **generation_kwargs
    ):
        """
        Generate text with emotion embedding conditioning.

        During inference, speaker_ids should be None to use inference_mode.

        Args:
            input_ids: [batch_size, seq_len]
            emotion_embeddings: [batch_size, emotion_dim]
            speaker_ids: [batch_size] (should be None during inference)
            attention_mask: [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            **generation_kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        # Get input embeddings
        if hasattr(self.base_model, 'get_input_embeddings'):
            embed_layer = self.base_model.get_input_embeddings()
        else:
            embed_layer = self.base_model.base_model.get_input_embeddings()

        input_embeds = embed_layer(input_ids)

        # Integrate emotion embeddings
        if emotion_embeddings is not None:
            projected_emotion = self.emotion_projection(
                emotion_embeddings,
                speaker_ids=speaker_ids
            )

            if self.integration_mode == "concat":
                input_embeds, attention_mask = self.integrate_emotion_concat(
                    input_embeds, projected_emotion, attention_mask
                )

        # Generate with the modified embeddings
        outputs = self.base_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generation_kwargs
        )

        return outputs


# Example usage
if __name__ == "__main__":
    # Initialize student model with SAN
    model = StudentModelWithSAN(
        model_name="Qwen/Qwen2.5-1.5B",
        emotion_dim=256,
        num_speakers=4,
        integration_mode="concat",
        use_lora=True,
        load_in_4bit=True,
        use_adaptive_norm=True,
        inference_mode="average"
    )

    # Example forward pass (training)
    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    emotion_embeddings = torch.randn(batch_size, 256)
    speaker_ids = torch.tensor([0, 1])  # Different speakers
    attention_mask = torch.ones(batch_size, seq_len)
    labels = input_ids.clone()

    # Training mode
    model.train()
    outputs_train = model(
        input_ids=input_ids,
        emotion_embeddings=emotion_embeddings,
        speaker_ids=speaker_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    print(f"Training loss: {outputs_train.loss.item()}")
    print(f"Training logits shape: {outputs_train.logits.shape}")

    # Inference mode (no speaker_ids)
    model.eval()
    with torch.no_grad():
        outputs_infer = model(
            input_ids=input_ids,
            emotion_embeddings=emotion_embeddings,
            speaker_ids=None,  # Triggers inference mode
            attention_mask=attention_mask,
            labels=labels
        )

    print(f"Inference loss: {outputs_infer.loss.item()}")
    print(f"Loss difference (should be small): {abs(outputs_train.loss.item() - outputs_infer.loss.item()):.4f}")
