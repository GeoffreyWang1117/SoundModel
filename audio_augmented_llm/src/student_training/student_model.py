"""
Student model with emotion embedding integration.
Extends a pre-trained LLM with the ability to incorporate emotion embeddings.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional, Dict, Any


class EmotionProjection(nn.Module):
    """
    Projects emotion embeddings to match the hidden size of the LLM.
    """

    def __init__(
        self,
        emotion_dim: int = 256,
        hidden_size: int = 2048,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize emotion projection network.

        Args:
            emotion_dim: Dimension of input emotion embeddings
            hidden_size: Hidden size of the target LLM
            num_layers: Number of projection layers
            dropout: Dropout probability
        """
        super().__init__()

        layers = []
        current_dim = emotion_dim

        for i in range(num_layers):
            next_dim = hidden_size if i == num_layers - 1 else (emotion_dim + hidden_size) // 2
            layers.append(nn.Linear(current_dim, next_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            current_dim = next_dim

        self.projection = nn.Sequential(*layers)

    def forward(self, emotion_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project emotion embeddings to LLM hidden size.

        Args:
            emotion_embeddings: [batch_size, emotion_dim]

        Returns:
            Projected embeddings: [batch_size, hidden_size]
        """
        return self.projection(emotion_embeddings)


class StudentModelWithEmotion(nn.Module):
    """
    Student LLM model augmented with emotion embeddings.
    Supports multiple integration modes: concat, attention, or cross-attention.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        emotion_dim: int = 256,
        integration_mode: str = "concat",  # "concat", "attention", "cross_attention"
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
    ):
        """
        Initialize student model with emotion integration.

        Args:
            model_name: Name or path of the base LLM
            emotion_dim: Dimension of emotion embeddings
            integration_mode: How to integrate emotion embeddings
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_config: LoRA configuration dict
            load_in_4bit: Whether to load model in 4-bit quantization
            load_in_8bit: Whether to load model in 8-bit quantization
        """
        super().__init__()

        self.model_name = model_name
        self.emotion_dim = emotion_dim
        self.integration_mode = integration_mode

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

        # Emotion projection layer
        self.emotion_projection = EmotionProjection(
            emotion_dim=emotion_dim,
            hidden_size=self.hidden_size,
            num_layers=2
        )

        # Apply LoRA if specified
        if use_lora:
            if lora_config is None:
                lora_config = {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "task_type": "CAUSAL_LM"
                }

            print("Applying LoRA...")
            self.base_model = prepare_model_for_kbit_training(self.base_model)
            lora_config_obj = LoraConfig(**lora_config)
            self.base_model = get_peft_model(self.base_model, lora_config_obj)

        # Integration mode-specific components
        if integration_mode == "attention":
            self.emotion_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        elif integration_mode == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )

        print(f"✓ Student model initialized with {integration_mode} integration")

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
            emotion_embeds: [batch_size, emotion_dim]
            attention_mask: [batch_size, seq_len]

        Returns:
            Combined embeddings and updated attention mask
        """
        batch_size, seq_len, _ = input_embeds.shape

        # Project emotion embeddings
        projected_emotion = self.emotion_projection(emotion_embeds)  # [batch, hidden]
        projected_emotion = projected_emotion.unsqueeze(1)  # [batch, 1, hidden]

        # Concatenate at the beginning of sequence
        combined_embeds = torch.cat([projected_emotion, input_embeds], dim=1)  # [batch, seq_len+1, hidden]

        # Update attention mask
        if attention_mask is not None:
            emotion_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            combined_mask = torch.cat([emotion_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        return combined_embeds, combined_mask

    def integrate_emotion_attention(
        self,
        input_embeds: torch.Tensor,
        emotion_embeds: torch.Tensor
    ):
        """
        Integrate emotion embeddings via self-attention.

        Args:
            input_embeds: [batch_size, seq_len, hidden_size]
            emotion_embeds: [batch_size, emotion_dim]

        Returns:
            Attended embeddings
        """
        # Project emotion embeddings
        projected_emotion = self.emotion_projection(emotion_embeds)  # [batch, hidden]
        projected_emotion = projected_emotion.unsqueeze(1)  # [batch, 1, hidden]

        # Apply attention: use emotion as query, input as key/value
        attended_embeds, _ = self.emotion_attention(
            query=projected_emotion,
            key=input_embeds,
            value=input_embeds
        )  # [batch, 1, hidden]

        # Combine with input embeddings
        combined_embeds = torch.cat([attended_embeds, input_embeds], dim=1)

        return combined_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        emotion_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with emotion embedding integration.

        Args:
            input_ids: [batch_size, seq_len]
            emotion_embeddings: [batch_size, emotion_dim] (optional)
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
            if self.integration_mode == "concat":
                input_embeds, attention_mask = self.integrate_emotion_concat(
                    input_embeds, emotion_embeddings, attention_mask
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

            elif self.integration_mode == "attention":
                input_embeds = self.integrate_emotion_attention(input_embeds, emotion_embeddings)

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
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        **generation_kwargs
    ):
        """
        Generate text with emotion embedding conditioning.

        Args:
            input_ids: [batch_size, seq_len]
            emotion_embeddings: [batch_size, emotion_dim]
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
            if self.integration_mode == "concat":
                input_embeds, attention_mask = self.integrate_emotion_concat(
                    input_embeds, emotion_embeddings, attention_mask
                )
            elif self.integration_mode == "attention":
                input_embeds = self.integrate_emotion_attention(input_embeds, emotion_embeddings)

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
    # Initialize student model
    model = StudentModelWithEmotion(
        model_name="Qwen/Qwen2.5-1.5B",
        emotion_dim=256,
        integration_mode="concat",
        use_lora=True,
        load_in_4bit=True
    )

    # Example forward pass
    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    emotion_embeddings = torch.randn(batch_size, 256)
    attention_mask = torch.ones(batch_size, seq_len)
    labels = input_ids.clone()

    # Forward
    outputs = model(
        input_ids=input_ids,
        emotion_embeddings=emotion_embeddings,
        attention_mask=attention_mask,
        labels=labels
    )

    print(f"Loss: {outputs.loss.item()}")
    print(f"Logits shape: {outputs.logits.shape}")
