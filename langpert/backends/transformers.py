"""
HuggingFace Transformers backend for LangPert.
"""

from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from .base import BaseBackend


class TransformersBackend(BaseBackend):
    """Backend for local HuggingFace transformers models."""

    def __init__(self, model_name: str = "google/txgemma-2b-predict",
                 quantization: bool = True, device_map: str = "auto",
                 cache_dir: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_name, quantization=quantization,
                         device_map=device_map, cache_dir=cache_dir, **kwargs)

        self.model_name = model_name
        self.quantization = quantization
        self.device_map = device_map

        # Use safe cache directory if none provided
        if cache_dir is None:
            from ..cache_utils import get_safe_cache_dir
            cache_dir = get_safe_cache_dir()
        self.cache_dir = cache_dir
        self.generation_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", 8192),
            "temperature": kwargs.get("temperature", 0.2),
            "do_sample": kwargs.get("do_sample", True),
            "pad_token_id": kwargs.get("pad_token_id"),
        }

        self._load_model()

    def _is_large_model(self) -> bool:
        """Determine if model needs quantization based on name."""
        return any(size in self.model_name.lower()
                  for size in ["7b", "9b", "13b", "30b", "70b"])

    def _load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )

        # Configure quantization for large models
        model_kwargs = {
            "device_map": self.device_map,
            "cache_dir": self.cache_dir,
        }

        if self.quantization and self._is_large_model():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.generation_config["pad_token_id"] = self.tokenizer.eos_token_id

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None,
                     **kwargs) -> str:
        """Generate text using local transformers model.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt (model-dependent formatting)
            **kwargs: Override generation parameters

        Returns:
            Generated text response
        """
        # Format prompt (model-specific logic can be added here)
        if system_prompt and "chat" in self.model_name.lower():
            # For chat models, format appropriately
            formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            formatted_prompt = prompt

        # Merge generation config
        gen_config = {**self.generation_config, **kwargs}

        # Generate
        outputs = self.pipe(
            formatted_prompt,
            **gen_config,
            return_full_text=False
        )

        return outputs[0]["generated_text"].strip()