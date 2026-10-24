"""
GPT-OSS-120B/GPT-OSS-20B backend for LangPert.
"""

from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from transformers.distributed import DistributedConfig
import torch
from .base import BaseBackend


class GPTOSSBackend(BaseBackend):
    """Backend for local HuggingFace transformers models."""

    def __init__(self, model_name: str = "openai/gpt-oss-120b",
                 device_map: str = "auto", parallelism = False,
                 cache_dir: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_name, quantization=False,
                         device_map=device_map, cache_dir=cache_dir, **kwargs)

        self.model_name = model_name
        self.device_map = device_map
        self.parallelism = parallelism

        # Use safe cache directory if none provided
        if cache_dir is None:
            from ..cache_utils import get_safe_cache_dir
            cache_dir = get_safe_cache_dir()
        self.cache_dir = cache_dir
        self.generation_config = {
            "max_new_tokens": 5000,
            "temperature": 0.7,
            # "do_sample": kwargs.get("do_sample", True),
            "pad_token_id": kwargs.get("pad_token_id"),
        }

        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir, padding_side="left"
        )

        # Configure quantization for large models
        model_kwargs = {
            "device_map": self.device_map,
            "cache_dir": self.cache_dir,
            "torch_dtype": "auto",
        }

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            attn_implementation="kernels-community/vllm-flash-attn3",
            **model_kwargs
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            print("Pad token is not present")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.generation_config["pad_token_id"] = self.tokenizer.eos_token_id

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, verbose=False, **kwargs) -> str:
        """Generate text using local transformers model.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt (model-dependent formatting)
            **kwargs: Override generation parameters

        Returns:
            Generated text response
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
        # Format prompt (model-specific logic can be added here)
        formatted_prompt = [{"role": "user", "content": full_prompt}]

        # Merge generation config
        gen_config = {**self.generation_config, **kwargs}
        inputs = self.tokenizer.apply_chat_template(
            formatted_prompt,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            **gen_config,
        )

        return self.tokenizer.decode(outputs[0])
    
    def generate_batch(self, prompts: List[str], system_prompt, verbose=False, **kwargs) -> List[str]:
        """Generate text for multiple prompts. Override for batched inference."""
        conversations = []
        for prompt in prompts:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            conversations.append([{"role": "user", "content": full_prompt}])
        
        inputs = self.tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,      # <-- important for batching
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gen_config = {**self.generation_config, **kwargs}
    
        outputs = self.model.generate(
            **inputs,
            **gen_config,
        )
        texts = [
            self.tokenizer.decode(out) for out in outputs
        ]
        return texts
        