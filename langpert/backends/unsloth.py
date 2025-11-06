"""
Unsloth backend for LangPert.
"""
from typing import Optional, List, Dict, Any

from .base import BaseBackend


class UnslothBackend(BaseBackend):
    """Backend using Unsloth FastLanguageModel (works with any supported model)."""

    def __init__(
        self,
        model_name: str = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            cache_dir=cache_dir,
            **kwargs,
        )

        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
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
            "top_p": kwargs.get("top_p", 0.9),
        }

        self._load_model()

    def _load_model(self):
        try:
            from unsloth import FastLanguageModel
        except Exception as e:
            raise ImportError(
                "unsloth is required for UnslothBackend. Install with `pip install unsloth` or use extras `pip install .[unsloth]`."
            ) from e

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            device_map=self.device_map,
            cache_dir=self.cache_dir,
        )

        # Set pad token if not present (required for batch inference)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        FastLanguageModel.for_inference(self.model)
        print(f"✓ Enabled Unsloth fast inference for {self.model_name}")

    def _format_as_chat(self, prompt: str, system_prompt: Optional[str]) -> Optional[List[Dict[str, str]]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, verbose: bool = False, **kwargs) -> str:
        try:
            import torch
        except Exception as e:
            raise ImportError("torch is required for UnslothBackend") from e

        gen_config = {**self.generation_config, **kwargs}

        messages = self._format_as_chat(prompt, system_prompt)

        input_ids = None
        formatted_text = None
        try:
            if hasattr(self.tokenizer, "apply_chat_template") and messages is not None:
                # First get the formatted text for verbose logging
                if verbose:
                    try:
                        formatted_text = self.tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                    except Exception:
                        pass

                # Then get the tokenized version
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
        except Exception:
            input_ids = None

        if input_ids is None:
            text_prompt = prompt if not system_prompt else f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            tokenized = self.tokenizer(text_prompt, return_tensors="pt")
            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask

            # Show fallback format for verbose mode
            if verbose:
                print(f"\nFinal Formatted Prompt (fallback format):")
                print(f"{'-'*40}")
                print(text_prompt)
                print(f"{'-'*40}")
        else:
            # For chat template case, we need to create attention mask
            attention_mask = torch.ones_like(input_ids)

            # Show chat template format for verbose mode
            if verbose and formatted_text:
                print(f"\nFinal Formatted Prompt (chat template):")
                print(f"{'-'*40}")
                print(formatted_text)
                print(f"{'-'*40}")

        if hasattr(self.model, "device"):
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_config.get("max_new_tokens", 8192),
                temperature=gen_config.get("temperature", 0.2),
                do_sample=gen_config.get("do_sample", True),
                top_p=gen_config.get("top_p", 0.9),
                pad_token_id=self.tokenizer.eos_token_id if getattr(self.tokenizer, "eos_token_id", None) is not None else None,
            )

        start = input_ids.shape[1]
        if outputs.ndim == 2:
            new_tokens = outputs[0, start:]
        else:
            new_tokens = outputs[start:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return generated_text.strip()

    def generate_batch(self, prompts: List[str], system_prompt: Optional[str] = None, **kwargs) -> List[str]:
        """Generate text for multiple prompts in parallel using batched inference."""

        try:
            import torch
        except Exception as e:
            raise ImportError("torch is required for UnslothBackend") from e

        gen_config = {**self.generation_config, **kwargs}

        # Format all prompts as chat messages
        all_messages = [self._format_as_chat(prompt, system_prompt) for prompt in prompts]

        # Set left padding for decoder-only models (required for correct batch generation)
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        # Tokenize with automatic padding
        inputs = self.tokenizer.apply_chat_template(
            all_messages,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
        )

        # Move to device and generate
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=gen_config.get("max_new_tokens", 8192),
                temperature=gen_config.get("temperature", 0.2),
                do_sample=gen_config.get("do_sample", True),
                top_p=gen_config.get("top_p", 0.9),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode outputs (excluding input tokens)
        generated_texts = self.tokenizer.batch_decode(
            outputs[:, inputs.shape[1]:],
            skip_special_tokens=True
        )

        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side

        return [text.strip() for text in generated_texts]
