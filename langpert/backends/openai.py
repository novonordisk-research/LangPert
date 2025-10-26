"""
OpenAI API backend for LangPert.

Supports OpenAI, Azure OpenAI, and any OpenAI-compatible APIs.
"""

import time
from typing import Optional
from openai import OpenAI

from .base import BaseBackend


class OpenAIBackend(BaseBackend):
    """Backend for OpenAI and OpenAI-compatible APIs (Azure OpenAI, etc.)."""

    def __init__(self, api_key: str, base_url: Optional[str] = None,
                 model: str = "gpt-4", temperature: float = 0.2,
                 timeout: int = 240, max_retries: int = 20, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, model=model,
                         temperature=temperature, timeout=timeout,
                         max_retries=max_retries, **kwargs)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries
        )
        self.model = model
        self.temperature = temperature

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None,
                     temperature: Optional[float] = None, verbose: bool = False, **kwargs) -> str:
        """Generate text using OpenAI-compatible API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            verbose: If True, print the formatted messages for debugging
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Print formatted messages if verbose mode is enabled
        if verbose:
            print(f"\nFinal Formatted Messages (OpenAI API):")
            print(f"{'-'*40}")
            for msg in messages:
                print(f"{msg['role'].upper()}: {msg['content']}")
                print(f"{'-'*20}")
            print(f"{'-'*40}")

        # Remove verbose from kwargs as a safety measure (it's already an explicit parameter)
        kwargs.pop('verbose', None)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            **kwargs
        )

        # Rate limiting (from original code)
        time.sleep(1)

        return completion.choices[0].message.content