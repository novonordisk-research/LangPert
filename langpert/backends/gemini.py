"""
Google Gemini API backend for LangPert.
"""

import time
from typing import Optional
import google.generativeai as genai

from .base import BaseBackend


class GeminiBackend(BaseBackend):
    """Backend for Google Gemini API."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-pro",
                 temperature: float = 0.2, timeout: int = 240,
                 max_retries: int = 20, **kwargs):
        super().__init__(api_key=api_key, model=model,
                         temperature=temperature, timeout=timeout,
                         max_retries=max_retries, **kwargs)

        # Configure the API
        genai.configure(api_key=api_key)
        
        # Initialize the model
        self.model_name = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.model = genai.GenerativeModel(model)
        
        # Store additional kwargs for generation
        self.generation_config = kwargs

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None,
                     temperature: Optional[float] = None, verbose: bool = False, **kwargs) -> str:
        """Generate text using Google Gemini API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt (will be prepended to the prompt)
            temperature: Override default temperature
            verbose: If True, print the formatted prompt for debugging
            **kwargs: Additional parameters (max_output_tokens, top_p, top_k, etc.)

        Returns:
            Generated text response
        """
        # Combine system prompt with user prompt if provided
        # Gemini doesn't have a separate system message, so we prepend it
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Print formatted prompt if verbose mode is enabled
        if verbose:
            print(f"\nFinal Formatted Prompt (Gemini API):")
            print(f"{'-'*40}")
            if system_prompt:
                print(f"SYSTEM: {system_prompt}")
                print(f"{'-'*20}")
            print(f"USER: {prompt}")
            print(f"{'-'*40}")

        # Prepare generation config
        generation_config = {
            "temperature": temperature or self.temperature,
            **self.generation_config,
            **kwargs
        }
        
        # Remove verbose from generation config if it exists
        generation_config.pop('verbose', None)

        # Generate response with retry logic
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                
                # Rate limiting
                time.sleep(1)
                
                return response.text
                
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    if verbose:
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed after {self.max_retries} attempts: {last_exception}")
        
        raise Exception(f"Failed to generate text: {last_exception}")
