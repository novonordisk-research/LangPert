"""
Base backend interface for LangPert.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseBackend(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text response from the given prompt.

        Args:
            prompt: Input prompt for the LLM
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text response
        """
        pass

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts. Override for batched inference."""
        return [self.generate_text(prompt, **kwargs) for prompt in prompts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config})"