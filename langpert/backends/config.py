"""
Configuration schemas for different backends.
"""

from pydantic import BaseModel
from typing import Optional


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI-compatible API backends."""
    api_key: str
    base_url: Optional[str] = None  # For Azure/custom endpoints
    model: str = "gpt-4"
    temperature: float = 0.2
    timeout: int = 240
    max_retries: int = 20
    max_tokens: Optional[int] = 8192


class TransformersConfig(BaseModel):
    """Configuration for local HuggingFace transformers."""
    model_name: str = "google/txgemma-2b-predict"
    quantization: bool = True
    device_map: str = "auto"
    cache_dir: Optional[str] = None
    max_new_tokens: int = 8192
    temperature: float = 0.2
    do_sample: bool = True
    pad_token_id: Optional[int] = None


class UnslothConfig(BaseModel):
    """Configuration for Unsloth backend."""
    model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    device_map: str = "auto"
    cache_dir: Optional[str] = None
    max_new_tokens: int = 8192
    temperature: float = 0.2
    do_sample: bool = True
    top_p: float = 0.9


class GeminiConfig(BaseModel):
    """Configuration for Google Gemini API backend."""
    api_key: str
    model: str = "gemini-1.5-pro"
    temperature: float = 0.2
    timeout: int = 240
    max_retries: int = 20
    max_output_tokens: Optional[int] = 8192
    top_p: Optional[float] = None
    top_k: Optional[int] = None
