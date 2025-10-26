"""Backend factory and registry."""

from typing import Dict, Type, Any, Optional
from .base import BaseBackend
from .openai import OpenAIBackend
from .config import OpenAIConfig, TransformersConfig, UnslothConfig, GeminiConfig

# Lazy imports for optional backends
_TransformersBackend = None
_UnslothBackend = None
_GeminiBackend = None


def _get_transformers_backend():
    """Lazy import for TransformersBackend."""
    global _TransformersBackend
    if _TransformersBackend is None:
        try:
            from .transformers import TransformersBackend
            _TransformersBackend = TransformersBackend
        except ImportError as e:
            raise ImportError(
                "TransformersBackend requires additional dependencies. "
                "Install with: pip install langpert[transformers]"
            ) from e
    return _TransformersBackend


def _get_unsloth_backend():
    """Lazy import for UnslothBackend."""
    global _UnslothBackend
    if _UnslothBackend is None:
        try:
            from .unsloth import UnslothBackend
            _UnslothBackend = UnslothBackend
        except ImportError as e:
            raise ImportError(
                "UnslothBackend requires additional dependencies. "
                "Install with: pip install langpert[unsloth]"
            ) from e
    return _UnslothBackend


def _get_gemini_backend():
    """Lazy import for GeminiBackend."""
    global _GeminiBackend
    if _GeminiBackend is None:
        try:
            from .gemini import GeminiBackend
            _GeminiBackend = GeminiBackend
        except ImportError as e:
            raise ImportError(
                "GeminiBackend requires additional dependencies. "
                "Install with: pip install langpert[gemini]"
            ) from e
    return _GeminiBackend


# Backend registry with lazy loading
BACKENDS: Dict[str, Type[BaseBackend]] = {
    "api": OpenAIBackend,
    "transformers": _get_transformers_backend,
    "unsloth": _get_unsloth_backend,
    "gemini": _get_gemini_backend,
}


def create_backend(backend_type: str, **config) -> BaseBackend:
    """Create backend: "api", "transformers", "unsloth", or "gemini"."""
    if backend_type not in BACKENDS:
        raise ValueError(f"Unknown backend type: {backend_type}. "
                        f"Available backends: {list(BACKENDS.keys())}")

    backend_class_or_loader = BACKENDS[backend_type]
    
    # Handle lazy loading - if it's a function, call it to get the class
    if callable(backend_class_or_loader) and not isinstance(backend_class_or_loader, type):
        backend_class = backend_class_or_loader()
    else:
        backend_class = backend_class_or_loader
    
    return backend_class(**config)


# Convenience constructors for cleaner usage
def openai_backend(api_key: str, base_url: Optional[str] = None,
                   model: str = "gpt-4", **config) -> BaseBackend:
    """Create OpenAI-compatible backend (OpenAI, Azure, etc.)."""
    return OpenAIBackend(api_key=api_key, base_url=base_url,
                         model=model, **config)



def transformers_backend(**config) -> BaseBackend:
    """Create HuggingFace Transformers backend."""
    TransformersBackend = _get_transformers_backend()
    return TransformersBackend(**config)


def unsloth_backend(**config) -> BaseBackend:
    """Create Unsloth backend."""
    UnslothBackend = _get_unsloth_backend()
    return UnslothBackend(**config)


def gemini_backend(api_key: str, model: str = "gemini-2.5-pro", **config) -> BaseBackend:
    """Create Google Gemini backend."""
    GeminiBackend = _get_gemini_backend()
    return GeminiBackend(api_key=api_key, model=model, **config)


__all__ = [
    "BaseBackend",
    "OpenAIBackend",
    "TransformersBackend",
    "UnslothBackend",
    "GeminiBackend",
    "create_backend",
    "openai_backend",
    "transformers_backend",
    "unsloth_backend",
    "gemini_backend",
    "OpenAIConfig",
    "TransformersConfig",
    "UnslothConfig",
    "GeminiConfig"
]


def __getattr__(name: str):
    """Lazy loading for optional backend classes."""
    if name == "TransformersBackend":
        return _get_transformers_backend()
    elif name == "UnslothBackend":
        return _get_unsloth_backend()
    elif name == "GeminiBackend":
        return _get_gemini_backend()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
