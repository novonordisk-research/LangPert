"""
Prompt loading utilities.
"""

from typing import List, Optional
from pathlib import Path
from .templates import PROMPT_TEMPLATES
from .system_prompts import SYSTEM_PROMPT_TEMPLATES


def load_prompt(template_name: str = "default") -> str:
    """Load a prompt template by name.

    Args:
        template_name: Name of the prompt template to load

    Returns:
        Prompt template string

    Raises:
        KeyError: If template_name is not found
    """
    if template_name not in PROMPT_TEMPLATES:
        raise KeyError(f"Unknown prompt template: {template_name}. "
                      f"Available templates: {list(PROMPT_TEMPLATES.keys())}")

    return PROMPT_TEMPLATES[template_name]


def list_available_prompts() -> List[str]:
    """List all available prompt template names.

    Returns:
        List of available prompt template names
    """
    return list(PROMPT_TEMPLATES.keys())


def load_system_prompt(template_name: str = "default") -> str:
    """Load a system prompt template by name.

    Args:
        template_name: Name of the system prompt template to load

    Returns:
        System prompt template string

    Raises:
        KeyError: If template_name is not found
    """
    if template_name not in SYSTEM_PROMPT_TEMPLATES:
        raise KeyError(f"Unknown system prompt template: {template_name}. "
                      f"Available templates: {list(SYSTEM_PROMPT_TEMPLATES.keys())}")

    return SYSTEM_PROMPT_TEMPLATES[template_name]


def list_available_system_prompts() -> List[str]:
    """List all available system prompt template names.

    Returns:
        List of available system prompt template names
    """
    return list(SYSTEM_PROMPT_TEMPLATES.keys())


def resolve_prompt_template(template: str) -> str:
    """Resolve template from file path, template name, or raw string."""
    path = Path(template)
    if path.suffix == '.txt' or path.is_file():
        return path.read_text().strip()
    return PROMPT_TEMPLATES.get(template, template)


def resolve_system_prompt(system_prompt: Optional[str]) -> Optional[str]:
    """Resolve system prompt from file path, template name, or raw string."""
    if system_prompt is None:
        return None
    path = Path(system_prompt)
    if path.suffix == '.txt' or path.is_file():
        return path.read_text().strip()
    return SYSTEM_PROMPT_TEMPLATES.get(system_prompt, system_prompt)


def format_prompt(template_name: str, gene: str, list_of_genes: List[str],
                  k_range: Optional[str] = None) -> str:
    """Format a prompt template with gene information.

    Accepts either a registered template name or a raw template string.

    Args:
        template_name: Registered template key (e.g., "default", "minimal")
                       or a raw template string containing `{gene}` and `{list_of_genes}` placeholders.
        gene: Target gene to analyze
        list_of_genes: List of candidate genes for similarity
        k_range: Number or range of neighbors to find (e.g., "5-10", "10", "3-5").
                 Defaults to "5-10" if not specified.

    Returns:
        Formatted prompt string ready for LLM
    """
    # If a known template key is provided, load from registry; otherwise treat as raw template string
    if template_name in PROMPT_TEMPLATES:
        template = PROMPT_TEMPLATES[template_name]
    else:
        template = template_name
    genes_str = ", ".join(list_of_genes)

    # Default k_range if not provided
    if k_range is None:
        k_range = "5-10"

    return template.format(gene=gene, list_of_genes=genes_str, k_range=k_range)


def list_prompt_templates() -> None:
    """Print all available prompt templates with descriptions."""
    print("=== Available Prompt Templates ===\n")

    template_descriptions = {
        "default": "General biological similarity (with reasoning)",
        "minimal": "Simplified version of default",
        "no_reasoning": "Quick results without explanations",
        "k562": "K562 cell line specific analysis"
    }

    for template_name in PROMPT_TEMPLATES.keys():
        description = template_descriptions.get(template_name, "Custom template")
        print(f"  - {template_name}: {description}")

    print("\nUsage: model.predict_perturbation('TP53', prompt_template='default')")
