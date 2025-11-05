"""
Utility functions for LangPert.
"""

import json
import re
import numpy as np
from typing import List, Dict, Optional, Any, Tuple


def extract_genes_from_output(response: str) -> Tuple[List[str], Optional[str]]:
    """Extract kNN genes and reasoning from LLM response.

    Args:
        response: Raw LLM response text

    Returns:
        Tuple of (gene_list, reasoning_text)
    """
    # Case 1: Response is already valid JSON
    try:
        data = json.loads(response.strip())
        if 'kNN' in data:
            return data.get('kNN', []), data.get('reasoning')
    except json.JSONDecodeError:
        pass

    # Case 2: JSON in code blocks (markdown format) or XML-style tags
    json_pattern = r'(?:```(?:json)?\s*((?:\{|\[).*?(?:\}|\]))(?:\n|$)\s*```|<json>\s*((?:\{|\[).*?(?:\}|\]))\s*</json>)'
    matches = re.findall(json_pattern, response, re.DOTALL)

    for match in matches:
        # match is a tuple (markdown_group, xml_group), use whichever is non-empty
        json_str = match[0] if match[0] else match[1]
        if not json_str:
            continue
        try:
            data = json.loads(json_str.strip())
            if 'kNN' in data:
                return data.get('kNN', []), data.get('reasoning')
        except json.JSONDecodeError:
            continue

    # Case 3: Find JSON-like structures in text
    json_pattern = r'\{[^{}]*"kNN"[^{}]*\}'
    matches = re.findall(json_pattern, response, re.DOTALL)

    for match in matches:
        try:
            cleaned = match.strip()
            data = json.loads(cleaned)
            if 'kNN' in data:
                return data.get('kNN', []), data.get('reasoning')
        except json.JSONDecodeError:
            continue

    # Case 4: Extract from array patterns like ["Gene1", "Gene2"]
    array_pattern = r'\["[^"]+"\s*(?:,\s*"[^"]+"\s*)*\]'
    matches = re.findall(array_pattern, response)

    for match in matches:
        try:
            genes = json.loads(match)
            if genes and isinstance(genes, list):
                return genes, None
        except json.JSONDecodeError:
            continue

    return [], None


def calculate_knn_mean(knn_genes: List[str], obs_mean: Dict[str, np.ndarray],
                      fallback_value: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate mean expression profile from k-nearest neighbor genes.

    Args:
        knn_genes: List of gene names to average
        obs_mean: Dictionary mapping gene names to expression profiles
        fallback_value: Fallback expression profile if no valid genes found

    Returns:
        Mean expression profile across valid kNN genes

    Raises:
        ValueError: If no valid genes found and no fallback provided
    """
    values = [obs_mean[gene] for gene in knn_genes if gene in obs_mean]

    if not values:
        if fallback_value is not None:
            return fallback_value
        raise ValueError("No valid genes found to calculate mean")

    print(f"Calculating mean over {len(values)} genes: {[g for g in knn_genes if g in obs_mean]}")

    return np.mean(values, axis=0)


def validate_gene_list(genes: List[str], available_genes: List[str]) -> List[str]:
    """Validate and filter gene list against available genes.

    Args:
        genes: List of gene names to validate
        available_genes: List of available gene names

    Returns:
        Filtered list containing only valid genes
    """
    available_set = set(available_genes)
    valid_genes = [gene for gene in genes if gene in available_set]

    if len(valid_genes) != len(genes):
        invalid = [gene for gene in genes if gene not in available_set]
        print(f"Warning: {len(invalid)} genes not found in available set: {invalid}")

    return valid_genes