"""
Core LangPert model for cell perturbation predictions using hybrid LLM-kNN approach.
"""

import json
import numpy as np
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from tqdm import tqdm

from ..backends.base import BaseBackend
from ..prompts.loader import format_prompt, resolve_system_prompt
from ..utils import extract_genes_from_output, calculate_knn_mean, validate_gene_list


@dataclass
class PredictionResult:
    """Result from LangPert prediction."""
    gene: str
    knn_genes: List[str]
    prediction: np.ndarray
    llm_response: str
    reasoning: Optional[str] = None


class LangPert:
    """
    LangPert: Hybrid LLM-kNN model for cell perturbation predictions.
    """

    def __init__(self, backend: BaseBackend,
                 observed_effects: Dict[str, np.ndarray],
                 prompt_template: str = "default",
                 fallback_mean: Optional[np.ndarray] = None,
                 system_prompt: Optional[str] = "default"):
        """
        Initialize LangPert model.

        Args:
            backend: LLM backend for gene similarity queries
            observed_effects: Dict mapping gene names to observed perturbation effects
            prompt_template: Name of prompt template to use
            fallback_mean: Fallback prediction when no valid kNN genes found
            system_prompt: System prompt for the LLM. Can be a template name (e.g., "default", "biologist")
                          or a custom prompt string. Defaults to "default" template.
        """
        self.backend = backend
        self.observed_effects = observed_effects
        self.prompt_template = prompt_template
        self.fallback_mean = fallback_mean
        # Resolve system prompt (handles both template names and direct strings)
        self.system_prompt = resolve_system_prompt(system_prompt)

        # Cache available genes for validation
        self.available_genes = list(observed_effects.keys())

        # Track last printed verbose info to avoid repetition
        self._last_verbose_template = None
        self._last_verbose_system_prompt = None

    def _prepare_candidates(self, target_gene: str, candidate_genes: Optional[List[str]] = None) -> List[str]:
        """Prepare and validate candidate genes for a target gene."""
        if candidate_genes is None:
            candidates = [g for g in self.available_genes if g != target_gene]
        else:
            candidates = validate_gene_list(candidate_genes, self.available_genes)
            candidates = [g for g in candidates if g != target_gene]

        if not candidates:
            raise ValueError(f"No valid candidate genes available for {target_gene}")

        return sorted(candidates)

    def _process_llm_response(self, gene: str, llm_response: str) -> PredictionResult:
        """Process LLM response into a PredictionResult."""
        knn_genes, reasoning = extract_genes_from_output(llm_response)
        valid_knn_genes = validate_gene_list(knn_genes, self.available_genes)

        if not valid_knn_genes:
            if self.fallback_mean is None:
                raise ValueError(f"No valid kNN genes found for {gene} and no fallback provided")
            prediction = self.fallback_mean.copy()
        else:
            prediction = calculate_knn_mean(
                valid_knn_genes,
                self.observed_effects,
                self.fallback_mean
            )

        return PredictionResult(
            gene=gene,
            knn_genes=valid_knn_genes,
            prediction=prediction,
            llm_response=llm_response,
            reasoning=reasoning
        )

    def predict_perturbation(self, target_gene: str,
                      candidate_genes: Optional[List[str]] = None,
                      prompt_template: Optional[str] = None,
                      k_range: Optional[str] = None,
                      verbose: bool = False,
                      **llm_kwargs) -> PredictionResult:
        """
        Predict perturbation effect for a single gene.

        Args:
            target_gene: Gene to predict perturbation effect for
            candidate_genes: List of candidate genes for kNN selection.
                           If None, uses all available genes except target.
            prompt_template: Override default prompt template
            k_range: Number or range of neighbors to find (e.g., "5-10", "10", "3-5").
                    Defaults to "5-10" if not specified.
            verbose: If True, print the LLM prompt and response for debugging
            **llm_kwargs: Additional arguments passed to LLM backend

        Returns:
            PredictionResult containing prediction and metadata
        """
        # Prepare candidate genes
        candidate_genes = self._prepare_candidates(target_gene, candidate_genes)

        # Format prompt
        template = prompt_template or self.prompt_template
        prompt = format_prompt(template, target_gene, candidate_genes, k_range=k_range)

        # Print prompt template info only once (or when changed)
        if verbose:
            if (self._last_verbose_template != template or
                self._last_verbose_system_prompt != self.system_prompt):
                print(f"\n{'='*60}")
                print(f"VERBOSE MODE: Configuration")
                print(f"{'='*60}")
                print(f"Prompt Template: {template}")
                print(f"System Prompt: {self.system_prompt}")
                print(f"\nExample Formatted Prompt (for {target_gene}):")
                print(f"{'-'*40}")
                print(prompt)
                print(f"{'-'*40}")
                print(f"{'='*60}\n")
                # Update tracking
                self._last_verbose_template = template
                self._last_verbose_system_prompt = self.system_prompt

        # Query LLM for similar genes
        if not verbose:
            print(f"Querying LLM for gene similarities to {target_gene}")
        else:
            print(f"\n[{target_gene}] Querying LLM...")

        llm_response = self.backend.generate_text(
            prompt,
            system_prompt=self.system_prompt,
            verbose=False,
            **llm_kwargs
        )

        # Print LLM response if verbose mode is enabled
        if verbose:
            print(f"[{target_gene}] LLM Response:")
            print(f"{'-'*40}")
            print(llm_response)
            print(f"{'-'*40}")
            print()

        # Process response into prediction result
        result = self._process_llm_response(target_gene, llm_response)

        # Print extracted genes in verbose mode
        if verbose and result.knn_genes:
            print(f"[{target_gene}] Extracted kNN genes: {', '.join(result.knn_genes)}")

        return result

    def predict_batch(self, target_genes: List[str],
                     candidate_genes: Optional[List[str]] = None,
                     prompt_template: Optional[str] = None,
                     k_range: Optional[str] = None,
                     batch_size: Optional[int] = None,
                     **llm_kwargs) -> Dict[str, PredictionResult]:
        """
        Predict perturbation effects for multiple genes.

        Args:
            target_genes: List of genes to predict
            candidate_genes: List of candidate genes for kNN selection
            prompt_template: Override default prompt template
            k_range: Number or range of neighbors to find (e.g., "5-10", "10", "3-5").
                    Defaults to "5-10" if not specified.
            batch_size: Number of genes to process in parallel. If None, processes sequentially.
            **llm_kwargs: Additional arguments passed to LLM backend

        Returns:
            Dict mapping gene names to PredictionResult objects
        """
        results = {}

        # Sequential processing when batch_size is None or 1
        if batch_size is None or batch_size <= 1:
            for gene in tqdm(target_genes, desc="Predicting genes"):
                try:
                    results[gene] = self.predict_perturbation(
                        gene,
                        candidate_genes=candidate_genes,
                        prompt_template=prompt_template,
                        k_range=k_range,
                        **llm_kwargs
                    )
                except Exception as e:
                    tqdm.write(f"Error processing {gene}: {e}")
            return results

        # Batched processing
        template = prompt_template or self.prompt_template

        for i in tqdm(range(0, len(target_genes), batch_size), desc="Predicting genes (batched)"):
            batch_genes = target_genes[i:i+batch_size]
            batch_prompts = []
            batch_valid_genes = []

            # Prepare prompts for this batch
            for gene in batch_genes:
                try:
                    gene_candidates = self._prepare_candidates(gene, candidate_genes)
                    prompt = format_prompt(template, gene, gene_candidates, k_range=k_range)
                    batch_prompts.append(prompt)
                    batch_valid_genes.append(gene)
                except ValueError as e:
                    tqdm.write(f"Warning: {e}, skipping")
                    continue

            if not batch_prompts:
                continue

            # Generate responses for batch
            try:
                llm_responses = self.backend.generate_batch(
                    batch_prompts,
                    system_prompt=self.system_prompt,
                    **llm_kwargs
                )

                # Process each response
                for gene, llm_response in zip(batch_valid_genes, llm_responses):
                    try:
                        results[gene] = self._process_llm_response(gene, llm_response)
                    except ValueError as e:
                        tqdm.write(f"Warning: {e}, skipping")

            except Exception as e:
                tqdm.write(f"Error in batch generation: {e}")

        return results

    def get_predictions_matrix(self, results: Dict[str, PredictionResult]) -> np.ndarray:
        """
        Convert prediction results to matrix format.

        Args:
            results: Dict of prediction results from predict_batch

        Returns:
            Tuple of (matrix, gene_names) where matrix has genes as rows
            and features as columns, and gene_names preserves row order.
        """
        if not results:
            return np.array([]), []

        # Get prediction dimension from first result
        first_pred = next(iter(results.values())).prediction
        n_features = len(first_pred)
        n_genes = len(results)

        # Initialize matrix
        matrix = np.zeros((n_genes, n_features))
        gene_names = []

        for i, (gene, result) in enumerate(results.items()):
            matrix[i] = result.prediction
            gene_names.append(gene)

        return matrix, gene_names

    def update_observed_effects(self, new_effects: Dict[str, np.ndarray]):
        """
        Update the observed effects database.

        Args:
            new_effects: Dict of new gene -> effect mappings to add
        """
        self.observed_effects.update(new_effects)
        self.available_genes = list(self.observed_effects.keys())
        print(f"Updated observed effects. Now have {len(self.available_genes)} genes.")

    def get_available_genes(self) -> List[str]:
        """Get list of genes with observed perturbation effects."""
        return self.available_genes.copy()

    def __repr__(self) -> str:
        return (f"LangPert(backend={self.backend.__class__.__name__}, "
               f"n_genes={len(self.available_genes)}, "
               f"prompt_template='{self.prompt_template}')")
