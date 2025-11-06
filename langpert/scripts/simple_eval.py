"""Simple evaluation script for LangPert.

Usage:
  # Run with default config (configs/config.yaml + configs/backend/unsloth.yaml)
  python -m langpert.scripts.simple_eval

  # Override specific settings
  python -m langpert.scripts.simple_eval backend=openai dataset.split_fold=1

  # Switch backends
  python -m langpert.scripts.simple_eval backend=transformers

  # Save outputs
  python -m langpert.scripts.simple_eval eval.save_metrics=true eval.output_dir=my_results
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from tqdm import tqdm

import langpert
from langpert.backends import openai_backend, transformers_backend, unsloth_backend
from langpert.prompts.loader import resolve_prompt_template

# Eval dependencies (optional)
try:
    from perturbdict import PerturbDict
    from langpert.scripts.eval_metrics import calculate_eval_metrics
except ImportError as e:
    raise ImportError(
        f"Missing eval dependencies. Install with: pip install langpert[eval] ({e})"
    ) from e
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from pathlib import Path
import wandb


logger = logging.getLogger(__name__)


@dataclass
class BackendConfig:
    """Backend configuration - edit for different models."""
    name: str = "unsloth"  # or "openai" or "transformers"
    model_name: Optional[str] = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
    temperature: float = 0.2
    # OpenAI specific
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # Local model specific
    cache_dir: Optional[str] = "/tmp/hf_cache"
    max_seq_length: int = 8192
    load_in_4bit: bool = True


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    data_path: str = "/path/to/data/replogle_k562_essential/pert_dict.pkl"
    split_k: int = 5
    split_fold: int = 0
    split_seed: int = 42
    subsample: Optional[int] = None  # Set to N to only eval N perturbations
    subsample_seed: int = 1


@dataclass
class EvalConfig:
    """Evaluation settings."""
    k_range: str = "3-5"
    prompt_template: str = "default"
    system_prompt: str = "default"
    verbose: bool = False
    save_predictions: bool = False
    save_metrics: bool = False
    output_dir: str = "outputs/eval"
    # Performance settings
    batch_size: Optional[int] = 4  # Set to enable batch inference (e.g., 4, 8, 16)
    num_workers: int = 1  # For future use
    # Weights & Biases tracking
    use_wandb: bool = False
    wandb_project: Optional[str] = "langpert-eval"
    wandb_entity: Optional[str] = None  # Your wandb username/team
    wandb_run_name: Optional[str] = None  # Auto-generated if None


@dataclass
class Config:
    """Top-level config combining all settings."""
    backend: BackendConfig = field(default_factory=BackendConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def create_backend(cfg: BackendConfig):
    """Create a LangPert backend based on configuration."""
    if cfg.name == "openai":
        return openai_backend(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.model_name or "gpt-4o-mini",
            temperature=cfg.temperature,
        )
    elif cfg.name == "unsloth":
        return unsloth_backend(
            model_name=cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            temperature=cfg.temperature,
            cache_dir=cfg.cache_dir,
        )
    elif cfg.name == "transformers":
        return transformers_backend(
            model_name=cfg.model_name,
            temperature=cfg.temperature,
            max_seq_length=cfg.max_seq_length,
            cache_dir=cfg.cache_dir,
        )
    else:
        raise ValueError(f"Unknown backend: {cfg.name}")


def run_evaluation(cfg: Config):
    """Execute the evaluation pipeline."""

    # ─── Initialize Wandb (Optional) ───
    wandb_run = None
    if cfg.eval.use_wandb:
        # Automatic config tracking - wandb logs everything!
        config_dict = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else {
            "backend": vars(cfg.backend),
            "dataset": vars(cfg.dataset),
            "eval": vars(cfg.eval),
        }

        wandb_run = wandb.init(
            project=cfg.eval.wandb_project,
            entity=cfg.eval.wandb_entity,
            name=cfg.eval.wandb_run_name,
            config=config_dict,
        )
        logger.info("Initialized wandb run: %s", wandb_run.name)

    # ─── Load Data ───
    logger.info("Loading data from %s", cfg.dataset.data_path)
    pert_dict = PerturbDict().load(cfg.dataset.data_path)
    train_data, test_data = pert_dict.get_split_data(
        k=cfg.dataset.split_k,
        fold=cfg.dataset.split_fold,
        seed=cfg.dataset.split_seed,
    )

    # Optional subsampling for quick experiments
    if cfg.dataset.subsample:
        logger.info("Subsampling %d perturbations", cfg.dataset.subsample)
        np.random.seed(cfg.dataset.subsample_seed)
        selected = np.random.choice(
            list(test_data.keys()),
            size=min(cfg.dataset.subsample, len(test_data)),
            replace=False,
        )
        test_data = {k: test_data[k] for k in selected}

    # ─── Create Model ───
    logger.info("Creating %s backend with model: %s", cfg.backend.name, cfg.backend.model_name)
    backend = create_backend(cfg.backend)

    # Resolve prompt template (handles file paths, template names, or raw strings)
    resolved_prompt = resolve_prompt_template(cfg.eval.prompt_template)

    model = langpert.LangPert(
        backend=backend,
        observed_effects=train_data,
        fallback_mean=np.zeros(len(pert_dict.gene_names)),
        prompt_template=resolved_prompt,
        system_prompt=cfg.eval.system_prompt,
    )

    # ─── Run Predictions ───
    logger.info("Predicting %d perturbations...", len(test_data))
    
    # Use batch inference if batch_size is specified
    if cfg.eval.batch_size and cfg.eval.batch_size > 1:
        logger.info("Using batch inference with batch_size=%d", cfg.eval.batch_size)
        batch_results = model.predict_batch(
            list(test_data.keys()),
            k_range=cfg.eval.k_range,
            batch_size=cfg.eval.batch_size,
        )
        predictions = {gene: result.prediction for gene, result in batch_results.items()}
    else:
        # Sequential processing
        predictions = {}
        for pert_name in tqdm(test_data.keys(), desc="Predicting"):
            result = model.predict_perturbation(
                pert_name,
                k_range=cfg.eval.k_range,
                verbose=cfg.eval.verbose,
            )
            predictions[pert_name] = result.prediction

    # ─── Compute Metrics ───
    logger.info("Computing metrics...")
    metrics = calculate_eval_metrics(predictions, test_data, pert_dict)

    # ─── Log to Wandb ───
    if cfg.eval.use_wandb and wandb_run is not None:
        # Log key aggregate metrics
        key_metrics = {
            "mae_top20": metrics["mae_top20"].mean(),
            "cor_top20": metrics["cor_top20"].mean(),
            "cor_top50": metrics["cor_top50"].mean(),
            "cor_top100": metrics["cor_top100"].mean(),
            "frac_correct_direction_top20": metrics["frac_correct_direction_top20"].mean(),
            "overlap_20": metrics["overlap_20"].mean(),
            "n_perturbations": len(metrics),
        }
        wandb.log(key_metrics)

        # Log the full per-perturbation metrics table
        wandb.log({"per_perturbation_metrics": wandb.Table(dataframe=metrics)})

        # Also set as run summary for easy comparison across runs
        for key, value in key_metrics.items():
            wandb.run.summary[key] = value

        logger.info("Logged metrics to wandb")

    # ─── Save Outputs (Optional) ───
    if cfg.eval.save_predictions or cfg.eval.save_metrics:
        output_dir = Path(cfg.eval.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.eval.save_predictions:
            preds_df = pd.DataFrame.from_dict(predictions, orient="index")
            preds_df.index.name = "perturbation"
            preds_path = output_dir / "predictions.csv"
            preds_df.to_csv(preds_path)
            logger.info("Saved predictions to %s", preds_path)

        if cfg.eval.save_metrics:
            metrics_path = output_dir / "metrics.csv"
            metrics.to_csv(metrics_path, index=False)
            logger.info("Saved metrics to %s", metrics_path)

    # ─── Display Results ───
    print("\n" + "="*70)
    print(f"EVALUATION RESULTS: {cfg.backend.name} / {cfg.backend.model_name}")
    print("="*70)
    print(f"\nMean metrics across {len(metrics)} perturbations:")
    print(metrics.mean(numeric_only=True).to_string())
    print("\nPer-perturbation metrics (first 5):")
    print(metrics.head().to_string())

    # ─── Finish Wandb ───
    if cfg.eval.use_wandb and wandb_run is not None:
        wandb.finish()
        logger.info("Finished wandb run")

    return metrics


def main(cfg: Optional[Config] = None):
    """Main entry point - can be called programmatically or via CLI.

    If cfg is None, uses dataclass defaults. In practice, you should always
    provide a config either via Hydra or by calling this function directly.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # If no config provided, use dataclass defaults
    if cfg is None:
        cfg = Config()
        logger.warning(
            "No config provided - using dataclass defaults. "
            "Update defaults in the dataclass definitions or pass a config."
        )

    return run_evaluation(cfg)


# ─── Hydra Integration ───
@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def hydra_main(cfg: DictConfig):
    """Hydra-decorated main for config file support.

    The config_path is relative to this file: langpert/scripts/simple_eval.py
    So ../configs points to langpert/configs/ directory.
    """
    # Convert OmegaConf to dataclass
    config = OmegaConf.to_object(cfg)
    if not isinstance(config, Config):
        # If using dict configs, convert to Config
        config = Config(
            backend=BackendConfig(**cfg.backend),
            dataset=DatasetConfig(**cfg.dataset),
            eval=EvalConfig(**cfg.eval),
        )
    return main(config)


if __name__ == "__main__":
    hydra_main()
