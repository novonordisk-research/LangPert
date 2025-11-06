"""Evaluation metrics for perturbation predictions."""

import numpy as np
import pandas as pd
from typing import Mapping

from perturbdict import PerturbDict


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Pearson correlation, returning 0.0 for NaN"""
    corr = np.corrcoef(x, y)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def calculate_eval_metrics(
    predictions: Mapping[str, np.ndarray],
    observed: Mapping[str, np.ndarray],
    pert_dict: PerturbDict,
) -> pd.DataFrame:
    """Compute evaluation metrics (MAE, MSE, correlation, direction accuracy, DE gene overlap)."""
    metrics = []
    ctrl_mean = pert_dict.get_ctrl_mean()

    for pert_name, pred in predictions.items():
        obs = observed[pert_name]

        # Calculate deltas (change from control)
        pred_delta = pred - ctrl_mean
        obs_delta = obs - ctrl_mean

        # Get masks for top differentially expressed genes
        mask_top20 = pert_dict.get_de_mask(pert_name, k=20)
        mask_top50 = pert_dict.get_de_mask(pert_name, k=50)
        mask_top100 = pert_dict.get_de_mask(pert_name, k=100)

        # Calculate overlap of top DE genes
        pred_top20_idx = np.argsort(np.abs(pred_delta))[-20:]
        pred_top100_idx = np.argsort(np.abs(pred_delta))[-100:]
        obs_top20_idx = np.where(mask_top20)[0]
        obs_top100_idx = np.where(mask_top100)[0]

        metrics.append({
            "perturbation": pert_name,
            # Mean Absolute Error
            "mae": float(np.mean(np.abs(pred - obs))),
            "mae_top20": float(np.mean(np.abs(pred[mask_top20] - obs[mask_top20]))),
            # Mean Squared Error
            "mse": float(np.mean((pred - obs) ** 2)),
            "mse_top20": float(np.mean((pred[mask_top20] - obs[mask_top20]) ** 2)),
            # Pearson Correlation
            "cor": safe_pearsonr(pred_delta, obs_delta),
            "cor_top20": safe_pearsonr(pred_delta[mask_top20], obs_delta[mask_top20]),
            "cor_top50": safe_pearsonr(pred_delta[mask_top50], obs_delta[mask_top50]),
            "cor_top100": safe_pearsonr(pred_delta[mask_top100], obs_delta[mask_top100]),
            # Direction Accuracy
            "frac_correct_direction": float(np.mean(pred_delta * obs_delta > 0)),
            "frac_correct_direction_top20": float(
                np.mean(pred_delta[mask_top20] * obs_delta[mask_top20] > 0)
            ),
            # Effect Sizes
            "effect_size": float(np.mean(obs_delta**2)),
            "effect_size_top20": float(np.mean(obs_delta[mask_top20] ** 2)),
            # Top DE Gene Overlap
            "overlap_20": float(len(np.intersect1d(pred_top20_idx, obs_top20_idx)) / 20.0),
            "overlap_100": float(len(np.intersect1d(pred_top100_idx, obs_top100_idx)) / 100.0),
        })

    return pd.DataFrame(metrics)
