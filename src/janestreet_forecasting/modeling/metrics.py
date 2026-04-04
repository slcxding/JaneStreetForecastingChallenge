"""
Evaluation metrics for the Jane Street competition.

Primary metric: sum of per-date weighted Pearson correlations.

Formula (per the competition):
  For each date t:
    ρ_t = Σ_i(w_i * r_i * p_i) / sqrt(Σ_i(w_i * r_i²) × Σ_i(w_i * p_i²))

  Final score = Σ_t ρ_t

Where:
  w_i = sample weight for observation i
  r_i = actual responder_6 value
  p_i = predicted value

This is equivalent to the weighted Pearson correlation per date summed
across all dates.  A perfect score = number of unique dates.

Offline we work with the mean (score / n_dates) to get a per-date figure.

Secondary metrics (for diagnostics):
  - Weighted R² (sklearn-style, for comparison with literature)
  - RMSE, MAE
"""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

from janestreet_forecasting.data import schemas as S


def competition_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    dates: np.ndarray,
) -> float:
    """
    Compute the competition's primary metric: sum of per-date weighted correlations.

    Args:
        y_true:  Actual responder_6 values, shape (N,).
        y_pred:  Predicted values, shape (N,).
        weights: Sample weights, shape (N,).
        dates:   date_id for each observation, shape (N,).

    Returns:
        Scalar competition score. Higher is better.
    """
    unique_dates = np.unique(dates)
    total = 0.0

    for d in unique_dates:
        mask = dates == d
        w = weights[mask]
        r = y_true[mask]
        p = y_pred[mask]

        numerator = np.sum(w * r * p)
        denom = np.sqrt(np.sum(w * r ** 2) * np.sum(w * p ** 2))

        if denom < 1e-10:
            # Degenerate date (all predictions or actuals are zero)
            continue

        total += numerator / denom

    return float(total)


def mean_competition_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    dates: np.ndarray,
) -> float:
    """
    Per-date average of the competition score.

    Normalises by the number of dates so the metric doesn't scale with the
    number of validation dates.  More comparable across different val set sizes.
    """
    n_dates = len(np.unique(dates))
    return competition_score(y_true, y_pred, weights, dates) / max(n_dates, 1)


def competition_score_from_df(
    df: pl.DataFrame,
    pred_col: str = "prediction",
    target_col: str = S.TARGET_COL,
    weight_col: str = S.WEIGHT,
    date_col: str = S.DATE_ID,
) -> float:
    """
    Convenience wrapper: compute competition score from a Polars DataFrame.

    The DataFrame must contain prediction, target, weight, and date columns.
    """
    return competition_score(
        y_true=df[target_col].to_numpy(),
        y_pred=df[pred_col].to_numpy(),
        weights=df[weight_col].to_numpy(),
        dates=df[date_col].to_numpy(),
    )


def weighted_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted R² (coefficient of determination).

    This is a commonly used diagnostic metric and matches sklearn's convention.
    Note: this is NOT the competition metric — use for diagnostics only.

    R² = 1 - Σ(w * (y - ŷ)²) / Σ(w * (y - ȳ)²)
    """
    weighted_mean = np.average(y_true, weights=weights)
    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_true - weighted_mean) ** 2)

    if ss_tot < 1e-10:
        return 0.0

    return float(1.0 - ss_res / ss_tot)


def weighted_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted root mean squared error."""
    return float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=weights)))


def weighted_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted mean absolute error."""
    return float(np.average(np.abs(y_true - y_pred), weights=weights))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    dates: np.ndarray,
) -> dict[str, float]:
    """
    Compute all metrics and return as a dict.  Convenient for logging.

    Returns:
        Dict with keys: competition_score, mean_competition_score,
        weighted_r2, weighted_rmse, weighted_mae.
    """
    metrics = {
        "competition_score": competition_score(y_true, y_pred, weights, dates),
        "mean_competition_score": mean_competition_score(y_true, y_pred, weights, dates),
        "weighted_r2": weighted_r2(y_true, y_pred, weights),
        "weighted_rmse": weighted_rmse(y_true, y_pred, weights),
        "weighted_mae": weighted_mae(y_true, y_pred, weights),
    }
    return metrics


def lgbm_competition_metric(y_pred: np.ndarray, dataset) -> tuple[str, float, bool]:
    """
    Custom LightGBM metric for the competition score.

    LightGBM expects a function with signature:
      (y_pred, dataset) -> (metric_name, value, is_higher_better)

    Args:
        y_pred:  Predictions from the model.
        dataset: LightGBM Dataset with label, weight, and group info.

    Note: LightGBM doesn't natively pass date_id through the Dataset, so we
    use a simplified weighted Pearson correlation across all rows (not per date).
    The per-date metric is computed in the full evaluation pipeline.
    """
    y_true = dataset.get_label()
    weights = dataset.get_weight()

    if weights is None:
        weights = np.ones_like(y_true)

    # Simplified: treat all rows as one group
    num = np.sum(weights * y_true * y_pred)
    denom = np.sqrt(np.sum(weights * y_true ** 2) * np.sum(weights * y_pred ** 2))
    score = num / max(denom, 1e-10)

    return "competition_score", float(score), True  # True = higher is better
