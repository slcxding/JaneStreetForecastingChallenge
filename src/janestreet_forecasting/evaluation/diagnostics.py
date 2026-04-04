"""
Diagnostic utilities for understanding model behaviour.

Functions:
  - plot_feature_importance: Bar chart of top-N features by gain
  - plot_residuals_over_time: Residual magnitude per date (detect regime shifts)
  - plot_prediction_vs_actual: Scatter + density plot
  - compute_error_by_symbol: Per-symbol error breakdown
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger


def plot_feature_importance(
    feature_importance: dict[str, float],
    top_n: int = 30,
    output_path: Path | None = None,
    title: str = "Feature Importance (Mean Gain)",
) -> None:
    """
    Horizontal bar chart of the top-N most important features.

    Args:
        feature_importance: Dict mapping feature name → importance score.
        top_n:              Number of features to display.
        output_path:        If provided, save the plot here.
        title:              Plot title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping feature importance plot")
        return

    sorted_items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top = sorted_items[:top_n]
    names = [x[0] for x in top][::-1]  # Reverse for bottom-up bar chart
    values = [x[1] for x in top][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n // 3)))
    ax.barh(range(len(names)), values, color="steelblue", alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean Gain", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Feature importance plot saved to {}", output_path)
    else:
        plt.show()

    plt.close(fig)


def plot_residuals_over_time(
    per_date_metrics: pl.DataFrame,
    metric_col: str = "competition_score",
    output_path: Path | None = None,
) -> None:
    """
    Time series plot of per-date metric values.

    Useful for detecting:
      - Regime shifts (sudden drops in score)
      - Systematic trends (model improving/degrading over time)
      - Date-specific outliers
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping residuals plot")
        return

    dates = per_date_metrics["date_id"].to_numpy()
    scores = per_date_metrics[metric_col].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, scores, lw=0.8, alpha=0.7, label="Per-date score")

    # Rolling average
    window = min(20, len(scores) // 5)
    if window > 1:
        rolling = np.convolve(scores, np.ones(window) / window, mode="valid")
        ax.plot(dates[window - 1:], rolling, lw=2, color="red", label=f"Rolling({window})")

    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("Date ID")
    ax.set_ylabel(metric_col)
    ax.set_title(f"Per-Date {metric_col}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Residuals plot saved to {}", output_path)
    else:
        plt.show()

    plt.close(fig)


def compute_error_by_symbol(
    predictions_df: pl.DataFrame,
    symbol_col: str = "symbol_id",
) -> pl.DataFrame:
    """
    Compute per-symbol error statistics.

    Helps identify instruments the model struggles with or excels at.

    Returns:
        DataFrame with columns: symbol_id, n_rows, rmse, mae, mean_pred, mean_true.
    """
    return (
        predictions_df
        .with_columns([
            ((pl.col("y_pred") - pl.col("y_true")) ** 2).alias("sq_err"),
            (pl.col("y_pred") - pl.col("y_true")).abs().alias("abs_err"),
        ])
        .group_by(symbol_col)
        .agg([
            pl.len().alias("n_rows"),
            pl.col("sq_err").mean().sqrt().alias("rmse"),
            pl.col("abs_err").mean().alias("mae"),
            pl.col("y_pred").mean().alias("mean_pred"),
            pl.col("y_true").mean().alias("mean_true"),
        ])
        .sort("rmse", descending=True)
    )
