"""
Time-series backtest evaluation.

Given a trained model and a held-out test set (or OOF predictions), computes
comprehensive performance metrics and generates diagnostic plots.

The backtest follows a strict temporal protocol:
  - Only uses the model as a black box (predict function)
  - Never retrains or adjusts on the test set
  - Reports per-date metrics to expose time-varying performance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.modeling.metrics import (
    compute_all_metrics,
    competition_score,
)


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""

    overall_metrics: dict[str, float]
    per_date_metrics: pl.DataFrame  # Columns: date_id, competition_score, weighted_r2, ...
    predictions: pl.DataFrame        # Columns: date_id, symbol_id, y_true, y_pred, weight

    def summary(self) -> str:
        lines = ["=== Backtest Summary ==="]
        for k, v in self.overall_metrics.items():
            lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions.write_parquet(output_dir / "predictions.parquet")
        self.per_date_metrics.write_parquet(output_dir / "per_date_metrics.parquet")

        import json
        with open(output_dir / "overall_metrics.json", "w") as f:
            json.dump(self.overall_metrics, f, indent=2)

        logger.info("Backtest results saved to {}", output_dir)


def run_backtest(
    predict_fn,
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str = S.TARGET_COL,
    weight_col: str = S.WEIGHT,
    date_col: str = S.DATE_ID,
) -> BacktestResult:
    """
    Run a backtest: call predict_fn on df, compute all metrics.

    Args:
        predict_fn:   Callable that takes a DataFrame and returns np.ndarray.
        df:           Evaluation DataFrame.
        feature_cols: Feature columns used by predict_fn.
        target_col:   Ground truth column.
        weight_col:   Sample weight column.
        date_col:     Date grouping column.

    Returns:
        BacktestResult with overall and per-date metrics.
    """
    logger.info("Running backtest on {} rows", len(df))

    y_pred = predict_fn(df)
    y_true = df[target_col].to_numpy()
    weights = df[weight_col].to_numpy()
    dates = df[date_col].to_numpy()

    overall_metrics = compute_all_metrics(y_true, y_pred, weights, dates)

    # Per-date metrics
    per_date_rows = []
    for d in np.unique(dates):
        mask = dates == d
        d_metrics = compute_all_metrics(
            y_true[mask], y_pred[mask], weights[mask], dates[mask]
        )
        d_metrics[date_col] = int(d)
        per_date_rows.append(d_metrics)

    per_date_df = pl.DataFrame(per_date_rows).sort(date_col)

    predictions_df = df.select([date_col, S.SYMBOL_ID]).with_columns([
        pl.Series("y_true", y_true),
        pl.Series("y_pred", y_pred),
        df[weight_col],
    ])

    logger.info(overall_metrics)
    return BacktestResult(
        overall_metrics=overall_metrics,
        per_date_metrics=per_date_df,
        predictions=predictions_df,
    )


def compare_models(
    results: dict[str, BacktestResult],
    metric: str = "competition_score",
) -> pl.DataFrame:
    """
    Build a comparison table of multiple backtest results.

    Args:
        results: Dict mapping experiment_id → BacktestResult.
        metric:  Primary metric for sorting.

    Returns:
        Polars DataFrame with one row per model, sorted by metric descending.
    """
    rows = []
    for exp_id, result in results.items():
        row = {"experiment": exp_id}
        row.update(result.overall_metrics)
        rows.append(row)

    df = pl.DataFrame(rows).sort(metric, descending=True)
    return df
