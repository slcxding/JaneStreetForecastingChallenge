"""
Report generation — produces JSON and markdown summaries of evaluation results.

Keeps reporting logic separate from compute logic so each can evolve
independently.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from loguru import logger

from janestreet_forecasting.evaluation.backtest import BacktestResult
from janestreet_forecasting.evaluation.diagnostics import (
    compute_error_by_symbol,
    plot_feature_importance,
    plot_residuals_over_time,
)


def generate_evaluation_report(
    result: BacktestResult,
    feature_importance: dict[str, float] | None,
    output_dir: Path,
    experiment_id: str = "experiment",
) -> None:
    """
    Generate a full evaluation report including metrics, plots, and tables.

    Args:
        result:             BacktestResult from run_backtest().
        feature_importance: Optional feature importance dict.
        output_dir:         Directory to write report files.
        experiment_id:      Name prefix for files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save raw data
    result.save(output_dir)

    # 2. Feature importance plot
    if feature_importance:
        plot_feature_importance(
            feature_importance,
            top_n=30,
            output_path=output_dir / "feature_importance.png",
            title=f"{experiment_id} — Feature Importance",
        )

    # 3. Per-date performance plot
    if len(result.per_date_metrics) > 0:
        plot_residuals_over_time(
            result.per_date_metrics,
            output_path=output_dir / "per_date_score.png",
        )

    # 4. Per-symbol error table
    if "symbol_id" in result.predictions.columns:
        symbol_errors = compute_error_by_symbol(result.predictions)
        symbol_errors.write_parquet(output_dir / "symbol_errors.parquet")
        # Write top-20 worst symbols to markdown
        _write_symbol_table(symbol_errors.head(20), output_dir / "worst_symbols.md")

    # 5. Markdown summary
    _write_markdown_summary(result, experiment_id, output_dir)

    logger.info("Evaluation report written to {}", output_dir)


def _write_markdown_summary(
    result: BacktestResult,
    experiment_id: str,
    output_dir: Path,
) -> None:
    lines = [
        f"# Evaluation Report: {experiment_id}",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in result.overall_metrics.items():
        lines.append(f"| {k} | {v:.4f} |")

    lines += [
        "",
        "## Files",
        "- `predictions.parquet` — full prediction table",
        "- `per_date_metrics.parquet` — per-date metric breakdown",
        "- `overall_metrics.json` — metrics as JSON",
        "- `feature_importance.png` — feature importance chart",
        "- `per_date_score.png` — per-date competition score",
        "- `symbol_errors.parquet` — per-symbol error breakdown",
    ]

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(lines))


def _write_symbol_table(df: pl.DataFrame, path: Path) -> None:
    lines = ["# Worst Symbols by RMSE", "", "| symbol_id | n_rows | rmse | mae |", "|-----------|--------|------|-----|"]
    for row in df.iter_rows(named=True):
        lines.append(
            f"| {row['symbol_id']} | {row['n_rows']} | {row['rmse']:.4f} | {row['mae']:.4f} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines))
