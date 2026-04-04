"""
Local inference simulator.

Mimics the Kaggle API's batch-serving environment locally, enabling:
  1. Debugging the full inference pipeline before submission
  2. Evaluating whether the model degrades due to state management issues
  3. Measuring inference latency per batch

The simulator reads from a local parquet file (processed test or held-out
training data) and feeds it to the PredictorPipeline one time_id at a time,
exactly as the Kaggle API would.

Key differences from Kaggle API:
  - Kaggle provides a Python generator object; here we read from parquet
  - Kaggle's `predict` function must return a pd.DataFrame with a
    "responder_6" column; here we collect predictions in memory
  - Kaggle reveals ground truth for lag features; here we use the lag_0
    columns that were precomputed during feature engineering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.inference.predict import PredictorPipeline
from janestreet_forecasting.modeling.metrics import compute_all_metrics


@dataclass
class SimulationResult:
    """Results from a local inference simulation run."""

    n_batches: int
    total_rows: int
    predictions: pl.DataFrame   # All predictions with index columns
    metrics: dict[str, float]   # Overall metrics (requires ground truth)
    per_batch_latency_ms: list[float] = field(default_factory=list)

    @property
    def mean_latency_ms(self) -> float:
        return float(np.mean(self.per_batch_latency_ms)) if self.per_batch_latency_ms else 0.0

    def summary(self) -> str:
        lines = [
            f"Simulation complete: {self.n_batches} batches, {self.total_rows} rows",
            f"Mean latency: {self.mean_latency_ms:.1f}ms/batch",
        ]
        for k, v in self.metrics.items():
            lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)


class LocalInferenceSimulator:
    """
    Simulates the Kaggle real-time inference environment.

    Args:
        predictor:    Fitted PredictorPipeline.
        data_path:    Path to parquet file (processed test or val data).
        max_batches:  Stop after this many batches (None = all).
        batch_col:    Column to group rows into batches (default: time_id).
    """

    def __init__(
        self,
        predictor: PredictorPipeline,
        data_path: Path | str,
        max_batches: int | None = None,
        batch_col: str = S.TIME_ID,
        date_col: str = S.DATE_ID,
    ) -> None:
        self.predictor = predictor
        self.data_path = Path(data_path)
        self.max_batches = max_batches
        self.batch_col = batch_col
        self.date_col = date_col

        if not self.data_path.exists():
            raise FileNotFoundError(f"Simulation data not found at {self.data_path}")

    def run(self, output_path: Path | None = None) -> SimulationResult:
        """
        Execute the full inference simulation.

        For each batch (time_id group):
          1. Receive batch from "API" (read from file)
          2. Call predictor.predict(batch)
          3. Call predictor.update_state(batch)
          4. Record predictions and latency

        Args:
            output_path: Optional path to write prediction parquet.

        Returns:
            SimulationResult with predictions and metrics.
        """
        df = pl.read_parquet(self.data_path)
        batches = list(self._iter_batches(df))

        if self.max_batches is not None:
            batches = batches[: self.max_batches]

        all_preds: list[pl.DataFrame] = []
        latencies: list[float] = []

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Simulating {len(batches)} batches...", total=len(batches)
            )

            for batch in batches:
                import time

                t0 = time.perf_counter()
                preds = self.predictor.predict(batch)
                latency_ms = (time.perf_counter() - t0) * 1000

                self.predictor.update_state(batch)
                latencies.append(latency_ms)

                pred_df = batch.select(S.INDEX_COLS).with_columns(
                    pl.Series("prediction", preds)
                )
                # Attach ground truth if available (for offline eval)
                if S.TARGET_COL in batch.columns:
                    pred_df = pred_df.with_columns(batch[S.TARGET_COL])
                if S.WEIGHT in batch.columns:
                    pred_df = pred_df.with_columns(batch[S.WEIGHT])

                all_preds.append(pred_df)
                progress.update(task, advance=1)

        predictions_df = pl.concat(all_preds)

        # Compute metrics if ground truth is available
        metrics: dict[str, float] = {}
        if S.TARGET_COL in predictions_df.columns:
            metrics = compute_all_metrics(
                y_true=predictions_df[S.TARGET_COL].to_numpy(),
                y_pred=predictions_df["prediction"].to_numpy(),
                weights=predictions_df[S.WEIGHT].to_numpy(),
                dates=predictions_df[S.DATE_ID].to_numpy(),
            )

        result = SimulationResult(
            n_batches=len(batches),
            total_rows=len(predictions_df),
            predictions=predictions_df,
            metrics=metrics,
            per_batch_latency_ms=latencies,
        )

        logger.info(result.summary())

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_df.write_parquet(output_path)
            logger.info("Predictions written to {}", output_path)

        return result

    def _iter_batches(self, df: pl.DataFrame) -> list[pl.DataFrame]:
        """Split df into batches by (date_id, time_id) order."""
        return [
            group
            for (date_id, time_id), group in (
                df.sort([self.date_col, self.batch_col])
                .group_by([self.date_col, self.batch_col], maintain_order=True)
            )
        ]
