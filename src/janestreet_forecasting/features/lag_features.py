"""
Lag feature generation.

LEAKAGE WARNING: Every lag feature must represent information that was
*available at prediction time*. The Kaggle API provides `responder_*_lag_0`
(the previous time_id's values) but nothing further back.  Additional lags
beyond lag_0 require a rolling state buffer (see inference/state.py).

For offline training we can compute lags on the sorted training frame, but
we must be careful:
  - Sort by (symbol_id, date_id, time_id) before computing lags.
  - Use within-group shifts so lag never crosses symbol boundaries.
  - Never include future observations.

The `LagTransformer` here computes lags from the training frame directly
(safe for training).  The `InferenceState` handles lags at inference time.
"""

from __future__ import annotations

from typing import Any, Sequence

import polars as pl
from loguru import logger

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.features.base import BaseTransformer


class LagTransformer(BaseTransformer):
    """
    Compute lagged values for specified columns, grouped by symbol.

    For each (column, lag) pair, creates a new column named
    `{column}_lag{lag}` containing the value `lag` time steps earlier for the
    same symbol_id.

    Args:
        columns: Column names to lag.
        lags:    List of positive integers — how many steps back to look.
        group_col: Column to group by (default: symbol_id).
        sort_cols: Columns to sort by within each group before lagging.
    """

    def __init__(
        self,
        columns: Sequence[str],
        lags: Sequence[int],
        group_col: str = S.SYMBOL_ID,
        sort_cols: Sequence[str] = (S.DATE_ID, S.TIME_ID),
    ) -> None:
        super().__init__()
        if any(lag <= 0 for lag in lags):
            raise ValueError("All lag values must be positive integers.")
        self.columns = list(columns)
        self.lags = list(lags)
        self.group_col = group_col
        self.sort_cols = list(sort_cols)

    def _fit(self, df: pl.DataFrame, **kwargs: Any) -> None:
        # Lag transformer has no learnable parameters — fit is a no-op.
        # We still require fit() to be called so the pipeline is uniform.
        self._output_cols = [
            f"{col}_lag{lag}" for col in self.columns for lag in self.lags
        ]
        logger.debug(
            "LagTransformer fitted: {} columns × {} lags = {} new features",
            len(self.columns), len(self.lags), len(self._output_cols),
        )

    def _transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Sort by symbol + time, then compute within-group shifts.

        Important: This assumes df is the *full* contiguous training frame.
        Do NOT call this on a randomly shuffled slice — the shifts would
        compute lags across symbol boundaries, causing leakage.
        """
        sort_by = [self.group_col] + self.sort_cols
        df_sorted = df.sort(sort_by)

        lag_exprs = [
            pl.col(col)
            .shift(lag)
            .over(self.group_col)
            .alias(f"{col}_lag{lag}")
            for col in self.columns
            for lag in self.lags
        ]

        result = df_sorted.with_columns(lag_exprs)
        logger.debug("LagTransformer: added {} columns", len(self._output_cols))
        return result

    @property
    def feature_names_out(self) -> list[str]:
        return self._output_cols
