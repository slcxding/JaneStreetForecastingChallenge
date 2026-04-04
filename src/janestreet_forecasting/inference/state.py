"""
InferenceState — per-symbol rolling buffer for real-time feature computation.

This is the most critical piece for correct real-time inference.

Problem:
  Rolling features (e.g. 10-period rolling mean of feature_00 for symbol X)
  require the last 10 observations for symbol X at every time step.  In batch
  training we compute these by sorting the full DataFrame.  At inference time
  (one batch per time_id) we have no history — it must be maintained as state.

Solution:
  InferenceState keeps a fixed-size ring buffer of past observations for each
  symbol.  On every call to `update(batch)`, new rows are appended to the
  buffer.  On every call to `compute_features(batch)`, rolling stats are
  computed from the buffer and appended to the batch as new columns.

  This exactly mirrors the offline feature engineering and guarantees that
  inference features match training features.

Key invariants:
  1. `update()` is called AFTER `compute_features()` for each batch.
     (We compute features from past history, then update history with current.)
  2. Buffer size must be >= max rolling window used in feature engineering.
  3. When the buffer is shorter than the window, rolling stats are NaN —
     the model (GBDT) handles this gracefully.

Thread safety: not thread-safe by design.  The Kaggle inference environment
is single-threaded.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Sequence

import numpy as np
import polars as pl
from loguru import logger

from janestreet_forecasting.data import schemas as S


class InferenceState:
    """
    Maintains a per-symbol rolling history buffer for online feature engineering.

    Args:
        buffer_size:    Maximum number of past time steps to retain per symbol.
        feature_cols:   Feature columns to buffer (for rolling stats).
        rolling_windows: Window sizes to compute (must be <= buffer_size).
        rolling_stats:  Statistics to compute per window.
        lag_cols:       Columns to compute additional lags from (beyond lag_0).
        extra_lags:     Additional lag values to compute (e.g. [1, 2] for lags 1, 2).
    """

    def __init__(
        self,
        buffer_size: int = 30,
        feature_cols: Sequence[str] = (),
        rolling_windows: Sequence[int] = (5, 10, 20),
        rolling_stats: Sequence[str] = ("mean", "std"),
        lag_cols: Sequence[str] = (),
        extra_lags: Sequence[int] = (),
    ) -> None:
        if rolling_windows and buffer_size < max(rolling_windows):
            raise ValueError(
                f"buffer_size={buffer_size} is smaller than max rolling window "
                f"{max(rolling_windows)}.  Increase buffer_size or reduce rolling_windows."
            )

        self.buffer_size = buffer_size
        self.feature_cols = list(feature_cols)
        self.rolling_windows = list(rolling_windows)
        self.rolling_stats = list(rolling_stats)
        self.lag_cols = list(lag_cols)
        self.extra_lags = list(extra_lags)

        # Per-symbol deques — auto-evict oldest entries at buffer_size
        # Key: symbol_id (int), Value: deque of row dicts
        self._buffers: dict[int, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.buffer_size)
        )

        self._n_updates: int = 0  # Total batches processed

    def update(self, batch: pl.DataFrame) -> None:
        """
        Append the current batch to each symbol's history buffer.

        Call this AFTER compute_features() for the same batch.

        Args:
            batch: Current time step's rows (one row per symbol).
        """
        cols_to_buffer = list(
            {S.SYMBOL_ID, S.DATE_ID, S.TIME_ID}
            | set(self.feature_cols)
            | set(self.lag_cols)
        )
        # Only buffer columns that exist in the batch
        cols_to_buffer = [c for c in cols_to_buffer if c in batch.columns]

        for row in batch.select(cols_to_buffer).iter_rows(named=True):
            symbol = row[S.SYMBOL_ID]
            self._buffers[symbol].append(row)

        self._n_updates += 1

    def compute_features(self, batch: pl.DataFrame) -> pl.DataFrame:
        """
        Compute rolling and lag features for the current batch.

        Rolling stats are computed over [buffer history] + [current row],
        matching the offline RollingTransformer which includes the current row.
        Call update() AFTER this method to add current rows to the buffer.

        Args:
            batch: Current time step's rows.

        Returns:
            batch with additional feature columns added.
        """
        if not self.feature_cols and not self.lag_cols:
            return batch

        # Build a symbol → current row lookup for O(1) access
        current_rows: dict[int, dict[str, Any]] = {
            row[S.SYMBOL_ID]: row
            for row in batch.iter_rows(named=True)
        }

        # Compute new columns for each row in the batch
        new_col_data: dict[str, list[Any]] = defaultdict(list)

        for symbol in batch[S.SYMBOL_ID].to_list():
            history = list(self._buffers[symbol])  # Past rows, oldest first
            current = current_rows.get(symbol, {})

            # Rolling statistics: buffer + current row (inclusive, mirrors offline)
            for col in self.feature_cols:
                col_history = np.array(
                    [r.get(col, np.nan) for r in history], dtype=np.float32
                )
                current_val = np.float32(current.get(col, np.nan))
                col_with_current = np.append(col_history, current_val)

                for win in self.rolling_windows:
                    window_vals = (
                        col_with_current[-win:]
                        if len(col_with_current) >= win
                        else col_with_current
                    )
                    for stat in self.rolling_stats:
                        key = f"{col}_roll{win}_{stat}"
                        valid = window_vals[~np.isnan(window_vals)]
                        if len(valid) == 0:
                            new_col_data[key].append(np.nan)
                        elif stat == "mean":
                            new_col_data[key].append(float(np.mean(valid)))
                        elif stat == "std":
                            new_col_data[key].append(
                                float(np.std(valid)) if len(valid) > 1 else np.nan
                            )
                        elif stat == "min":
                            new_col_data[key].append(float(np.min(valid)))
                        elif stat == "max":
                            new_col_data[key].append(float(np.max(valid)))

            # Additional lag features from history
            for lag_col in self.lag_cols:
                col_history = [r.get(lag_col, np.nan) for r in history]
                for lag in self.extra_lags:
                    key = f"{lag_col}_lag{lag}"
                    idx = -(lag + 1)  # -1 = most recent, -2 = one before, etc.
                    if len(col_history) >= abs(idx):
                        val = col_history[idx]
                        new_col_data[key].append(float(val) if val is not None else np.nan)
                    else:
                        new_col_data[key].append(np.nan)

        if not new_col_data:
            return batch

        # Append new columns to the batch
        new_cols = [
            pl.Series(name, vals, dtype=pl.Float32)
            for name, vals in new_col_data.items()
        ]
        return batch.with_columns(new_cols)

    def warm_up(self, historical_df: pl.DataFrame) -> None:
        """
        Pre-populate buffers from historical data before the inference loop starts.

        This avoids NaN features in the first N time steps (where N = buffer_size).
        Warm-up uses the last `buffer_size` time steps before the inference period.

        Args:
            historical_df: Historical data sorted by (date_id, time_id).
        """
        # Get the last buffer_size unique time steps
        unique_times = (
            historical_df.select([S.DATE_ID, S.TIME_ID])
            .unique()
            .sort([S.DATE_ID, S.TIME_ID])
            .tail(self.buffer_size)
        )

        warm_df = historical_df.join(unique_times, on=[S.DATE_ID, S.TIME_ID])

        # Process each time step in order
        for (date_id, time_id), group in (
            warm_df.sort([S.DATE_ID, S.TIME_ID])
            .group_by([S.DATE_ID, S.TIME_ID], maintain_order=True)
        ):
            self.update(group)

        logger.info(
            "InferenceState warmed up: {} symbols, {} time steps",
            len(self._buffers), self._n_updates,
        )

    def reset(self) -> None:
        """Clear all buffers. Call before starting a new inference session."""
        self._buffers.clear()
        self._n_updates = 0

    @property
    def n_symbols(self) -> int:
        return len(self._buffers)

    @property
    def n_updates(self) -> int:
        return self._n_updates
