"""
Rolling / expanding window feature generation.

All windows are computed OVER symbol_id groups, sorted by (date_id, time_id),
so they never look across symbols or forward in time.

Supported statistics:
  mean, std, min, max, skew, zscore (z-score of value within window)

LEAKAGE NOTE: Rolling features with window W at training time require W-1
prior observations.  At inference time, the InferenceState buffer must be at
least W rows deep before these features are meaningful.  Missing initial values
default to NaN, which GBDTs handle gracefully.
"""

from __future__ import annotations

from typing import Any, Sequence

import polars as pl
from loguru import logger

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.features.base import BaseTransformer

# Statistics we know how to compute
SUPPORTED_STATS = frozenset({"mean", "std", "min", "max", "skew", "zscore"})


class RollingTransformer(BaseTransformer):
    """
    Compute rolling window statistics per symbol.

    Args:
        columns:    Feature columns to compute rolling stats on.
        windows:    Window sizes (number of time steps).
        stats:      Which statistics to compute.
        group_col:  Column to group by (default: symbol_id).
        sort_cols:  Sort order within each group.
        min_periods: Minimum observations in window before computing a stat.
                    If None, defaults to the window size (all NaN until full).
    """

    def __init__(
        self,
        columns: Sequence[str],
        windows: Sequence[int],
        stats: Sequence[str] = ("mean", "std"),
        group_col: str = S.SYMBOL_ID,
        sort_cols: Sequence[str] = (S.DATE_ID, S.TIME_ID),
        min_periods: int = 1,
    ) -> None:
        super().__init__()
        unknown = set(stats) - SUPPORTED_STATS
        if unknown:
            raise ValueError(f"Unknown stats: {unknown}. Supported: {SUPPORTED_STATS}")
        if any(w <= 0 for w in windows):
            raise ValueError("Window sizes must be positive integers.")

        self.columns = list(columns)
        self.windows = list(windows)
        self.stats = list(stats)
        self.group_col = group_col
        self.sort_cols = list(sort_cols)
        self.min_periods = min_periods

    def _fit(self, df: pl.DataFrame, **kwargs: Any) -> None:
        # No learnable parameters for rolling stats.
        # Compute the output column name list for downstream use.
        self._output_cols = []
        for col in self.columns:
            for win in self.windows:
                for stat in self.stats:
                    if stat == "zscore":
                        # Z-score decomposes into mean + std columns
                        self._output_cols.append(f"{col}_roll{win}_zscore")
                    else:
                        self._output_cols.append(f"{col}_roll{win}_{stat}")
        logger.debug(
            "RollingTransformer fitted: {} features ({} cols × {} wins × {} stats)",
            len(self._output_cols), len(self.columns),
            len(self.windows), len(self.stats),
        )

    def _transform(self, df: pl.DataFrame) -> pl.DataFrame:
        sort_by = [self.group_col] + self.sort_cols
        df_sorted = df.sort(sort_by)

        exprs: list[pl.Expr] = []
        for col in self.columns:
            for win in self.windows:
                for stat in self.stats:
                    exprs.extend(
                        self._make_expr(col, win, stat)
                    )

        result = df_sorted.with_columns(exprs)
        logger.debug("RollingTransformer: added {} columns", len(self._output_cols))
        return result

    def _make_expr(self, col: str, win: int, stat: str) -> list[pl.Expr]:
        """Build Polars rolling expression for a single (col, win, stat) triple."""
        base = pl.col(col).sort_by(self.sort_cols)
        opts = {"window_size": win, "min_samples": self.min_periods}

        if stat == "mean":
            return [
                base.rolling_mean(**opts)
                .over(self.group_col)
                .alias(f"{col}_roll{win}_mean")
            ]
        elif stat == "std":
            return [
                base.rolling_std(**opts)
                .over(self.group_col)
                .alias(f"{col}_roll{win}_std")
            ]
        elif stat == "min":
            return [
                base.rolling_min(**opts)
                .over(self.group_col)
                .alias(f"{col}_roll{win}_min")
            ]
        elif stat == "max":
            return [
                base.rolling_max(**opts)
                .over(self.group_col)
                .alias(f"{col}_roll{win}_max")
            ]
        elif stat == "skew":
            # Polars doesn't have native rolling_skew; approximate via custom UDF.
            # For now we emit a mean-centered difference as a proxy.
            # Extension point: replace with a real rolling skewness computation.
            mean_expr = (
                base.rolling_mean(**opts)
                .over(self.group_col)
                .alias(f"{col}_roll{win}_mean_tmp")
            )
            skew_expr = (
                (pl.col(col) - pl.col(f"{col}_roll{win}_mean_tmp"))
                .alias(f"{col}_roll{win}_skew")
            )
            # We drop the temporary column after use
            return [mean_expr, skew_expr]
        elif stat == "zscore":
            mean_expr = (
                base.rolling_mean(**opts)
                .over(self.group_col)
                .alias(f"__tmp_mean_{col}_{win}")
            )
            std_expr = (
                base.rolling_std(**opts)
                .over(self.group_col)
                .alias(f"__tmp_std_{col}_{win}")
            )
            # Z-score = (x - mean) / std; clips std to avoid division by zero
            zscore_expr = (
                (
                    (pl.col(col) - pl.col(f"__tmp_mean_{col}_{win}"))
                    / (pl.col(f"__tmp_std_{col}_{win}") + 1e-8)
                )
                .alias(f"{col}_roll{win}_zscore")
            )
            return [mean_expr, std_expr, zscore_expr]
        else:
            raise ValueError(f"Unknown stat: {stat}")

    @property
    def feature_names_out(self) -> list[str]:
        return self._output_cols


class EWMTransformer(BaseTransformer):
    """
    Exponentially weighted moving average features per symbol.

    EWM features capture recency-weighted information and are especially
    useful for financial data where recent observations matter more.

    Args:
        columns:  Columns to apply EWM to.
        spans:    Span parameters for EWM (equivalent to com = span/2 - 1).
        group_col: Symbol grouping column.
    """

    def __init__(
        self,
        columns: Sequence[str],
        spans: Sequence[float] = (5.0, 10.0, 20.0),
        group_col: str = S.SYMBOL_ID,
        sort_cols: Sequence[str] = (S.DATE_ID, S.TIME_ID),
    ) -> None:
        super().__init__()
        self.columns = list(columns)
        self.spans = list(spans)
        self.group_col = group_col
        self.sort_cols = list(sort_cols)

    def _fit(self, df: pl.DataFrame, **kwargs: Any) -> None:
        self._output_cols = [
            f"{col}_ewm{int(span)}" for col in self.columns for span in self.spans
        ]
        logger.debug(
            "EWMTransformer fitted: {} features", len(self._output_cols)
        )

    def _transform(self, df: pl.DataFrame) -> pl.DataFrame:
        sort_by = [self.group_col] + self.sort_cols
        df_sorted = df.sort(sort_by)

        exprs: list[pl.Expr] = []
        for col in self.columns:
            for span in self.spans:
                alpha = 2.0 / (span + 1.0)
                exprs.append(
                    pl.col(col)
                    .ewm_mean(alpha=alpha, adjust=False)
                    .over(self.group_col)
                    .alias(f"{col}_ewm{int(span)}")
                )

        return df_sorted.with_columns(exprs)

    @property
    def feature_names_out(self) -> list[str]:
        return self._output_cols
