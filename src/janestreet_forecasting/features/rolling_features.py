"""
Rolling / expanding window feature generation.

All windows are computed OVER symbol_id groups, sorted by (date_id, time_id),
so they never look across symbols or forward in time.

Supported statistics:
  mean, std, min, max, zscore

REMOVED: "skew" — the previous implementation computed `x - rolling_mean`
(a deviation, not a third-moment skewness).  Mislabelled features cause
silent model bugs.  Rolling skewness requires a three-pass algorithm that
doesn't compose cleanly with Polars' streaming `.over()` expressions; the
right place to implement it is in `InferenceState` where you have explicit
access to the raw window buffer.

LEAKAGE NOTE: Rolling features with window W at row t use rows t-W+1..t.
At inference time, the InferenceState buffer must hold at least W rows of
history before these features are meaningful.  Missing initial values are
NaN — GBDTs handle this gracefully.
"""

from __future__ import annotations

from typing import Any, Sequence

import polars as pl
from loguru import logger

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.features.base import BaseTransformer

SUPPORTED_STATS = frozenset({"mean", "std", "min", "max", "zscore"})


class RollingTransformer(BaseTransformer):
    """
    Compute rolling window statistics per symbol.

    Output columns are named `{col}_roll{win}_{stat}`.  The DataFrame returned
    by transform() contains ONLY the original columns plus the new stat columns.
    No intermediate/temporary columns are exposed.

    Args:
        columns:     Feature columns to compute rolling stats on.
        windows:     Window sizes (number of time steps).
        stats:       Which statistics to compute.
        group_col:   Column to group by (default: symbol_id).
        sort_cols:   Sort order within each group.
        min_periods: Minimum non-null observations required before a value is
                     emitted.  1 = emit partial windows (recommended for
                     financial data where early NaNs are unwanted).
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
        self._output_cols = [
            f"{col}_roll{win}_{stat}"
            for col in self.columns
            for win in self.windows
            for stat in self.stats
        ]
        logger.debug(
            "RollingTransformer fitted: {} features ({} cols × {} wins × {} stats)",
            len(self._output_cols), len(self.columns),
            len(self.windows), len(self.stats),
        )

    def _transform(self, df: pl.DataFrame) -> pl.DataFrame:
        sort_by = [self.group_col] + self.sort_cols
        df_sorted = df.sort(sort_by)

        # Build all expressions.  For zscore we use struct expressions so that
        # mean and std are computed once without leaking intermediate columns.
        exprs: list[pl.Expr] = []
        for col in self.columns:
            for win in self.windows:
                exprs.extend(self._make_exprs(col, win))

        result = df_sorted.with_columns(exprs)
        logger.debug("RollingTransformer: added {} columns", len(self._output_cols))
        return result

    def _make_exprs(self, col: str, win: int) -> list[pl.Expr]:
        """Return all expressions for a single (col, win) pair."""
        base = pl.col(col).sort_by(self.sort_cols)
        opts: dict[str, Any] = {"window_size": win, "min_samples": self.min_periods}
        out: list[pl.Expr] = []

        for stat in self.stats:
            if stat == "mean":
                out.append(
                    base.rolling_mean(**opts)
                    .over(self.group_col)
                    .alias(f"{col}_roll{win}_mean")
                )
            elif stat == "std":
                out.append(
                    base.rolling_std(**opts)
                    .over(self.group_col)
                    .alias(f"{col}_roll{win}_std")
                )
            elif stat == "min":
                out.append(
                    base.rolling_min(**opts)
                    .over(self.group_col)
                    .alias(f"{col}_roll{win}_min")
                )
            elif stat == "max":
                out.append(
                    base.rolling_max(**opts)
                    .over(self.group_col)
                    .alias(f"{col}_roll{win}_max")
                )
            elif stat == "zscore":
                # Compute mean and std using map_elements on a struct so we
                # can derive both in a single pass and avoid adding temp columns.
                # We use two separate .over() expressions because Polars can
                # CSE-deduplicate them, and this is simpler and safer than
                # trying to pass a struct through .over().
                roll_mean = (
                    base.rolling_mean(**opts).over(self.group_col)
                )
                roll_std = (
                    base.rolling_std(**opts).over(self.group_col)
                )
                out.append(
                    ((pl.col(col) - roll_mean) / (roll_std + 1e-8))
                    .alias(f"{col}_roll{win}_zscore")
                )
            else:
                raise ValueError(f"Unknown stat: {stat}")  # unreachable after __init__ check

        return out

    @property
    def feature_names_out(self) -> list[str]:
        return self._output_cols


class EWMTransformer(BaseTransformer):
    """
    Exponentially weighted moving average features per symbol.

    EWM features capture recency-weighted information and are especially
    useful for financial data where recent observations matter more than older
    ones.  The `span` parameter controls how quickly weight decays:
      alpha = 2 / (span + 1)   [pandas-compatible definition]
    A span of 5 assigns the most weight to the last ~5 observations.

    Args:
        columns:   Columns to apply EWM to.
        spans:     Span parameters.
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
        logger.debug("EWMTransformer fitted: {} features", len(self._output_cols))

    def _transform(self, df: pl.DataFrame) -> pl.DataFrame:
        sort_by = [self.group_col] + self.sort_cols
        df_sorted = df.sort(sort_by)

        exprs = [
            pl.col(col)
            .ewm_mean(alpha=2.0 / (span + 1.0), adjust=False)
            .over(self.group_col)
            .alias(f"{col}_ewm{int(span)}")
            for col in self.columns
            for span in self.spans
        ]
        return df_sorted.with_columns(exprs)

    @property
    def feature_names_out(self) -> list[str]:
        return self._output_cols
