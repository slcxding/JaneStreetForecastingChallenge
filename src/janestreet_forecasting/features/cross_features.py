"""
Cross-sectional feature generation.

Cross-sectional features capture how a symbol compares to all other symbols
at the same point in time (same date_id + time_id).  Examples:

  - Cross-sectional rank of feature_00 at time t
  - Z-score of responder_6_lag_0 relative to the universe at time t

These features are commonly used in equity factor models.

LEAKAGE NOTE: Cross-sectional features are computed within each (date_id,
time_id) group using only the values available at that snapshot — no future
information is used.  However, the cross-section requires all symbols to have
submitted data, which the Kaggle API guarantees.
"""

from __future__ import annotations

from typing import Any, Sequence

import polars as pl
from loguru import logger

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.features.base import BaseTransformer


class CrossSectionalRankTransformer(BaseTransformer):
    """
    Compute cross-sectional ranks of features within each (date_id, time_id).

    Ranks are optionally normalised to [0, 1] or z-scored.

    Args:
        columns:       Feature columns to rank.
        group_cols:    Snapshot grouping columns (default: date_id + time_id).
        rank_method:   Polars rank method: "average" | "min" | "max" | "dense" | "ordinal" | "random".
        normalize:     If True, z-score normalise the ranks within each snapshot.
    """

    def __init__(
        self,
        columns: Sequence[str],
        group_cols: Sequence[str] = (S.DATE_ID, S.TIME_ID),
        rank_method: str = "average",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.columns = list(columns)
        self.group_cols = list(group_cols)
        self.rank_method = rank_method
        self.normalize = normalize

    def _fit(self, df: pl.DataFrame, **kwargs: Any) -> None:
        suffix = "_xrank_norm" if self.normalize else "_xrank"
        self._output_cols = [f"{col}{suffix}" for col in self.columns]
        logger.debug(
            "CrossSectionalRankTransformer fitted: {} features", len(self._output_cols)
        )

    def _transform(self, df: pl.DataFrame) -> pl.DataFrame:
        exprs: list[pl.Expr] = []
        suffix = "_xrank_norm" if self.normalize else "_xrank"

        for col in self.columns:
            rank_expr = (
                pl.col(col)
                .rank(method=self.rank_method)
                .over(self.group_cols)
            )

            if self.normalize:
                # Z-score normalise within the cross-section
                mean_expr = rank_expr.mean().over(self.group_cols)
                std_expr = rank_expr.std().over(self.group_cols)
                final_expr = (
                    ((rank_expr - mean_expr) / (std_expr + 1e-8))
                    .alias(f"{col}{suffix}")
                )
            else:
                final_expr = rank_expr.alias(f"{col}{suffix}")

            exprs.append(final_expr)

        return df.with_columns(exprs)

    @property
    def feature_names_out(self) -> list[str]:
        return self._output_cols


class CrossSectionalZScoreTransformer(BaseTransformer):
    """
    Z-score each feature relative to the cross-section at each time step.

    This is a simpler alternative to ranking — useful when the distribution
    is approximately normal.  Z-score is more sensitive to outliers than rank.
    """

    def __init__(
        self,
        columns: Sequence[str],
        group_cols: Sequence[str] = (S.DATE_ID, S.TIME_ID),
    ) -> None:
        super().__init__()
        self.columns = list(columns)
        self.group_cols = list(group_cols)

    def _fit(self, df: pl.DataFrame, **kwargs: Any) -> None:
        self._output_cols = [f"{col}_xzscore" for col in self.columns]
        logger.debug(
            "CrossSectionalZScoreTransformer fitted: {} features",
            len(self._output_cols),
        )

    def _transform(self, df: pl.DataFrame) -> pl.DataFrame:
        exprs = [
            (
                (pl.col(col) - pl.col(col).mean().over(self.group_cols))
                / (pl.col(col).std().over(self.group_cols) + 1e-8)
            ).alias(f"{col}_xzscore")
            for col in self.columns
        ]
        return df.with_columns(exprs)

    @property
    def feature_names_out(self) -> list[str]:
        return self._output_cols
