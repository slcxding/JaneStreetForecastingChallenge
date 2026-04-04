"""
Baseline models that establish a performance floor.

Every new model must beat these baselines — if it doesn't, something is wrong
with either the model or the feature engineering.

Baselines implemented:
  - ZeroPredictor     : always predicts 0.0 (useful lower bound)
  - WeightedMean      : predicts the weighted mean of the training target
  - MedianPredictor   : predicts the median of the training target
  - SymbolMeanPredictor: predicts each symbol's historical mean return
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from janestreet_forecasting.data import schemas as S


class ZeroPredictor:
    """Always predicts 0.0. Establishes an absolute floor."""

    def fit(self, df: pl.DataFrame, **kwargs: Any) -> "ZeroPredictor":
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype=np.float32)

    @property
    def name(self) -> str:
        return "zero_predictor"


class WeightedMeanPredictor:
    """Predicts the weighted mean of the training target for all rows."""

    def __init__(
        self,
        target_col: str = S.TARGET_COL,
        weight_col: str = S.WEIGHT,
    ) -> None:
        self.target_col = target_col
        self.weight_col = weight_col
        self._mean: float = 0.0

    def fit(self, df: pl.DataFrame, **kwargs: Any) -> "WeightedMeanPredictor":
        y = df[self.target_col].to_numpy()
        w = df[self.weight_col].to_numpy()
        # Guard against zero-weight scenarios
        total_weight = w.sum()
        if total_weight < 1e-10:
            self._mean = float(y.mean())
        else:
            self._mean = float(np.average(y, weights=w))
        logger.debug("WeightedMeanPredictor fitted: mean={:.6f}", self._mean)
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return np.full(len(df), self._mean, dtype=np.float32)

    @property
    def name(self) -> str:
        return "weighted_mean_predictor"


class MedianPredictor:
    """Predicts the median of the training target."""

    def __init__(self, target_col: str = S.TARGET_COL) -> None:
        self.target_col = target_col
        self._median: float = 0.0

    def fit(self, df: pl.DataFrame, **kwargs: Any) -> "MedianPredictor":
        self._median = float(df[self.target_col].median() or 0.0)
        logger.debug("MedianPredictor fitted: median={:.6f}", self._median)
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return np.full(len(df), self._median, dtype=np.float32)

    @property
    def name(self) -> str:
        return "median_predictor"


class SymbolMeanPredictor:
    """
    Predicts each symbol's historical weighted mean return.

    For symbols not seen during training, falls back to the global mean.
    This captures symbol-level return persistence (if any exists).
    """

    def __init__(
        self,
        target_col: str = S.TARGET_COL,
        weight_col: str = S.WEIGHT,
        symbol_col: str = S.SYMBOL_ID,
    ) -> None:
        self.target_col = target_col
        self.weight_col = weight_col
        self.symbol_col = symbol_col
        self._symbol_means: dict[int, float] = {}
        self._global_mean: float = 0.0

    def fit(self, df: pl.DataFrame, **kwargs: Any) -> "SymbolMeanPredictor":
        # Compute weighted mean per symbol
        agg = (
            df.group_by(self.symbol_col)
            .agg(
                (
                    (pl.col(self.target_col) * pl.col(self.weight_col)).sum()
                    / pl.col(self.weight_col).sum()
                ).alias("wmean")
            )
        )
        self._symbol_means = dict(
            zip(agg[self.symbol_col].to_list(), agg["wmean"].to_list())
        )

        # Global fallback
        y = df[self.target_col].to_numpy()
        w = df[self.weight_col].to_numpy()
        self._global_mean = float(np.average(y, weights=w))

        logger.debug(
            "SymbolMeanPredictor fitted: {} symbols, global_mean={:.6f}",
            len(self._symbol_means), self._global_mean,
        )
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        symbols = df[self.symbol_col].to_list()
        preds = np.array(
            [self._symbol_means.get(s, self._global_mean) for s in symbols],
            dtype=np.float32,
        )
        return preds

    @property
    def name(self) -> str:
        return "symbol_mean_predictor"
