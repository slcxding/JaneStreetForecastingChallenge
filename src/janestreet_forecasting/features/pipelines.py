"""
Feature pipeline: composes multiple transformers in sequence and handles
saving/loading of the fitted pipeline.

Design choice: we build our own thin pipeline instead of using sklearn's
Pipeline because:
  1. Our transformers work on Polars DataFrames, not NumPy arrays.
  2. sklearn Pipeline's column handling doesn't fit our "add columns in place"
     pattern cleanly.
  3. We want explicit control over which column names flow through.

The `FeaturePipeline` tracks all fitted transformers and the final feature
column list so inference can reconstruct the exact same feature set.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import polars as pl
from loguru import logger

from janestreet_forecasting.features.base import BaseTransformer, TransformerProtocol


class FeaturePipeline:
    """
    Sequential feature transformation pipeline.

    Applies a list of transformers in order, passing the output DataFrame of
    each step as input to the next.  Tracks the final list of feature columns
    for downstream model training.

    Args:
        transformers:    List of (name, transformer) pairs.
        feature_cols:    Explicit list of columns to use as model features.
                         If None, the pipeline collects them from all
                         transformer `feature_names_out` lists.
        passthrough_cols: Columns that always pass through (index + target).
    """

    def __init__(
        self,
        transformers: list[tuple[str, TransformerProtocol]],
        feature_cols: list[str] | None = None,
        passthrough_cols: list[str] | None = None,
    ) -> None:
        self.transformers = transformers
        self._explicit_feature_cols = feature_cols
        self.passthrough_cols = passthrough_cols or []
        self._fitted = False
        self._feature_cols: list[str] = []

    def fit(self, df: pl.DataFrame, **kwargs: Any) -> "FeaturePipeline":
        """Fit all transformers on the training DataFrame."""
        current_df = df
        for name, transformer in self.transformers:
            logger.info("Fitting transformer: {}", name)
            transformer.fit(current_df, **kwargs)
            current_df = transformer.transform(current_df)

        self._feature_cols = self._resolve_feature_cols(current_df)
        self._fitted = True
        logger.info(
            "FeaturePipeline fitted: {} total feature columns", len(self._feature_cols)
        )
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all fitted transformers."""
        self._assert_fitted()
        for _name, transformer in self.transformers:
            df = transformer.transform(df)
        return df

    def fit_transform(self, df: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
        """Fit then transform on the same data."""
        return self.fit(df, **kwargs).transform(df)

    @property
    def feature_cols(self) -> list[str]:
        """Final list of model input columns."""
        self._assert_fitted()
        return self._feature_cols

    def save(self, path: Path | str) -> None:
        """Serialise the fitted pipeline to disk (pickle)."""
        self._assert_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("FeaturePipeline saved to {}", path)

    @classmethod
    def load(cls, path: Path | str) -> "FeaturePipeline":
        """Load a previously saved pipeline."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        logger.info("FeaturePipeline loaded from {}", path)
        return pipeline

    def _resolve_feature_cols(self, df: pl.DataFrame) -> list[str]:
        """Determine final feature column list after all transforms."""
        if self._explicit_feature_cols is not None:
            # Validate that all specified columns exist
            missing = set(self._explicit_feature_cols) - set(df.columns)
            if missing:
                raise ValueError(
                    f"Specified feature columns not found after transform: {missing}"
                )
            return self._explicit_feature_cols

        # Collect from all transformers
        all_added = []
        for _, transformer in self.transformers:
            all_added.extend(transformer.feature_names_out)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique = []
        for col in all_added:
            if col not in seen and col in df.columns:
                seen.add(col)
                unique.append(col)

        if not unique:
            raise ValueError(
                "No feature columns collected from pipeline. "
                "Specify feature_cols explicitly or ensure transformers populate "
                "feature_names_out."
            )
        return unique

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "FeaturePipeline must be fitted before calling transform() or "
                "accessing feature_cols."
            )


def build_default_pipeline(config: dict[str, Any]) -> FeaturePipeline:
    """
    Build the default feature pipeline from a features config dict.

    This is the main entry point for building a pipeline from YAML config.
    Each feature group (lag, rolling, cross-sectional) can be toggled via
    config, making it easy to run ablation studies.

    Args:
        config: Contents of a features/*.yaml config file.

    Returns:
        Unfitted FeaturePipeline.
    """
    from janestreet_forecasting.features.cross_features import (
        CrossSectionalRankTransformer,
    )
    from janestreet_forecasting.features.lag_features import LagTransformer
    from janestreet_forecasting.features.rolling_features import RollingTransformer

    transformers: list[tuple[str, Any]] = []

    lag_cfg = config.get("lag", {})
    if lag_cfg.get("enabled", True):
        transformers.append((
            "lag",
            LagTransformer(
                columns=lag_cfg["columns"],
                lags=lag_cfg["windows"],
            ),
        ))

    roll_cfg = config.get("rolling", {})
    if roll_cfg.get("enabled", True):
        transformers.append((
            "rolling",
            RollingTransformer(
                columns=roll_cfg["columns"],
                windows=roll_cfg["windows"],
                stats=roll_cfg.get("stats", ["mean", "std"]),
            ),
        ))

    xs_cfg = config.get("cross_sectional", {})
    if xs_cfg.get("enabled", True):
        transformers.append((
            "cross_sectional",
            CrossSectionalRankTransformer(
                columns=xs_cfg["columns"],
                rank_method=xs_cfg.get("rank_method", "standard"),
                normalize=xs_cfg.get("normalize", True),
            ),
        ))

    return FeaturePipeline(transformers=transformers)
