"""
PredictorPipeline — bundles model + feature pipeline + inference state.

This is the single object that gets loaded at inference time.  It owns:
  - The trained model(s) (one per fold, averaged)
  - The fitted feature pipeline (for stateless transforms like cross-sectional)
  - The InferenceState (for stateful rolling features)
  - The feature column list (must match training exactly)

Usage:
    predictor = PredictorPipeline.load(artifact_dir)
    for batch in batches:
        preds = predictor.predict(batch)
        predictor.update_state(batch)  # always call AFTER predict
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from loguru import logger

from janestreet_forecasting.inference.state import InferenceState


class PredictorPipeline:
    """
    Self-contained prediction object for real-time inference.

    Args:
        predict_fn:      Function (df: pl.DataFrame) -> np.ndarray.
                         Wraps the underlying model(s) — handles averaging
                         across folds internally.
        feature_cols:    Ordered list of feature column names expected by predict_fn.
        state:           InferenceState for rolling feature computation.
        clip_range:      (min, max) clipping for predictions.  None = no clipping.
    """

    def __init__(
        self,
        predict_fn: Callable[[pl.DataFrame], np.ndarray],
        feature_cols: list[str],
        state: InferenceState,
        clip_range: tuple[float, float] | None = (-5.0, 5.0),
    ) -> None:
        self.predict_fn = predict_fn
        self.feature_cols = feature_cols
        self.state = state
        self.clip_range = clip_range

    def predict(self, batch: pl.DataFrame) -> np.ndarray:
        """
        Generate predictions for a batch.

        Steps:
          1. Compute rolling/lag features from state (stateful)
          2. Apply predict_fn (stateless)
          3. Clip predictions

        Args:
            batch: Current time step's rows (from Kaggle API or local simulator).

        Returns:
            Predictions array, shape (len(batch),).
        """
        # Add state-derived features to the batch
        enriched = self.state.compute_features(batch)

        # Ensure all required feature columns exist (fill missing with NaN)
        missing_cols = [c for c in self.feature_cols if c not in enriched.columns]
        if missing_cols:
            logger.warning("Missing feature columns at inference: {}", missing_cols)
            for c in missing_cols:
                enriched = enriched.with_columns(pl.lit(None).cast(pl.Float32).alias(c))

        preds = self.predict_fn(enriched)

        if self.clip_range is not None:
            preds = np.clip(preds, self.clip_range[0], self.clip_range[1])

        return preds.astype(np.float32)

    def update_state(self, batch: pl.DataFrame) -> None:
        """
        Update the rolling buffer with the current batch.

        MUST be called after predict() for each batch.
        In the Kaggle API environment, this is called after the prediction
        is submitted and the next batch arrives.
        """
        self.state.update(batch)

    @classmethod
    def load_lgbm(cls, artifact_dir: Path, **kwargs) -> "PredictorPipeline":
        """Load a LightGBM predictor from an artifact directory."""
        from janestreet_forecasting.modeling.train_lgbm import (
            load_lgbm_fold_models,
            predict_lgbm,
        )

        models = load_lgbm_fold_models(artifact_dir)

        with open(artifact_dir / "trainer_state.pkl", "rb") as f:
            state_dict = pickle.load(f)

        feature_cols = state_dict["feature_cols"]

        def predict_fn(df: pl.DataFrame) -> np.ndarray:
            return predict_lgbm(models, df, feature_cols)

        inference_state = cls._build_state(artifact_dir, feature_cols)
        return cls(predict_fn=predict_fn, feature_cols=feature_cols, state=inference_state, **kwargs)

    @classmethod
    def load_xgb(cls, artifact_dir: Path, **kwargs) -> "PredictorPipeline":
        """Load an XGBoost predictor from an artifact directory."""
        from janestreet_forecasting.modeling.train_xgb import (
            load_xgb_fold_models,
            predict_xgb,
        )

        models = load_xgb_fold_models(artifact_dir)

        with open(artifact_dir / "trainer_state.pkl", "rb") as f:
            state_dict = pickle.load(f)

        feature_cols = state_dict["feature_cols"]

        def predict_fn(df: pl.DataFrame) -> np.ndarray:
            return predict_xgb(models, df, feature_cols)

        inference_state = cls._build_state(artifact_dir, feature_cols)
        return cls(predict_fn=predict_fn, feature_cols=feature_cols, state=inference_state, **kwargs)

    @classmethod
    def load_catboost(cls, artifact_dir: Path, **kwargs) -> "PredictorPipeline":
        """Load a CatBoost predictor from an artifact directory."""
        from janestreet_forecasting.modeling.train_catboost import (
            load_catboost_fold_models,
            predict_catboost,
        )

        models = load_catboost_fold_models(artifact_dir)

        with open(artifact_dir / "trainer_state.pkl", "rb") as f:
            state_dict = pickle.load(f)

        feature_cols = state_dict["feature_cols"]

        def predict_fn(df: pl.DataFrame) -> np.ndarray:
            return predict_catboost(models, df, feature_cols)

        inference_state = cls._build_state(artifact_dir, feature_cols)
        return cls(predict_fn=predict_fn, feature_cols=feature_cols, state=inference_state, **kwargs)

    @staticmethod
    def _build_state(artifact_dir: Path, feature_cols: list[str]) -> InferenceState:
        """Build InferenceState from a state config file if present."""
        state_config_path = artifact_dir / "inference_state_config.pkl"
        if state_config_path.exists():
            with open(state_config_path, "rb") as f:
                cfg = pickle.load(f)
            return InferenceState(**cfg)
        # Default state — minimal rolling features from lag columns
        return InferenceState(buffer_size=30)
