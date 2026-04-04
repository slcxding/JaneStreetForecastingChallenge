"""
LightGBM training pipeline with PurgedGroupKFold cross-validation.

Returns:
  - Per-fold trained models
  - Out-of-fold predictions (for ensembling + model selection)
  - Feature importance aggregated across folds
  - Per-fold metric scores

Design notes:
  - We use LightGBM's native early stopping with a held-out fold.
  - Feature importance is averaged across folds for stability.
  - The final artifact includes all fold models (for ensemble) and the
    feature importance sorted by mean gain.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from loguru import logger
from rich.console import Console
from rich.table import Table

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.data.splits import FoldIndices, PurgedGroupKFold
from janestreet_forecasting.modeling.datasets import make_lgbm_dataset, to_numpy_arrays
from janestreet_forecasting.modeling.metrics import compute_all_metrics

console = Console()


class LGBMTrainer:
    """
    Manages the full LightGBM training + CV loop.

    Args:
        params:      LightGBM training parameters (from config).
        cv:          Cross-validation strategy (PurgedGroupKFold or WalkForward).
        feature_cols: Feature column names.
        target_col:  Target column name.
        weight_col:  Weight column name.
        artifact_dir: Where to save model artefacts.
    """

    def __init__(
        self,
        params: dict[str, Any],
        cv: PurgedGroupKFold,
        feature_cols: list[str],
        target_col: str = S.TARGET_COL,
        weight_col: str = S.WEIGHT,
        date_col: str = S.DATE_ID,
        artifact_dir: Path | None = None,
        early_stopping_rounds: int = 100,
    ) -> None:
        self.params = params.copy()
        self.cv = cv
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.weight_col = weight_col
        self.date_col = date_col
        self.artifact_dir = artifact_dir
        self.early_stopping_rounds = early_stopping_rounds

        self.fold_models: list[lgb.Booster] = []
        self.fold_scores: list[dict[str, float]] = []
        self.oof_preds: np.ndarray | None = None
        self.feature_importance: dict[str, float] = {}

    def train(self, df: pl.DataFrame) -> "LGBMTrainer":
        """
        Run the full cross-validation loop.

        Args:
            df: Full training DataFrame (will be split internally).

        Returns:
            self (for chaining)
        """
        n = len(df)
        oof_predictions = np.zeros(n, dtype=np.float32)
        all_importances: list[np.ndarray] = []

        for fold in self.cv.split(df, date_col=self.date_col):
            model, oof_pred, importance = self._train_fold(df, fold)
            self.fold_models.append(model)
            all_importances.append(importance)

            oof_predictions[fold.val_idx] = oof_pred

            # Compute metrics for this fold
            val_df = df[fold.val_idx]
            scores = compute_all_metrics(
                y_true=val_df[self.target_col].to_numpy(),
                y_pred=oof_pred,
                weights=val_df[self.weight_col].to_numpy(),
                dates=val_df[self.date_col].to_numpy(),
            )
            self.fold_scores.append(scores)
            logger.info(
                "Fold {} | competition_score={:.4f} | weighted_r2={:.4f}",
                fold.fold_idx + 1,
                scores["competition_score"],
                scores["weighted_r2"],
            )

        self.oof_preds = oof_predictions

        # Average feature importance across folds
        importance_matrix = np.stack(all_importances, axis=0)
        mean_importance = importance_matrix.mean(axis=0)
        self.feature_importance = dict(
            zip(self.feature_cols, mean_importance.tolist())
        )

        self._log_cv_summary()

        if self.artifact_dir:
            self._save_artifacts(df)

        return self

    def _train_fold(
        self,
        df: pl.DataFrame,
        fold: FoldIndices,
    ) -> tuple[lgb.Booster, np.ndarray, np.ndarray]:
        """Train a single fold. Returns (model, oof_predictions, feature_importance)."""
        t0 = time.time()

        train_df = df[fold.train_idx]
        val_df = df[fold.val_idx]

        logger.info(
            "Fold {}: training on {} rows, validating on {} rows",
            fold.fold_idx + 1, len(train_df), len(val_df),
        )

        dtrain = make_lgbm_dataset(train_df, self.feature_cols, self.target_col, self.weight_col)
        dval = make_lgbm_dataset(val_df, self.feature_cols, self.target_col, self.weight_col, reference=dtrain)

        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds,
                verbose=False,
            ),
        ]

        model = lgb.train(
            params=self.params,
            train_set=dtrain,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=callbacks,
        )

        X_val, _, _ = to_numpy_arrays(val_df, self.feature_cols, self.target_col, self.weight_col)
        oof_pred = model.predict(X_val).astype(np.float32)

        importance = model.feature_importance(importance_type="gain").astype(np.float32)

        elapsed = time.time() - t0
        logger.info("Fold {} trained in {:.1f}s | best_iteration={}", fold.fold_idx + 1, elapsed, model.best_iteration)

        return model, oof_pred, importance

    def _log_cv_summary(self) -> None:
        """Pretty-print cross-validation results."""
        scores_by_metric: dict[str, list[float]] = {}
        for fold_scores in self.fold_scores:
            for metric, val in fold_scores.items():
                scores_by_metric.setdefault(metric, []).append(val)

        table = Table(title="LightGBM CV Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="green")
        table.add_column("Std", style="yellow")

        for metric, vals in scores_by_metric.items():
            arr = np.array(vals)
            table.add_row(metric, f"{arr.mean():.4f}", f"{arr.std():.4f}")

        console.print(table)

    def _save_artifacts(self, df: pl.DataFrame) -> None:
        """Save models, OOF predictions, and feature importance."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # Save each fold model
        for i, model in enumerate(self.fold_models):
            model_path = self.artifact_dir / f"lgbm_fold{i}.txt"
            model.save_model(str(model_path))
            logger.debug("Saved fold {} model to {}", i, model_path)

        # Save OOF predictions alongside the index columns
        if self.oof_preds is not None:
            oof_df = df.select(S.INDEX_COLS + [S.TARGET_COL, S.WEIGHT]).with_columns(
                pl.Series("oof_prediction", self.oof_preds)
            )
            oof_df.write_parquet(self.artifact_dir / "oof_predictions.parquet")

        # Save feature importance as a sorted parquet
        fi_df = pl.DataFrame({
            "feature": list(self.feature_importance.keys()),
            "importance_gain": list(self.feature_importance.values()),
        }).sort("importance_gain", descending=True)
        fi_df.write_parquet(self.artifact_dir / "feature_importance.parquet")

        # Save trainer state (for loading fold models programmatically)
        state = {
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "weight_col": self.weight_col,
            "params": self.params,
            "fold_scores": self.fold_scores,
        }
        with open(self.artifact_dir / "trainer_state.pkl", "wb") as f:
            pickle.dump(state, f)

        logger.info("Artifacts saved to {}", self.artifact_dir)

    @property
    def mean_cv_score(self) -> float:
        """Mean competition_score across folds."""
        if not self.fold_scores:
            return 0.0
        return float(np.mean([s["competition_score"] for s in self.fold_scores]))


def load_lgbm_fold_models(artifact_dir: Path) -> list[lgb.Booster]:
    """Load all saved fold models from an artifact directory."""
    models = []
    for path in sorted(artifact_dir.glob("lgbm_fold*.txt")):
        model = lgb.Booster(model_file=str(path))
        models.append(model)
        logger.debug("Loaded fold model from {}", path)

    if not models:
        raise FileNotFoundError(f"No LightGBM fold models found in {artifact_dir}")

    return models


def predict_lgbm(
    models: list[lgb.Booster],
    df: pl.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Average predictions across all fold models.

    Args:
        models:       List of LightGBM Boosters (one per CV fold).
        df:           DataFrame to predict on.
        feature_cols: Feature column names (must match training).

    Returns:
        Averaged predictions, shape (N,).
    """
    X = df.select(feature_cols).to_numpy(allow_copy=True).astype(np.float32)
    all_preds = np.stack([m.predict(X) for m in models], axis=0)
    return all_preds.mean(axis=0).astype(np.float32)
