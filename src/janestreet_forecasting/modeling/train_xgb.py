"""
XGBoost training pipeline with PurgedGroupKFold cross-validation.

Mirrors the structure of train_lgbm.py. Where differences in the API exist
(e.g., DMatrix vs Dataset, callback API), they are handled here.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
from loguru import logger
from rich.console import Console
from rich.table import Table

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.data.splits import FoldIndices, PurgedGroupKFold
from janestreet_forecasting.modeling.datasets import make_xgb_dmatrix, to_numpy_arrays
from janestreet_forecasting.modeling.metrics import compute_all_metrics

console = Console()


class XGBTrainer:
    """
    Manages XGBoost training + CV loop.

    The interface mirrors LGBMTrainer so they are interchangeable in the
    training CLI.
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
        n_estimators: int = 1000,
    ) -> None:
        self.params = {k: v for k, v in params.items() if k != "n_estimators"}
        self.n_estimators = params.get("n_estimators", n_estimators)
        self.cv = cv
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.weight_col = weight_col
        self.date_col = date_col
        self.artifact_dir = artifact_dir
        self.early_stopping_rounds = early_stopping_rounds

        self.fold_models: list[xgb.Booster] = []
        self.fold_scores: list[dict[str, float]] = []
        self.oof_preds: np.ndarray | None = None
        self.feature_importance: dict[str, float] = {}

    def train(self, df: pl.DataFrame) -> "XGBTrainer":
        n = len(df)
        oof_predictions = np.zeros(n, dtype=np.float32)
        all_importances: list[dict[str, float]] = []

        for fold in self.cv.split(df, date_col=self.date_col):
            model, oof_pred, importance = self._train_fold(df, fold)
            self.fold_models.append(model)
            all_importances.append(importance)

            oof_predictions[fold.val_idx] = oof_pred

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

        # Average feature importance (by gain)
        all_features = self.feature_cols
        merged: dict[str, list[float]] = {f: [] for f in all_features}
        for imp in all_importances:
            for feat in all_features:
                merged[feat].append(imp.get(feat, 0.0))
        self.feature_importance = {f: float(np.mean(v)) for f, v in merged.items()}

        self._log_cv_summary()

        if self.artifact_dir:
            self._save_artifacts(df)

        return self

    def _train_fold(
        self,
        df: pl.DataFrame,
        fold: FoldIndices,
    ) -> tuple[xgb.Booster, np.ndarray, dict[str, float]]:
        t0 = time.time()

        train_df = df[fold.train_idx]
        val_df = df[fold.val_idx]

        logger.info(
            "Fold {}: training on {} rows, validating on {} rows",
            fold.fold_idx + 1, len(train_df), len(val_df),
        )

        dtrain = make_xgb_dmatrix(train_df, self.feature_cols, self.target_col, self.weight_col)
        dval = make_xgb_dmatrix(val_df, self.feature_cols, self.target_col, self.weight_col)

        evals_result: dict = {}
        model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=self.early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=100,
        )

        X_val, _, _ = to_numpy_arrays(val_df, self.feature_cols, self.target_col, self.weight_col)
        dval_pred = xgb.DMatrix(X_val, feature_names=self.feature_cols)
        oof_pred = model.predict(dval_pred).astype(np.float32)

        importance = model.get_score(importance_type="gain")

        elapsed = time.time() - t0
        logger.info(
            "Fold {} trained in {:.1f}s | best_iteration={}",
            fold.fold_idx + 1, elapsed, model.best_iteration,
        )

        return model, oof_pred, importance

    def _log_cv_summary(self) -> None:
        scores_by_metric: dict[str, list[float]] = {}
        for fold_scores in self.fold_scores:
            for metric, val in fold_scores.items():
                scores_by_metric.setdefault(metric, []).append(val)

        table = Table(title="XGBoost CV Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="green")
        table.add_column("Std", style="yellow")

        for metric, vals in scores_by_metric.items():
            arr = np.array(vals)
            table.add_row(metric, f"{arr.mean():.4f}", f"{arr.std():.4f}")

        console.print(table)

    def _save_artifacts(self, df: pl.DataFrame) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        for i, model in enumerate(self.fold_models):
            model_path = self.artifact_dir / f"xgb_fold{i}.json"
            model.save_model(str(model_path))

        if self.oof_preds is not None:
            oof_df = df.select(S.INDEX_COLS + [S.TARGET_COL, S.WEIGHT]).with_columns(
                pl.Series("oof_prediction", self.oof_preds)
            )
            oof_df.write_parquet(self.artifact_dir / "oof_predictions.parquet")

        fi_df = pl.DataFrame({
            "feature": list(self.feature_importance.keys()),
            "importance_gain": list(self.feature_importance.values()),
        }).sort("importance_gain", descending=True)
        fi_df.write_parquet(self.artifact_dir / "feature_importance.parquet")

        state = {
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "params": self.params,
            "fold_scores": self.fold_scores,
        }
        with open(self.artifact_dir / "trainer_state.pkl", "wb") as f:
            pickle.dump(state, f)

        logger.info("Artifacts saved to {}", self.artifact_dir)

    @property
    def mean_cv_score(self) -> float:
        if not self.fold_scores:
            return 0.0
        return float(np.mean([s["competition_score"] for s in self.fold_scores]))


def load_xgb_fold_models(artifact_dir: Path) -> list[xgb.Booster]:
    models = []
    for path in sorted(artifact_dir.glob("xgb_fold*.json")):
        model = xgb.Booster()
        model.load_model(str(path))
        models.append(model)
    if not models:
        raise FileNotFoundError(f"No XGBoost fold models found in {artifact_dir}")
    return models


def predict_xgb(
    models: list[xgb.Booster],
    df: pl.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    X = df.select(feature_cols).to_numpy(allow_copy=True).astype(np.float32)
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    all_preds = np.stack([m.predict(dmat) for m in models], axis=0)
    return all_preds.mean(axis=0).astype(np.float32)
