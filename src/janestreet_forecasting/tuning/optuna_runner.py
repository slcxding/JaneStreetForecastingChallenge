"""
Optuna hyperparameter search for LightGBM, XGBoost, and CatBoost.

Design:
  - Each model type has its own `suggest_*_params` function that defines the
    search space.  Modify these to explore different ranges.
  - The objective function runs a single fold of PurgedGroupKFold to keep
    each trial fast.  Use n_splits=1 for tuning, then retrain with n_splits=5
    for the final model.
  - Optuna's MedianPruner cuts unpromising trials early (after the first fold).
  - Results are stored in an SQLite database so tuning can be resumed after
    interruption: `js-tune --resume`.

Usage:
    runner = OptunaRunner(
        model_type="lgbm",
        df=features_df,
        feature_cols=feature_cols,
        n_trials=100,
        study_name="lgbm_v1",
    )
    best_params = runner.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import optuna
import polars as pl
from loguru import logger
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.data.splits import PurgedGroupKFold
from janestreet_forecasting.modeling.metrics import mean_competition_score


class OptunaRunner:
    """
    Orchestrates Optuna hyperparameter search for a given model type.

    Args:
        model_type:   "lgbm" | "xgb" | "catboost"
        df:           Full training DataFrame (with features + target).
        feature_cols: Feature column names.
        n_trials:     Number of Optuna trials.
        n_cv_folds:   Number of CV folds per trial (1 = fast, 5 = thorough).
        purge_days:   Purge parameter for PurgedGroupKFold.
        embargo_days: Embargo parameter for PurgedGroupKFold.
        storage:      Optuna storage URI (sqlite:///path.db for persistence).
        study_name:   Name for the Optuna study.
        n_jobs:       Parallel trials (-1 = CPU count).  Note: parallel trials
                      share the trial DB but have independent Python processes.
    """

    def __init__(
        self,
        model_type: str,
        df: pl.DataFrame,
        feature_cols: list[str],
        n_trials: int = 100,
        n_cv_folds: int = 3,
        purge_days: int = 5,
        embargo_days: int = 10,
        storage: str | None = None,
        study_name: str = "js_tuning",
        n_jobs: int = 1,
        target_col: str = S.TARGET_COL,
        weight_col: str = S.WEIGHT,
        date_col: str = S.DATE_ID,
    ) -> None:
        if model_type not in ("lgbm", "xgb", "catboost"):
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model_type = model_type
        self.df = df
        self.feature_cols = feature_cols
        self.n_trials = n_trials
        self.n_cv_folds = n_cv_folds
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.storage = storage
        self.study_name = study_name
        self.n_jobs = n_jobs
        self.target_col = target_col
        self.weight_col = weight_col
        self.date_col = date_col

        self._best_params: dict[str, Any] | None = None

    def run(self) -> dict[str, Any]:
        """
        Run the hyperparameter search.

        Returns:
            Best hyperparameters found.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=True,  # Allows resuming
        )

        logger.info(
            "Starting Optuna study '{}' for {} | {} trials",
            self.study_name, self.model_type, self.n_trials,
        )

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )

        self._best_params = study.best_params
        best_score = study.best_value

        logger.info(
            "Tuning complete. Best score={:.4f} | params={}",
            best_score, self._best_params,
        )

        return self._best_params

    def _objective(self, trial: optuna.Trial) -> float:
        """Single trial: suggest params, run fast CV, return score."""
        params = self._suggest_params(trial)

        cv = PurgedGroupKFold(
            n_splits=self.n_cv_folds,
            purge_days=self.purge_days,
            embargo_days=self.embargo_days,
        )

        fold_scores = []
        for fold_idx, fold in enumerate(cv.split(self.df, date_col=self.date_col)):
            score = self._train_and_score_fold(params, fold)
            fold_scores.append(score)

            # Report intermediate value for pruning
            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest hyperparameters for the current model type."""
        if self.model_type == "lgbm":
            return _suggest_lgbm_params(trial)
        elif self.model_type == "xgb":
            return _suggest_xgb_params(trial)
        elif self.model_type == "catboost":
            return _suggest_catboost_params(trial)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _train_and_score_fold(self, params: dict[str, Any], fold) -> float:
        """Train one fold and return the competition score."""
        train_df = self.df[fold.train_idx]
        val_df = self.df[fold.val_idx]

        X_train = train_df.select(self.feature_cols).to_numpy().astype(np.float32)
        y_train = train_df[self.target_col].to_numpy().astype(np.float32)
        w_train = train_df[self.weight_col].to_numpy().astype(np.float32)

        X_val = val_df.select(self.feature_cols).to_numpy().astype(np.float32)
        y_val = val_df[self.target_col].to_numpy().astype(np.float32)
        w_val = val_df[self.weight_col].to_numpy().astype(np.float32)
        d_val = val_df[self.date_col].to_numpy()

        preds = self._train_fold_model(params, X_train, y_train, w_train, X_val, y_val, w_val)

        return mean_competition_score(y_val, preds, w_val, d_val)

    def _train_fold_model(
        self,
        params: dict[str, Any],
        X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, w_val: np.ndarray,
    ) -> np.ndarray:
        """Train and predict for one fold, dispatching on model_type."""
        if self.model_type == "lgbm":
            return _train_lgbm_fold(params, X_train, y_train, w_train, X_val, y_val, w_val)
        elif self.model_type == "xgb":
            return _train_xgb_fold(params, X_train, y_train, w_train, X_val, y_val, w_val)
        elif self.model_type == "catboost":
            return _train_catboost_fold(params, X_train, y_train, w_train, X_val, y_val, w_val)
        else:
            raise ValueError(self.model_type)

    def save_best_params(self, path: Path) -> None:
        """Write best parameters to a YAML file for inspection."""
        import yaml
        if self._best_params is None:
            raise RuntimeError("Call run() first.")
        with open(path, "w") as f:
            yaml.dump({"params": self._best_params}, f, default_flow_style=False)
        logger.info("Best params saved to {}", path)


# ---------------------------------------------------------------------------
# Search spaces — modify these to explore different regions
# ---------------------------------------------------------------------------


def _suggest_lgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 512, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
    }


def _suggest_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "verbosity": 0,
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }


def _suggest_catboost_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "loss_function": "RMSE",
        "iterations": 500,
        "verbose": 0,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "random_seed": 42,
        "thread_count": -1,
        "nan_mode": "Min",
    }


# ---------------------------------------------------------------------------
# Thin per-library training wrappers (no CV overhead, just fit+predict)
# ---------------------------------------------------------------------------


def _train_lgbm_fold(params, X_train, y_train, w_train, X_val, y_val, w_val) -> np.ndarray:
    import lightgbm as lgb
    n_est = params.pop("n_estimators", 500)
    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train)
    dval = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=n_est,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    params["n_estimators"] = n_est  # Restore for next trial
    return model.predict(X_val).astype(np.float32)


def _train_xgb_fold(params, X_train, y_train, w_train, X_val, y_val, w_val) -> np.ndarray:
    import xgboost as xgb
    n_est = params.pop("n_estimators", 500)
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)
    model = xgb.train(
        params, dtrain, num_boost_round=n_est,
        evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False,
    )
    params["n_estimators"] = n_est
    return model.predict(dval).astype(np.float32)


def _train_catboost_fold(params, X_train, y_train, w_train, X_val, y_val, w_val) -> np.ndarray:
    from catboost import CatBoostRegressor, Pool
    train_pool = Pool(X_train, label=y_train, weight=w_train)
    val_pool = Pool(X_val, label=y_val, weight=w_val)
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)
    return model.predict(X_val).astype(np.float32)
