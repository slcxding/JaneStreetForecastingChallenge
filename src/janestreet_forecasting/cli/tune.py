"""
CLI: js-tune — Optuna hyperparameter search.

Usage:
    js-tune --experiment configs/experiments/exp001_lgbm_baseline.yaml --n-trials 100
    js-tune --experiment configs/experiments/exp001_lgbm_baseline.yaml --n-trials 50 --resume
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from loguru import logger

from janestreet_forecasting.logging_utils import setup_logging
from janestreet_forecasting.paths import ARTIFACTS_DIR, PROCESSED_DIR

app = typer.Typer(help="Run Optuna hyperparameter search for a model.")


@app.command()
def main(
    experiment: Path = typer.Option(
        ...,
        "--experiment", "-e",
        help="Path to experiment YAML config.",
    ),
    n_trials: int = typer.Option(100, "--n-trials", help="Number of Optuna trials."),
    n_cv_folds: int = typer.Option(3, "--n-cv-folds", help="CV folds per trial."),
    features_path: Path = typer.Option(
        PROCESSED_DIR / "features.parquet",
        "--features",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume a previously started study (loads from SQLite).",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run Optuna HPO and write best params to artifact dir."""
    setup_logging(level=log_level)

    import pickle

    import polars as pl

    with open(experiment) as f:
        exp_cfg = yaml.safe_load(f)

    with open(exp_cfg["model_config"]) as f:
        model_cfg = yaml.safe_load(f)

    with open(exp_cfg["train_config"]) as f:
        train_cfg = yaml.safe_load(f)

    experiment_id = exp_cfg["experiment_id"]
    artifact_dir = ARTIFACTS_DIR / experiment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_type = model_cfg["model_type"].replace("lightgbm", "lgbm")

    logger.info("Loading features...")
    df = pl.read_parquet(features_path)

    non_feature = {
        train_cfg["data"]["date_col"],
        "time_id", "symbol_id",
        train_cfg["data"]["target_col"],
        train_cfg["data"]["weight_col"],
    }
    feature_cols = [c for c in df.columns if c not in non_feature]

    storage_path = artifact_dir / "optuna_study.db"
    storage = f"sqlite:///{storage_path}" if resume else None

    from janestreet_forecasting.tuning.optuna_runner import OptunaRunner
    runner = OptunaRunner(
        model_type=model_type,
        df=df,
        feature_cols=feature_cols,
        n_trials=n_trials,
        n_cv_folds=n_cv_folds,
        purge_days=train_cfg["cv"]["purge_days"],
        embargo_days=train_cfg["cv"]["embargo_days"],
        storage=storage,
        study_name=f"{experiment_id}_tuning",
        target_col=train_cfg["data"]["target_col"],
        weight_col=train_cfg["data"]["weight_col"],
        date_col=train_cfg["data"]["date_col"],
    )

    best_params = runner.run()

    best_params_path = artifact_dir / "best_params.yaml"
    runner.save_best_params(best_params_path)
    logger.info("Best params written to {}", best_params_path)
    logger.info("Use these params in your model config to retrain with the full CV.")


if __name__ == "__main__":
    app()
