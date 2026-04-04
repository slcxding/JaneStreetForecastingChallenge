"""
CLI: js-train — Train a model defined by an experiment config.

Usage:
    js-train --experiment configs/experiments/exp001_lgbm_baseline.yaml
    js-train --experiment configs/experiments/exp002_xgb_baseline.yaml
    js-train --experiment configs/experiments/exp004_ensemble.yaml
"""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
import yaml
from loguru import logger

from janestreet_forecasting.logging_utils import setup_logging
from janestreet_forecasting.paths import ARTIFACTS_DIR, PROCESSED_DIR

app = typer.Typer(help="Train a model from an experiment config.")


@app.command()
def main(
    experiment: Path = typer.Option(
        ...,
        "--experiment", "-e",
        help="Path to experiment YAML config.",
    ),
    features_path: Path = typer.Option(
        PROCESSED_DIR / "features.parquet",
        "--features",
        help="Path to processed features parquet.",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Load data and validate config without training.",
    ),
) -> None:
    """Train model, run CV, and save artifacts."""
    setup_logging(level=log_level)

    import polars as pl

    with open(experiment) as f:
        exp_cfg = yaml.safe_load(f)

    experiment_id = exp_cfg["experiment_id"]
    artifact_dir = ARTIFACTS_DIR / experiment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Always copy the experiment config into the artifact directory
    shutil.copy(experiment, artifact_dir / "experiment_config.yaml")

    # Load sub-configs
    with open(exp_cfg["model_config"]) as f:
        model_cfg = yaml.safe_load(f)
    with open(exp_cfg["train_config"]) as f:
        train_cfg = yaml.safe_load(f)

    # Apply overrides from experiment config
    overrides = exp_cfg.get("overrides", {})
    if "model" in overrides:
        _deep_merge(model_cfg, overrides["model"])
    if "train" in overrides:
        _deep_merge(train_cfg, overrides["train"])

    model_type = model_cfg["model_type"]
    logger.info("Experiment: {} | Model: {}", experiment_id, model_type)

    if not features_path.exists():
        logger.error("Features not found at {}. Run `js-features` first.", features_path)
        raise typer.Exit(code=1)

    logger.info("Loading features from {}", features_path)
    df = pl.read_parquet(features_path)
    logger.info("Loaded {} rows × {} columns", len(df), len(df.columns))

    if dry_run:
        logger.info("[dry-run] Config validated. No training performed.")
        return

    # Resolve CV strategy
    from janestreet_forecasting.data.splits import PurgedGroupKFold, WalkForwardSplit
    cv_cfg = train_cfg["cv"]
    strategy = cv_cfg.get("strategy", "purged_group_kfold")

    if strategy == "purged_group_kfold":
        cv = PurgedGroupKFold(
            n_splits=cv_cfg["n_splits"],
            purge_days=cv_cfg["purge_days"],
            embargo_days=cv_cfg["embargo_days"],
            forward_only=cv_cfg.get("forward_only", True),
            min_train_dates=cv_cfg.get("min_train_dates", 30),
        )
    elif strategy == "walk_forward":
        wf = cv_cfg["walk_forward"]
        cv = WalkForwardSplit(
            min_train_days=wf["min_train_days"],
            test_window=wf["test_window"],
            step_days=wf["step_days"],
            embargo_days=wf["embargo_days"],
        )
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")

    # Load feature column list from the saved pipeline — do NOT derive it by
    # excluding index/target/weight from the DataFrame.  Exclusion-based
    # selection would silently include any intermediate columns that leaked
    # through the pipeline (e.g. zscore temp columns).
    pipeline_path = PROCESSED_DIR / "feature_pipeline.pkl"
    if pipeline_path.exists():
        from janestreet_forecasting.features.pipelines import FeaturePipeline
        pipeline = FeaturePipeline.load(pipeline_path)
        feature_cols = pipeline.feature_cols
        logger.info("Feature columns from saved pipeline: {}", len(feature_cols))
    else:
        # Fallback for cases where the raw features file is used directly
        # (e.g. skip feature engineering step).  Still use explicit exclusion
        # but from schemas — not hard-coded strings.
        from janestreet_forecasting.data import schemas as S
        non_feature = set(S.INDEX_COLS + [
            train_cfg["data"]["target_col"],
            train_cfg["data"]["weight_col"],
        ])
        feature_cols = [c for c in df.columns if c not in non_feature]
        logger.warning(
            "Pipeline not found at {}. Derived {} feature cols by exclusion. "
            "Run `js-features` first for deterministic feature sets.",
            pipeline_path, len(feature_cols),
        )

    # Dispatch to the appropriate trainer
    if model_type in ("lightgbm", "lgbm"):
        _train_lgbm(df, feature_cols, model_cfg, train_cfg, cv, artifact_dir)
    elif model_type == "xgboost":
        _train_xgb(df, feature_cols, model_cfg, train_cfg, cv, artifact_dir)
    elif model_type == "catboost":
        _train_catboost(df, feature_cols, model_cfg, train_cfg, cv, artifact_dir)
    elif model_type == "ensemble":
        _train_ensemble(exp_cfg, train_cfg, df)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Log to MLflow if available
    _try_mlflow_log(exp_cfg, artifact_dir)

    logger.info("Training complete. Artifacts at {}", artifact_dir)


def _train_lgbm(df, feature_cols, model_cfg, train_cfg, cv, artifact_dir):
    from janestreet_forecasting.modeling.train_lgbm import LGBMTrainer
    trainer = LGBMTrainer(
        params=model_cfg["params"],
        cv=cv,
        feature_cols=feature_cols,
        target_col=train_cfg["data"]["target_col"],
        weight_col=train_cfg["data"]["weight_col"],
        date_col=train_cfg["data"]["date_col"],
        artifact_dir=artifact_dir,
        early_stopping_rounds=model_cfg.get("early_stopping", {}).get("rounds", 100),
    )
    trainer.train(df)


def _train_xgb(df, feature_cols, model_cfg, train_cfg, cv, artifact_dir):
    from janestreet_forecasting.modeling.train_xgb import XGBTrainer
    trainer = XGBTrainer(
        params=model_cfg["params"],
        cv=cv,
        feature_cols=feature_cols,
        target_col=train_cfg["data"]["target_col"],
        weight_col=train_cfg["data"]["weight_col"],
        date_col=train_cfg["data"]["date_col"],
        artifact_dir=artifact_dir,
        early_stopping_rounds=model_cfg.get("early_stopping", {}).get("rounds", 100),
    )
    trainer.train(df)


def _train_catboost(df, feature_cols, model_cfg, train_cfg, cv, artifact_dir):
    from janestreet_forecasting.modeling.train_catboost import CatBoostTrainer
    trainer = CatBoostTrainer(
        params=model_cfg["params"],
        cv=cv,
        feature_cols=feature_cols,
        target_col=train_cfg["data"]["target_col"],
        weight_col=train_cfg["data"]["weight_col"],
        date_col=train_cfg["data"]["date_col"],
        artifact_dir=artifact_dir,
        early_stopping_rounds=model_cfg.get("early_stopping", {}).get("rounds", 100),
    )
    trainer.train(df)


def _train_ensemble(exp_cfg, train_cfg, df):
    from janestreet_forecasting.modeling.ensemble import build_ensemble_from_artifacts
    members = exp_cfg["ensemble"]["members"]
    member_dirs = [Path(m["artifact_dir"]) for m in members]
    method = exp_cfg["ensemble"].get("method", "weighted_average")
    ensemble = build_ensemble_from_artifacts(member_dirs, method=method)

    artifact_dir = ARTIFACTS_DIR / exp_cfg["experiment_id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(artifact_dir / "ensemble.pkl")


def _try_mlflow_log(exp_cfg, artifact_dir):
    try:
        import mlflow
        from janestreet_forecasting.settings import settings
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri or str(ARTIFACTS_DIR / "mlruns"))
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run(run_name=exp_cfg["experiment_id"]):
            mlflow.set_tags(exp_cfg.get("tags", {}))
            mlflow.log_artifact(str(artifact_dir / "experiment_config.yaml"))
    except Exception as e:
        logger.debug("MLflow logging skipped: {}", e)


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base in-place."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


if __name__ == "__main__":
    app()
