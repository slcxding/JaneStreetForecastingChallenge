"""
CLI: js-evaluate — Run backtest evaluation on a trained experiment.

Usage:
    js-evaluate --experiment configs/experiments/exp001_lgbm_baseline.yaml
    js-evaluate --experiment configs/experiments/exp001_lgbm_baseline.yaml --report reports/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml
from loguru import logger

from janestreet_forecasting.logging_utils import setup_logging
from janestreet_forecasting.paths import ARTIFACTS_DIR, PROCESSED_DIR, REPORTS_DIR

app = typer.Typer(help="Evaluate a trained model on a held-out dataset.")


@app.command()
def main(
    experiment: Path = typer.Option(
        ...,
        "--experiment", "-e",
        help="Path to experiment YAML config.",
    ),
    data_path: Path = typer.Option(
        PROCESSED_DIR / "features.parquet",
        "--data",
        help="Path to evaluation data parquet.",
    ),
    report_dir: Optional[Path] = typer.Option(
        None,
        "--report",
        help="Directory to write evaluation reports. Defaults to reports/{experiment_id}/.",
    ),
    val_fraction: float = typer.Option(
        0.2,
        "--val-fraction",
        help="Fraction of data to use for evaluation (last N% by date).",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Load OOF predictions (or generate new ones) and run full evaluation."""
    setup_logging(level=log_level)

    import polars as pl

    with open(experiment) as f:
        exp_cfg = yaml.safe_load(f)

    experiment_id = exp_cfg["experiment_id"]
    artifact_dir = ARTIFACTS_DIR / experiment_id

    if not artifact_dir.exists():
        logger.error("Artifact dir {} not found. Run `js-train` first.", artifact_dir)
        raise typer.Exit(code=1)

    eff_report_dir = report_dir or (REPORTS_DIR / experiment_id)
    eff_report_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading evaluation data from {}", data_path)
    df = pl.read_parquet(data_path)

    # Use OOF predictions if available (most accurate estimate)
    oof_path = artifact_dir / "oof_predictions.parquet"
    if oof_path.exists():
        logger.info("Using OOF predictions from {}", oof_path)
        oof_df = pl.read_parquet(oof_path)
        _evaluate_from_oof(oof_df, df, experiment_id, artifact_dir, eff_report_dir)
    else:
        logger.info("No OOF predictions found — evaluating on held-out val split")
        _evaluate_on_holdout(df, exp_cfg, artifact_dir, eff_report_dir, val_fraction)


def _evaluate_from_oof(oof_df, full_df, experiment_id, artifact_dir, report_dir):
    from janestreet_forecasting.data import schemas as S
    from janestreet_forecasting.evaluation.backtest import run_backtest
    from janestreet_forecasting.evaluation.reports import generate_evaluation_report

    import numpy as np

    def predict_fn(df):
        return oof_df["oof_prediction"].to_numpy()

    # Join OOF predictions onto the full df for metric computation
    pred_df = full_df.join(
        oof_df.select([*S.INDEX_COLS, "oof_prediction"]),
        on=S.INDEX_COLS,
        how="inner",
    )

    result = run_backtest(
        predict_fn=lambda df: df["oof_prediction"].to_numpy(),
        df=pred_df.rename({"oof_prediction": "prediction"}),
        feature_cols=[],
    )

    # Load feature importance if available
    fi_path = artifact_dir / "feature_importance.parquet"
    fi = None
    if fi_path.exists():
        fi_df = pl.read_parquet(fi_path)
        fi = dict(zip(fi_df["feature"].to_list(), fi_df["importance_gain"].to_list()))

    generate_evaluation_report(result, fi, report_dir, experiment_id)
    logger.info("Evaluation report written to {}", report_dir)


def _evaluate_on_holdout(df, exp_cfg, artifact_dir, report_dir, val_fraction):
    from janestreet_forecasting.data import schemas as S
    from janestreet_forecasting.data.splits import train_val_date_split
    from janestreet_forecasting.evaluation.backtest import run_backtest
    from janestreet_forecasting.evaluation.reports import generate_evaluation_report

    import pickle

    _, val_df = train_val_date_split(df, val_fraction=val_fraction)

    model_type = _detect_model_type(artifact_dir)
    with open(artifact_dir / "trainer_state.pkl", "rb") as f:
        state = pickle.load(f)
    feature_cols = state["feature_cols"]

    predict_fn = _load_predict_fn(model_type, artifact_dir, feature_cols)

    result = run_backtest(predict_fn, val_df, feature_cols)

    fi_path = artifact_dir / "feature_importance.parquet"
    fi = None
    if fi_path.exists():
        import polars as pl
        fi_df = pl.read_parquet(fi_path)
        fi = dict(zip(fi_df["feature"].to_list(), fi_df["importance_gain"].to_list()))

    generate_evaluation_report(result, fi, report_dir, exp_cfg["experiment_id"])


def _detect_model_type(artifact_dir: Path) -> str:
    if list(artifact_dir.glob("lgbm_fold*.txt")):
        return "lgbm"
    if list(artifact_dir.glob("xgb_fold*.json")):
        return "xgb"
    if list(artifact_dir.glob("catboost_fold*.cbm")):
        return "catboost"
    raise FileNotFoundError(f"No model files found in {artifact_dir}")


def _load_predict_fn(model_type, artifact_dir, feature_cols):
    if model_type == "lgbm":
        from janestreet_forecasting.modeling.train_lgbm import (
            load_lgbm_fold_models, predict_lgbm,
        )
        models = load_lgbm_fold_models(artifact_dir)
        return lambda df: predict_lgbm(models, df, feature_cols)
    elif model_type == "xgb":
        from janestreet_forecasting.modeling.train_xgb import (
            load_xgb_fold_models, predict_xgb,
        )
        models = load_xgb_fold_models(artifact_dir)
        return lambda df: predict_xgb(models, df, feature_cols)
    else:
        from janestreet_forecasting.modeling.train_catboost import (
            load_catboost_fold_models, predict_catboost,
        )
        models = load_catboost_fold_models(artifact_dir)
        return lambda df: predict_catboost(models, df, feature_cols)


if __name__ == "__main__":
    app()
