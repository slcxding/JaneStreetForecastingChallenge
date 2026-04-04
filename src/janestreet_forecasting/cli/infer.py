"""
CLI: js-infer — Run local inference simulation or generate Kaggle submission.

Subcommands:
    js-infer serve-local  — Run local batch inference simulation
    js-infer info         — Show loaded model and feature info
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml
from loguru import logger

from janestreet_forecasting.logging_utils import setup_logging
from janestreet_forecasting.paths import ARTIFACTS_DIR, PROCESSED_DIR, REPORTS_DIR

app = typer.Typer(help="Inference commands: local simulation and Kaggle submission.")


@app.command("serve-local")
def serve_local(
    experiment: Path = typer.Option(
        ...,
        "--experiment", "-e",
        help="Path to experiment YAML config.",
    ),
    data: Path = typer.Option(
        PROCESSED_DIR / "features.parquet",
        "--data",
        help="Path to data to simulate inference on.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Path to write predictions parquet.",
    ),
    max_batches: Optional[int] = typer.Option(
        None,
        "--max-batches",
        help="Stop after N batches (useful for quick testing).",
    ),
    model_type: str = typer.Option(
        "lgbm",
        "--model-type",
        help="Model type: lgbm | xgb | catboost",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Simulate real-time inference batch-by-batch, tracking rolling state."""
    setup_logging(level=log_level)

    with open(experiment) as f:
        exp_cfg = yaml.safe_load(f)

    experiment_id = exp_cfg["experiment_id"]
    artifact_dir = ARTIFACTS_DIR / experiment_id

    if not artifact_dir.exists():
        logger.error("Artifact dir not found at {}. Run `js-train` first.", artifact_dir)
        raise typer.Exit(code=1)

    from janestreet_forecasting.inference.predict import PredictorPipeline
    from janestreet_forecasting.inference.serve_local import LocalInferenceSimulator

    loaders = {
        "lgbm": PredictorPipeline.load_lgbm,
        "xgb": PredictorPipeline.load_xgb,
        "catboost": PredictorPipeline.load_catboost,
    }
    if model_type not in loaders:
        logger.error("Unknown model_type: {}. Choose from {}", model_type, list(loaders))
        raise typer.Exit(code=1)

    logger.info("Loading {} predictor from {}", model_type, artifact_dir)
    predictor = loaders[model_type](artifact_dir)

    out_path = output or (REPORTS_DIR / experiment_id / "local_inference_predictions.parquet")

    simulator = LocalInferenceSimulator(
        predictor=predictor,
        data_path=data,
        max_batches=max_batches,
    )

    result = simulator.run(output_path=out_path)

    logger.info("\n{}", result.summary())


@app.command("info")
def info(
    experiment: Path = typer.Option(
        ...,
        "--experiment", "-e",
    ),
    model_type: str = typer.Option("lgbm", "--model-type"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Show model info: feature count, fold count, CV scores."""
    setup_logging(level=log_level)

    import pickle

    with open(experiment) as f:
        exp_cfg = yaml.safe_load(f)

    artifact_dir = ARTIFACTS_DIR / exp_cfg["experiment_id"]
    state_path = artifact_dir / "trainer_state.pkl"

    if not state_path.exists():
        logger.error("trainer_state.pkl not found in {}.", artifact_dir)
        raise typer.Exit(code=1)

    with open(state_path, "rb") as f:
        state = pickle.load(f)

    from rich.console import Console
    from rich.table import Table
    console = Console()

    console.print(f"\n[bold cyan]Experiment:[/] {exp_cfg['experiment_id']}")
    console.print(f"[bold]Features:[/] {len(state['feature_cols'])}")
    console.print(f"[bold]Target:[/] {state['target_col']}")

    if state.get("fold_scores"):
        table = Table(title="CV Fold Scores")
        table.add_column("Fold")
        for metric in state["fold_scores"][0]:
            table.add_column(metric)
        for i, scores in enumerate(state["fold_scores"]):
            table.add_row(str(i + 1), *[f"{v:.4f}" for v in scores.values()])
        console.print(table)


if __name__ == "__main__":
    app()
