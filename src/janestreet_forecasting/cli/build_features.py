"""
CLI: js-features — Build feature set from interim data.

Reads data/interim/train.parquet, applies the feature pipeline defined in
the features config, writes to data/processed/features.parquet.

Usage:
    js-features --config configs/features/default.yaml
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from loguru import logger

from janestreet_forecasting.features.pipelines import build_default_pipeline
from janestreet_forecasting.logging_utils import setup_logging
from janestreet_forecasting.paths import INTERIM_DIR, PROCESSED_DIR

app = typer.Typer(help="Build feature set from interim training data.")


@app.command()
def main(
    config: Path = typer.Option(
        Path("configs/features/default.yaml"),
        "--config", "-c",
        help="Path to features config YAML.",
    ),
    input_path: Path = typer.Option(
        INTERIM_DIR / "train.parquet",
        "--input",
        help="Path to interim training parquet.",
    ),
    output_path: Path = typer.Option(
        PROCESSED_DIR / "features.parquet",
        "--output",
        help="Path to write feature parquet.",
    ),
    pipeline_path: Path = typer.Option(
        PROCESSED_DIR / "feature_pipeline.pkl",
        "--pipeline",
        help="Path to save fitted pipeline.",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Fit and apply feature pipeline, saving features + pipeline."""
    setup_logging(level=log_level)

    import polars as pl

    with open(config) as f:
        cfg = yaml.safe_load(f)

    if not input_path.exists():
        logger.error("Input not found at {}. Run `js-prepare` first.", input_path)
        raise typer.Exit(code=1)

    logger.info("Reading interim data from {}", input_path)
    df = pl.read_parquet(input_path)
    logger.info("Loaded {} rows", len(df))

    pipeline = build_default_pipeline(cfg)

    logger.info("Fitting feature pipeline...")
    df_out = pipeline.fit_transform(df)

    logger.info(
        "Feature pipeline complete: {} → {} columns",
        len(df.columns), len(df_out.columns),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.write_parquet(output_path)
    logger.info("Features written to {}", output_path)

    pipeline.save(pipeline_path)
    logger.info("Pipeline saved to {}", pipeline_path)


if __name__ == "__main__":
    app()
