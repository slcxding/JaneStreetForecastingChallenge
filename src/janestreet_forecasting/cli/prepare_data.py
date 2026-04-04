"""
CLI: js-prepare — Validate raw data and write to data/interim/.

Usage:
    js-prepare --config configs/data/default.yaml
    js-prepare --config configs/data/default.yaml --sample 100000
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml
from loguru import logger

from janestreet_forecasting.data.loaders import load_train
from janestreet_forecasting.data.validation import validate_dataframe
from janestreet_forecasting.logging_utils import setup_logging
from janestreet_forecasting.paths import INTERIM_DIR, RAW_DIR

app = typer.Typer(help="Validate and prepare raw competition data.")


@app.command()
def main(
    config: Path = typer.Option(
        Path("configs/data/default.yaml"),
        "--config", "-c",
        help="Path to data config YAML.",
    ),
    sample: Optional[int] = typer.Option(
        None,
        "--sample",
        help="Limit to N rows for quick iteration.",
    ),
    output_dir: Path = typer.Option(
        INTERIM_DIR,
        "--output-dir",
        help="Directory to write prepared data.",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Validate raw data and save to interim directory."""
    setup_logging(level=log_level)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    raw_path = Path(cfg["paths"]["raw_dir"]) / cfg["paths"]["train_file"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data from {}", raw_path)
    lf = load_train(raw_path, max_rows=sample)
    df = lf.collect()

    logger.info("Loaded {} rows × {} columns", len(df), len(df.columns))

    # Drop non-feature responder columns (prevent leakage)
    drop_cols = [c for c in cfg.get("loading", {}).get("drop_columns", []) if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)
        logger.info("Dropped {} leakage-prone columns", len(drop_cols))

    # Validate
    strict = cfg.get("validation", {}).get("strict", True)
    report = validate_dataframe(df, strict=strict)

    if not report.passed:
        logger.error("Validation failed. Fix data issues before proceeding.")
        raise typer.Exit(code=1)

    # Write to interim
    out_path = output_dir / "train.parquet"
    df.write_parquet(out_path)
    logger.info("Data prepared and saved to {} ({} rows)", out_path, len(df))


if __name__ == "__main__":
    app()
