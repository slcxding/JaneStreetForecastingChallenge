"""
Data loading utilities.

All loaders return Polars LazyFrames. Callers call .collect() when they need
materialised data. This defers disk reads and allows predicate push-down
(e.g. date_range filters) to avoid loading unused row groups from the 47M-row
training set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
from loguru import logger

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.settings import settings


def load_train(
    path: Path | str,
    date_range: tuple[int, int] | None = None,
    columns: Sequence[str] | None = None,
    max_rows: int | None = None,
) -> pl.LazyFrame:
    """
    Scan train.parquet (Hive-partitioned directory or single file).

    Args:
        path:       data/raw/train.parquet directory or a single .parquet file.
        date_range: Optional (min_date_id, max_date_id) inclusive filter.
        columns:    Column subset. None = all columns.
        max_rows:   Hard row cap for fast local iteration.

    Returns:
        LazyFrame — call .collect() to materialise.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. "
            "Run `make data-download` or use make_synthetic_dataset() for local testing."
        )

    lf = _scan(path, columns)

    if date_range is not None:
        lo, hi = date_range
        lf = lf.filter(pl.col(S.DATE_ID).is_between(lo, hi, closed="both"))

    cap = max_rows or settings.max_rows
    if cap is not None:
        lf = lf.limit(cap)

    logger.debug("Training LazyFrame constructed from {}", path)
    return lf


def load_lag(
    path: Path | str,
    date_range: tuple[int, int] | None = None,
) -> pl.LazyFrame:
    """
    Scan lag.parquet (Hive-partitioned by date_id).

    Columns: date_id, time_id, symbol_id, responder_0_lag_1 .. responder_8_lag_1.
    Join with train on (date_id, time_id, symbol_id) to add previous-day
    responder values as features.

    Args:
        path:       data/raw/lag.parquet directory.
        date_range: Optional (min_date_id, max_date_id) inclusive filter.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lag data not found at {path}.")

    lf = _scan(path)
    if date_range is not None:
        lo, hi = date_range
        lf = lf.filter(pl.col(S.DATE_ID).is_between(lo, hi, closed="both"))
    return lf


def load_test(path: Path | str) -> pl.LazyFrame:
    """
    Scan test.parquet (Hive-partitioned by date_id).

    No responder columns. Adds row_id (submission key) and is_scored (bool).

    Args:
        path: data/raw/test.parquet directory.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test data not found at {path}.")
    return _scan(path)


def load_parquet(path: Path | str, **kwargs) -> pl.LazyFrame:
    """Generic parquet scanner. kwargs forwarded to pl.scan_parquet."""
    return pl.scan_parquet(str(path), **kwargs)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def make_synthetic_dataset(
    n_dates: int = 10,
    n_times_per_date: int = 50,
    n_symbols: int = 20,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generate a deterministic synthetic dataset for tests.

    Preserves the real schema and distributional shape (zero-mean features,
    ~5% nulls, small AR-ish responders) without competition data.

    Args:
        n_dates:          Number of unique date_id values.
        n_times_per_date: Time steps per date.
        n_symbols:        Number of instruments.
        seed:             RNG seed for reproducibility.

    Returns:
        Polars DataFrame with the full training schema.
    """
    rng = np.random.default_rng(seed)
    rows = n_dates * n_times_per_date * n_symbols

    date_ids   = np.repeat(np.arange(n_dates), n_times_per_date * n_symbols)
    time_ids   = np.tile(np.repeat(np.arange(n_times_per_date), n_symbols), n_dates)
    symbol_ids = np.tile(np.arange(n_symbols), n_dates * n_times_per_date)

    data: dict[str, np.ndarray] = {
        S.DATE_ID:   date_ids.astype(np.int32),
        S.TIME_ID:   time_ids.astype(np.int32),
        S.SYMBOL_ID: symbol_ids.astype(np.int32),
        S.WEIGHT:    rng.exponential(1.0, rows).astype(np.float32),
    }

    for col in S.FEATURE_COLS:
        vals = rng.standard_normal(rows).astype(np.float32)
        vals[rng.random(rows) < 0.05] = np.nan  # ~5% missing
        data[col] = vals

    for col in S.RESPONDER_COLS:
        data[col] = (rng.standard_normal(rows) * 0.01).astype(np.float32)

    # Lag columns: shift responder values by one time step, per symbol.
    # Reshape to (time_steps, symbols) to avoid crossing symbol boundaries.
    n_steps = n_dates * n_times_per_date
    for col, lag_col in zip(S.RESPONDER_COLS, S.LAG_COLS):
        src = data[col].reshape(n_steps, n_symbols)
        lagged = np.empty_like(src)
        lagged[0, :]  = np.nan
        lagged[1:, :] = src[:-1, :]
        data[lag_col] = lagged.reshape(rows)

    df = pl.DataFrame(data)
    logger.info(
        "Synthetic dataset created: {} rows × {} cols ({} dates, {} symbols)",
        rows, len(df.columns), n_dates, n_symbols,
    )
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scan(path: Path, columns: Sequence[str] | None = None) -> pl.LazyFrame:
    """
    Scan a parquet file or Hive-partitioned directory.

    Competition directories use two-level nesting (partition_id=N/part-0.parquet
    or date_id=N/part-0.parquet). The **/*.parquet glob captures all files
    without adding spurious Hive partition columns to the schema.
    """
    scan_path = str(path) if path.is_file() else str(path / "**" / "*.parquet")
    return pl.scan_parquet(scan_path, **({"columns": list(columns)} if columns else {}))
