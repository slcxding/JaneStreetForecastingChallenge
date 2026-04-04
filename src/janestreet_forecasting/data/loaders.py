"""
Data loading utilities.

All loaders return Polars LazyFrames by default. Callers collect() when they
need materialised data. This avoids loading the full ~47M-row training set
into memory unless explicitly requested.

Design note — why Polars?
  Polars uses Apache Arrow memory layout (columnar, cache-friendly) and a
  lazy query planner that can push down filters before reading from disk.
  On a 47M-row dataset this translates to 3–8× faster load + transform vs
  pandas.  The NumPy array at the model training boundary costs one copy but
  is worth the upstream speed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
from loguru import logger

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.settings import settings


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_train(
    path: Path | str,
    date_range: tuple[int, int] | None = None,
    columns: Sequence[str] | None = None,
    max_rows: int | None = None,
) -> pl.LazyFrame:
    """
    Load the training parquet (Hive-partitioned directory or single file).

    The competition's train.parquet is stored as a Hive-partitioned directory:
        data/raw/train.parquet/
            partition_id=0/part-0.parquet   (~1.9M rows, dates 0–169)
            partition_id=1/part-0.parquet
            ...
            partition_id=9/part-0.parquet

    Polars scans all partitions lazily; predicate push-down limits disk I/O
    when date_range is specified.

    Args:
        path:       Directory (data/raw/train.parquet) or a single .parquet file.
        date_range: Optional (min_date_id, max_date_id) inclusive filter.
        columns:    Subset of columns to load. None = all columns.
        max_rows:   Hard row cap (useful for fast local iteration).

    Returns:
        LazyFrame — call .collect() to materialise.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. "
            "Download with `make data-download` or use make_synthetic_dataset() "
            "for testing without competition data."
        )

    effective_max = max_rows or settings.max_rows
    lf = _read_parquet(path, columns)

    if date_range is not None:
        lo, hi = date_range
        lf = lf.filter(pl.col(S.DATE_ID).is_between(lo, hi, closed="both"))

    if effective_max is not None:
        lf = lf.limit(effective_max)

    logger.debug("Training LazyFrame constructed from {}", path)
    return lf


def load_lag(
    path: Path | str,
    date_range: tuple[int, int] | None = None,
) -> pl.LazyFrame:
    """
    Load the supplementary lag.parquet file.

    The competition provides daily lagged responders in a separate Hive-
    partitioned directory:
        data/raw/lag.parquet/
            date_id=0/part-0.parquet
            date_id=1/part-0.parquet
            ...

    Columns: date_id, time_id, symbol_id,
             responder_0_lag_1 .. responder_8_lag_1

    These are the previous calendar day's responder values for each symbol —
    a powerful predictive signal since markets have day-over-day autocorrelation.

    Usage (join with train):
        train_lf = load_train(train_path)
        lag_lf   = load_lag(lag_path)
        combined = train_lf.join(lag_lf, on=[DATE_ID, TIME_ID, SYMBOL_ID], how="left")

    Args:
        path:       data/raw/lag.parquet directory.
        date_range: Optional (min_date_id, max_date_id) inclusive filter.

    Returns:
        LazyFrame with lag columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lag data not found at {path}.")

    lf = _read_parquet(path)
    if date_range is not None:
        lo, hi = date_range
        lf = lf.filter(pl.col(S.DATE_ID).is_between(lo, hi, closed="both"))
    return lf


def load_test(path: Path | str) -> pl.LazyFrame:
    """
    Load the test parquet (Hive-partitioned by date_id).

    Test data does NOT contain responder columns. It adds:
      - row_id    : submission identifier (join key for sample_submission.csv)
      - is_scored : boolean — only scored rows contribute to the leaderboard

    Args:
        path: data/raw/test.parquet directory.

    Returns:
        LazyFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test data not found at {path}.")
    return _read_parquet(path)


def load_parquet(path: Path | str, **kwargs) -> pl.LazyFrame:
    """Generic parquet loader. kwargs are forwarded to pl.scan_parquet."""
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
    Generate a realistic synthetic dataset for unit tests and smoke-tests.

    The synthetic data preserves the column schema and rough distributional
    shape of the real dataset but contains no proprietary information.

    Args:
        n_dates:           Number of unique date_id values.
        n_times_per_date:  Number of time_id values per date.
        n_symbols:         Number of unique symbol_id values.
        seed:              RNG seed for reproducibility.

    Returns:
        Materialised Polars DataFrame with the full training schema.
    """
    rng = np.random.default_rng(seed)

    rows = n_dates * n_times_per_date * n_symbols
    date_ids = np.repeat(np.arange(n_dates), n_times_per_date * n_symbols)
    time_ids = np.tile(
        np.repeat(np.arange(n_times_per_date), n_symbols), n_dates
    )
    symbol_ids = np.tile(np.arange(n_symbols), n_dates * n_times_per_date)

    data: dict[str, np.ndarray] = {
        S.DATE_ID: date_ids.astype(np.int32),
        S.TIME_ID: time_ids.astype(np.int32),
        S.SYMBOL_ID: symbol_ids.astype(np.int32),
        S.WEIGHT: rng.exponential(1.0, rows).astype(np.float32),
    }

    # Features — approximately zero-mean, unit-variance, with some NaNs
    for col in S.FEATURE_COLS:
        vals = rng.standard_normal(rows).astype(np.float32)
        mask = rng.random(rows) < 0.05  # ~5% missing rate
        vals[mask] = np.nan
        data[col] = vals

    # Responders — small AR(1)-ish signals
    for col in S.RESPONDER_COLS:
        data[col] = (rng.standard_normal(rows) * 0.01).astype(np.float32)

    # Lag columns — shift responders by one TIME STEP per symbol.
    # Must respect symbol boundaries: lag for symbol s at time t is the
    # value for symbol s at time t-1, NOT the value of the preceding row
    # in the flat array (which belongs to a different symbol).
    #
    # Layout: the array is ordered as symbol_0, symbol_1, ..., symbol_N-1
    # repeated for each (date, time) combination. We shift within each
    # symbol's contiguous block.
    n_steps = n_dates * n_times_per_date  # total time steps
    for col, lag_col in zip(S.RESPONDER_COLS, S.LAG_COLS):
        src = data[col].reshape(n_steps, n_symbols)  # (time_steps, symbols)
        lagged = np.empty_like(src)
        lagged[0, :] = np.nan          # first time step has no prior
        lagged[1:, :] = src[:-1, :]   # each row gets the previous time step's value
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


def _read_parquet(path: Path, columns: Sequence[str] | None = None) -> pl.LazyFrame:
    """
    Scan a parquet file or Hive-partitioned directory.

    For directories the competition uses nested Hive partitioning:
        train.parquet/partition_id=N/part-0.parquet
        lag.parquet/date_id=N/part-0.parquet

    Using **/*.parquet captures all files two levels deep without pulling in
    the Hive partition column as an extra column (unlike hive_partitioning=True,
    which would add a spurious 'partition_id' or 'date_id' column that
    conflicts with the 'date_id' already in the payload).
    """
    if path.is_file():
        scan_path = str(path)
    else:
        scan_path = str(path / "**" / "*.parquet")
    kwargs: dict = {}
    if columns:
        kwargs["columns"] = list(columns)
    return pl.scan_parquet(scan_path, **kwargs)
