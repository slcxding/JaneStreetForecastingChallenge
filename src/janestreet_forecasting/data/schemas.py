"""
Column name constants and schema definitions for the Jane Street dataset.

Centralising column names here prevents typos and makes refactoring safe.
All other modules import from here rather than hard-coding strings.

Competition data structure (training set):
  - date_id    : int — chronological date index (groups rows into "days")
  - time_id    : int — within-date time step index
  - symbol_id  : int — anonymised instrument identifier
  - feature_00 — feature_78 : float — anonymised market features (79 total)
  - responder_0 — responder_8 : float — response variables; responder_6 is
                                the primary prediction target
  - weight     : float — per-row sample weight (used in the scoring metric)

At inference time the Kaggle API also provides:
  - responder_0_lag_0 — responder_8_lag_0 : float — previous time_id's
    responder values for each symbol (safe to use as features)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Index / key columns
# ---------------------------------------------------------------------------
DATE_ID = "date_id"
TIME_ID = "time_id"
SYMBOL_ID = "symbol_id"
WEIGHT = "weight"

INDEX_COLS: list[str] = [DATE_ID, TIME_ID, SYMBOL_ID]

# ---------------------------------------------------------------------------
# Raw feature columns
# ---------------------------------------------------------------------------
N_FEATURES = 79
FEATURE_COLS: list[str] = [f"feature_{i:02d}" for i in range(N_FEATURES)]

# ---------------------------------------------------------------------------
# Responder / target columns
# ---------------------------------------------------------------------------
N_RESPONDERS = 9
RESPONDER_COLS: list[str] = [f"responder_{i}" for i in range(N_RESPONDERS)]

# Primary prediction target — this is what the competition scores
TARGET_COL: str = "responder_6"

# Lag versions of responders — provided by the Kaggle API at inference time.
# These are the *previous* time_id's values, so they're safe to use as
# features (no look-ahead).
LAG_SUFFIX = "_lag_0"
LAG_COLS: list[str] = [f"{r}{LAG_SUFFIX}" for r in RESPONDER_COLS]

# ---------------------------------------------------------------------------
# All columns expected in the raw training dataset
# ---------------------------------------------------------------------------
TRAIN_COLS: list[str] = (
    INDEX_COLS + FEATURE_COLS + RESPONDER_COLS + [WEIGHT] + LAG_COLS
)

# Columns that are safe to use as model features
# (excludes current-step responders to prevent leakage)
DEFAULT_FEATURE_COLS: list[str] = FEATURE_COLS + LAG_COLS

# ---------------------------------------------------------------------------
# Polars dtypes for the raw dataset (speeds up parquet reading)
# ---------------------------------------------------------------------------
import polars as pl  # noqa: E402 — import after constants so module is light

RAW_SCHEMA: dict[str, pl.DataType] = {
    DATE_ID: pl.Int32,
    TIME_ID: pl.Int32,
    SYMBOL_ID: pl.Int32,
    WEIGHT: pl.Float32,
    **{c: pl.Float32 for c in FEATURE_COLS},
    **{c: pl.Float32 for c in RESPONDER_COLS},
    **{c: pl.Float32 for c in LAG_COLS},
}
