"""
Column name constants and schema definitions for the Jane Street dataset.

Centralising column names here prevents typos and makes refactoring safe.
All other modules import from here rather than hard-coding strings.

Actual competition data files (verified against downloaded data):

  data/raw/train.parquet/          — 47.1M rows, 10 Hive partitions
    partition_id=0/part-0.parquet  — ~1.9M rows, date_id 0–169
    ...
    partition_id=9/part-0.parquet

    Columns: date_id, time_id, symbol_id, weight,
             feature_00 .. feature_78 (79 features, all Float32),
             responder_0 .. responder_8 (9 targets, all Float32)

    Key facts:
      - date_id:   0 – 1,689  (≈ 1,690 trading days, ~6.7 years)
      - time_id:   0 – 848    (849 intraday time steps per day)
      - symbol_id: 0 – 19     (exactly 20 anonymised instruments)
      - weight:    positive float; used in the scoring metric

  data/raw/lag.parquet/            — Hive-partitioned by date_id
    date_id=0/part-0.parquet       — 39 rows (1 per symbol × time step)
    ...

    Columns: date_id, time_id, symbol_id,
             responder_0_lag_1 .. responder_8_lag_1

    These are the PREVIOUS DAY's responder values. Join with train on
    (date_id, time_id, symbol_id) to get day-over-day lag features.

  data/raw/test.parquet/           — Hive-partitioned by date_id
    date_id=0/part-0.parquet

    Columns: row_id (submission key), date_id, time_id, symbol_id,
             weight, is_scored, feature_00 .. feature_78
    NO responder columns — those are what you predict.

  data/raw/features.csv            — 79 rows, feature metadata
    Columns: feature, tag_0 .. tag_16 (17 anonymous boolean tags)
    Use tags to group related features for targeted rolling/cross-sectional
    feature engineering.

  data/raw/responders.csv          — 9 rows, responder metadata
    Columns: responder, tag_0 .. tag_4 (5 anonymous boolean tags)
    responder_6 is the primary target; the tags hint at groupings (e.g.
    long/short horizons, asset classes) that can inform multi-task learning.

  data/raw/sample_submission.csv   — Columns: row_id, responder_6
    All-zero baseline. Your submission must match this format exactly.
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

# Primary prediction target
TARGET_COL: str = "responder_6"

# ---------------------------------------------------------------------------
# Lag columns from lag.parquet
# ---------------------------------------------------------------------------
# The competition distributes previous-day responder values in a separate
# lag.parquet file (Hive-partitioned by date_id). The suffix "_lag_1" means
# "1 calendar day back".  Join on (date_id, time_id, symbol_id).
#
# At inference time the Kaggle API provides these as the previous TIME STEP's
# responders (within-day lag, effectively "_lag_0" in API parlance). The
# naming difference is a Kaggle API implementation detail — both represent
# "the most recent prior observation" at different granularities.
LAG_SUFFIX = "_lag_1"           # suffix used in lag.parquet on disk
LAG_SUFFIX_API = "_lag_0"       # suffix used by the Kaggle inference API
LAG_COLS: list[str] = [f"{r}{LAG_SUFFIX}" for r in RESPONDER_COLS]
LAG_COLS_API: list[str] = [f"{r}{LAG_SUFFIX_API}" for r in RESPONDER_COLS]

# ---------------------------------------------------------------------------
# Test-only columns (not present in training data)
# ---------------------------------------------------------------------------
ROW_ID = "row_id"       # submission key — join with sample_submission.csv
IS_SCORED = "is_scored" # boolean; only scored=True rows count toward LB

# ---------------------------------------------------------------------------
# Column groupings for convenience
# ---------------------------------------------------------------------------
TRAIN_COLS: list[str] = INDEX_COLS + [WEIGHT] + FEATURE_COLS + RESPONDER_COLS

# Safe model inputs: raw features + day-lag responders.
# Do NOT include RESPONDER_COLS here — they are current-step values (leakage).
DEFAULT_FEATURE_COLS: list[str] = FEATURE_COLS + LAG_COLS

# ---------------------------------------------------------------------------
# Polars dtypes matching actual parquet schema
# ---------------------------------------------------------------------------
import polars as pl  # noqa: E402 — import after constants so module is light

RAW_SCHEMA: dict[str, pl.DataType] = {
    DATE_ID: pl.Int16,      # actual dtype in competition parquet
    TIME_ID: pl.Int16,
    SYMBOL_ID: pl.Int8,
    WEIGHT: pl.Float32,
    **{c: pl.Float32 for c in FEATURE_COLS},
    **{c: pl.Float32 for c in RESPONDER_COLS},
}

LAG_SCHEMA: dict[str, pl.DataType] = {
    DATE_ID: pl.Int16,
    TIME_ID: pl.Int16,
    SYMBOL_ID: pl.Int8,
    **{c: pl.Float32 for c in LAG_COLS},
}
