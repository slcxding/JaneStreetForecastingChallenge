"""
Shared test fixtures.

Provides a synthetic DataFrame that covers the full competition schema.
Tests use this instead of real data so they run without Kaggle downloads.
"""

import numpy as np
import polars as pl
import pytest

from janestreet_forecasting.data.loaders import make_synthetic_dataset
from janestreet_forecasting.data import schemas as S


@pytest.fixture(scope="session")
def synthetic_df() -> pl.DataFrame:
    """Full-schema synthetic dataset, session-scoped for speed."""
    return make_synthetic_dataset(n_dates=20, n_times_per_date=10, n_symbols=15, seed=0)


@pytest.fixture(scope="session")
def small_df() -> pl.DataFrame:
    """Tiny dataset for fast unit tests."""
    return make_synthetic_dataset(n_dates=5, n_times_per_date=5, n_symbols=5, seed=1)


@pytest.fixture
def feature_cols() -> list[str]:
    """Default feature column list: raw features + lag_0 columns."""
    return S.FEATURE_COLS + S.LAG_COLS
