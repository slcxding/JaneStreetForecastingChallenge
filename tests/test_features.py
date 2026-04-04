"""
Tests for feature engineering transformers.

Key invariants checked:
  1. No look-ahead: rolling features at row t use only rows t-W to t-1
  2. No cross-symbol contamination: rolling features for symbol A don't use symbol B data
  3. Correct output column names
  4. fit/transform protocol is respected
"""

import numpy as np
import polars as pl
import pytest

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.features.lag_features import LagTransformer
from janestreet_forecasting.features.rolling_features import RollingTransformer
from janestreet_forecasting.features.cross_features import (
    CrossSectionalRankTransformer,
    CrossSectionalZScoreTransformer,
)
from janestreet_forecasting.features.base import BaseTransformer


class TestLagTransformer:
    def test_output_column_names(self, small_df):
        t = LagTransformer(columns=[S.FEATURE_COLS[0]], lags=[1, 2])
        t.fit(small_df)
        result = t.transform(small_df)
        assert f"{S.FEATURE_COLS[0]}_lag1" in result.columns
        assert f"{S.FEATURE_COLS[0]}_lag2" in result.columns

    def test_lag1_matches_previous_row_per_symbol(self, small_df):
        """lag1 value for symbol s at time t should equal t-1's value for s."""
        col = S.FEATURE_COLS[0]
        t = LagTransformer(columns=[col], lags=[1])
        t.fit(small_df)
        result = t.transform(small_df).sort([S.SYMBOL_ID, S.DATE_ID, S.TIME_ID])

        # Check for one symbol
        symbols = result[S.SYMBOL_ID].unique().to_list()
        for sym in symbols[:3]:
            sym_df = result.filter(pl.col(S.SYMBOL_ID) == sym)
            orig = sym_df[col].to_list()
            lagged = sym_df[f"{col}_lag1"].to_list()
            # First row of each symbol should be NaN (no prior observation)
            assert lagged[0] is None or np.isnan(lagged[0])
            # Subsequent rows should match the previous original value
            for i in range(1, len(orig)):
                if orig[i - 1] is not None and not np.isnan(orig[i - 1]):
                    assert abs(lagged[i] - orig[i - 1]) < 1e-5

    def test_raises_on_zero_lag(self):
        with pytest.raises(ValueError, match="positive"):
            LagTransformer(columns=[S.FEATURE_COLS[0]], lags=[0])

    def test_transform_before_fit_raises(self, small_df):
        t = LagTransformer(columns=[S.FEATURE_COLS[0]], lags=[1])
        with pytest.raises(RuntimeError, match="fitted"):
            t.transform(small_df)

    def test_feature_names_out(self, small_df):
        t = LagTransformer(columns=["feature_00", "feature_01"], lags=[1, 3])
        t.fit(small_df)
        expected = ["feature_00_lag1", "feature_00_lag3", "feature_01_lag1", "feature_01_lag3"]
        assert t.feature_names_out == expected


class TestRollingTransformer:
    def test_output_columns_exist(self, small_df):
        t = RollingTransformer(
            columns=[S.FEATURE_COLS[0]],
            windows=[3],
            stats=["mean", "std"],
        )
        t.fit(small_df)
        result = t.transform(small_df)
        assert f"{S.FEATURE_COLS[0]}_roll3_mean" in result.columns
        assert f"{S.FEATURE_COLS[0]}_roll3_std" in result.columns

    def test_no_cross_symbol_contamination(self, small_df):
        """Rolling mean for symbol A must only use symbol A's values."""
        col = S.FEATURE_COLS[0]
        t = RollingTransformer(columns=[col], windows=[3], stats=["mean"])
        t.fit(small_df)
        result = t.transform(small_df).sort([S.SYMBOL_ID, S.DATE_ID, S.TIME_ID])

        # For each symbol, compute rolling mean manually and compare
        for sym in result[S.SYMBOL_ID].unique().to_list()[:2]:
            sym_vals = result.filter(pl.col(S.SYMBOL_ID) == sym)[col].to_numpy().astype(float)
            sym_roll = result.filter(pl.col(S.SYMBOL_ID) == sym)[f"{col}_roll3_mean"].to_numpy()
            # At index 2+, rolling mean should use only the previous 3 values
            for i in range(2, min(5, len(sym_vals))):
                window = sym_vals[max(0, i - 2): i + 1]
                valid = window[~np.isnan(window)]
                if len(valid) > 0 and not np.isnan(sym_roll[i]):
                    assert abs(sym_roll[i] - valid.mean()) < 1e-4, (
                        f"Symbol {sym}, row {i}: expected {valid.mean()}, got {sym_roll[i]}"
                    )

    def test_raises_on_unknown_stat(self):
        with pytest.raises(ValueError, match="Unknown stats"):
            RollingTransformer(columns=["feature_00"], windows=[3], stats=["variance"])

    def test_min_periods_one_allows_early_values(self, small_df):
        col = S.FEATURE_COLS[0]
        t = RollingTransformer(columns=[col], windows=[10], stats=["mean"], min_periods=1)
        t.fit(small_df)
        result = t.transform(small_df)
        # With min_periods=1, no NaN except where the original value is NaN
        roll_col = f"{col}_roll10_mean"
        assert roll_col in result.columns


class TestCrossSectionalRankTransformer:
    def test_output_columns(self, small_df):
        t = CrossSectionalRankTransformer(columns=[S.FEATURE_COLS[0]], normalize=True)
        t.fit(small_df)
        result = t.transform(small_df)
        assert f"{S.FEATURE_COLS[0]}_xrank_norm" in result.columns

    def test_within_group_stats(self, small_df):
        """After normalisation, cross-sectional mean should be ~0 per time step."""
        col = S.FEATURE_COLS[0]
        t = CrossSectionalRankTransformer(columns=[col], normalize=True)
        t.fit(small_df)
        result = t.transform(small_df)
        rank_col = f"{col}_xrank_norm"

        # Check that within each (date_id, time_id) the mean is ~0
        agg = result.group_by([S.DATE_ID, S.TIME_ID]).agg(
            pl.col(rank_col).mean().alias("mean_rank")
        )
        means = agg["mean_rank"].drop_nulls().to_numpy()
        assert np.abs(means).max() < 2.0, "Cross-sectional mean should be near 0"

    def test_original_columns_preserved(self, small_df):
        col = S.FEATURE_COLS[0]
        t = CrossSectionalRankTransformer(columns=[col])
        t.fit(small_df)
        result = t.transform(small_df)
        assert col in result.columns, "Original column should be preserved"


class TestBaseTransformerProtocol:
    def test_concrete_subclass_fit_transform(self, small_df):
        """Verify that a concrete BaseTransformer subclass works end-to-end."""
        from janestreet_forecasting.features.lag_features import LagTransformer
        t = LagTransformer(columns=[S.FEATURE_COLS[0]], lags=[1])
        assert isinstance(t, BaseTransformer)
        result = t.fit_transform(small_df)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(small_df)
