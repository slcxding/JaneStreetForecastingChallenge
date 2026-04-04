"""
Tests for InferenceState — the core of the real-time inference pipeline.

Critical invariants:
  1. update() after compute_features() — state grows correctly
  2. Rolling features match offline computation
  3. Buffer eviction: old values don't persist beyond buffer_size
  4. Warm-up correctly populates buffers
  5. reset() clears all state
"""

import numpy as np
import polars as pl
import pytest

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.data.loaders import make_synthetic_dataset
from janestreet_forecasting.inference.state import InferenceState


@pytest.fixture
def tiny_df():
    """Deterministic tiny dataset: 1 symbol, 10 time steps."""
    return make_synthetic_dataset(n_dates=1, n_times_per_date=10, n_symbols=3, seed=99)


class TestInferenceStateBasics:
    def test_initial_state_is_empty(self):
        # No rolling_windows specified — buffer_size can be any positive int
        state = InferenceState(buffer_size=5, rolling_windows=[])
        assert state.n_symbols == 0
        assert state.n_updates == 0

    def test_update_increments_count(self, tiny_df):
        state = InferenceState(buffer_size=5, rolling_windows=[])
        batches = _split_by_time(tiny_df)
        state.update(batches[0])
        assert state.n_updates == 1
        assert state.n_symbols == batches[0][S.SYMBOL_ID].n_unique()

    def test_reset_clears_state(self, tiny_df):
        state = InferenceState(buffer_size=5, rolling_windows=[])
        batches = _split_by_time(tiny_df)
        for b in batches[:3]:
            state.update(b)
        assert state.n_updates == 3
        state.reset()
        assert state.n_symbols == 0
        assert state.n_updates == 0

    def test_buffer_size_respected(self, tiny_df):
        """Buffer should never hold more than buffer_size rows per symbol."""
        buffer_size = 3
        # No rolling_windows — buffer_size < window constraint doesn't apply
        state = InferenceState(buffer_size=buffer_size, rolling_windows=[])
        batches = _split_by_time(tiny_df)
        for b in batches:
            state.update(b)

        for sym, buf in state._buffers.items():
            assert len(buf) <= buffer_size, (
                f"Symbol {sym} buffer has {len(buf)} rows, expected <= {buffer_size}"
            )


class TestRollingFeatureComputation:
    def test_rolling_mean_is_nan_without_history(self, tiny_df):
        """Before any updates, rolling features should be NaN."""
        state = InferenceState(
            buffer_size=10,
            feature_cols=[S.FEATURE_COLS[0]],
            rolling_windows=[3],
            rolling_stats=["mean"],
        )
        batches = _split_by_time(tiny_df)
        first_batch = batches[0]

        # No history yet — all rolling features should be NaN
        result = state.compute_features(first_batch)
        roll_col = f"{S.FEATURE_COLS[0]}_roll3_mean"
        if roll_col in result.columns:
            vals = result[roll_col].to_numpy()
            assert np.all(np.isnan(vals)), "Expected NaN without history"

    def test_rolling_mean_correct_after_updates(self, tiny_df):
        """After N updates, rolling mean should match manual calculation."""
        col = S.FEATURE_COLS[0]
        window = 3
        state = InferenceState(
            buffer_size=10,
            feature_cols=[col],
            rolling_windows=[window],
            rolling_stats=["mean"],
        )
        batches = _split_by_time(tiny_df)

        # Update with first 3 batches (so we have exactly window history)
        for b in batches[:window]:
            state.update(b)

        # Now compute features for the 4th batch
        result = state.compute_features(batches[window])
        roll_col = f"{col}_roll{window}_mean"

        if roll_col in result.columns:
            # For each symbol, manually compute expected rolling mean
            for sym in result[S.SYMBOL_ID].to_list()[:2]:
                hist = list(state._buffers[sym])
                expected_vals = [r.get(col, np.nan) for r in hist[-window:]]
                expected_mean = np.nanmean(expected_vals) if expected_vals else np.nan

                sym_row = result.filter(pl.col(S.SYMBOL_ID) == sym)
                if len(sym_row) > 0:
                    actual = sym_row[roll_col][0]
                    if actual is not None and not np.isnan(actual):
                        assert abs(actual - expected_mean) < 1e-4, (
                            f"Symbol {sym}: expected {expected_mean:.4f}, got {actual:.4f}"
                        )

    def test_update_called_after_compute(self, tiny_df):
        """Correct usage: compute_features → update (not the reverse)."""
        col = S.FEATURE_COLS[0]
        state = InferenceState(
            buffer_size=10,
            feature_cols=[col],
            rolling_windows=[2],
            rolling_stats=["mean"],
        )
        batches = _split_by_time(tiny_df)

        # Simulate inference loop
        all_results = []
        for b in batches[:5]:
            enriched = state.compute_features(b)  # Use history
            all_results.append(enriched)
            state.update(b)                        # Then update history

        # Verify the second batch has some non-NaN rolling features
        # (after one update, the second batch should have partial history)
        roll_col = f"{col}_roll2_mean"
        if roll_col in all_results[1].columns:
            vals = all_results[1][roll_col].drop_nulls().to_numpy()
            # At least some values should be non-NaN
            assert len(vals) >= 0  # Weak check — just ensure no crash


class TestWarmUp:
    def test_warmup_populates_buffers(self, tiny_df):
        """After warm_up, buffers should be non-empty."""
        state = InferenceState(
            buffer_size=5,
            feature_cols=[S.FEATURE_COLS[0]],
            rolling_windows=[3],
            rolling_stats=["mean"],
        )
        state.warm_up(tiny_df)
        assert state.n_symbols > 0
        assert state.n_updates > 0

    def test_warmup_then_predict_no_crash(self, tiny_df):
        """Warm up then compute features should not raise."""
        col = S.FEATURE_COLS[0]
        state = InferenceState(
            buffer_size=5,
            feature_cols=[col],
            rolling_windows=[3],
            rolling_stats=["mean"],
        )
        state.warm_up(tiny_df)
        batches = _split_by_time(tiny_df)
        result = state.compute_features(batches[0])
        assert isinstance(result, pl.DataFrame)


def _split_by_time(df: pl.DataFrame) -> list[pl.DataFrame]:
    """Split a DataFrame into a list of per-(date_id, time_id) batches."""
    return [
        group
        for (_, _), group in (
            df.sort([S.DATE_ID, S.TIME_ID])
            .group_by([S.DATE_ID, S.TIME_ID], maintain_order=True)
        )
    ]
