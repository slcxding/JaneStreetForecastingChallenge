"""
Tests for time-aware cross-validation splits.

These are the most critical tests in the project. A bug here means training
on future data and completely invalidating all results.
"""

import numpy as np
import polars as pl
import pytest

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.data.splits import (
    PurgedGroupKFold,
    WalkForwardSplit,
    train_val_date_split,
)


class TestPurgedGroupKFoldForwardOnly:
    """
    Tests for the default forward_only=True mode, which is what we use in
    production.  Every fold's training set must be temporally prior to its
    validation set.
    """

    def _cv(self, n_splits=3, purge=1, embargo=2) -> PurgedGroupKFold:
        return PurgedGroupKFold(
            n_splits=n_splits,
            purge_days=purge,
            embargo_days=embargo,
            forward_only=True,
            min_train_dates=3,
        )

    def test_no_overlap_between_train_and_val(self, synthetic_df):
        for fold in self._cv().split(synthetic_df):
            overlap = np.intersect1d(fold.train_dates, fold.val_dates)
            assert len(overlap) == 0, f"Fold {fold.fold_idx}: date overlap {overlap}"

    def test_max_train_date_strictly_before_min_val_date(self, synthetic_df):
        """The core temporal invariant: no future leakage."""
        for fold in self._cv().split(synthetic_df):
            assert fold.train_dates.max() < fold.val_dates.min(), (
                f"Fold {fold.fold_idx}: max_train={fold.train_dates.max()} "
                f">= min_val={fold.val_dates.min()} — future data used for training!"
            )

    def test_purge_removes_boundary_dates(self, synthetic_df):
        """The purge zone [min_val - purge_days, min_val) must be absent from train."""
        purge_days = 3
        cv = PurgedGroupKFold(
            n_splits=3, purge_days=purge_days, embargo_days=0,
            forward_only=True, min_train_dates=3,
        )
        for fold in cv.split(synthetic_df):
            min_val = int(fold.val_dates.min())
            purge_zone = set(range(min_val - purge_days, min_val))
            bad = purge_zone & set(fold.train_dates.tolist())
            assert len(bad) == 0, (
                f"Fold {fold.fold_idx}: purge zone dates {bad} found in train"
            )

    def test_folds_are_chronologically_ordered(self, synthetic_df):
        """Later folds should test on later dates (monotonically increasing)."""
        folds = list(self._cv().split(synthetic_df))
        assert len(folds) >= 2, "Need at least 2 folds to check ordering"
        for i in range(1, len(folds)):
            assert folds[i].val_dates.min() > folds[i - 1].val_dates.max(), (
                f"Folds {i-1} and {i} test sets are not in chronological order"
            )

    def test_train_set_grows_across_folds(self, synthetic_df):
        """Each fold should have more training data than the previous (expanding window)."""
        folds = list(self._cv().split(synthetic_df))
        for i in range(1, len(folds)):
            assert len(folds[i].train_dates) >= len(folds[i - 1].train_dates), (
                f"Fold {i} has fewer training dates than fold {i-1}"
            )

    def test_all_val_rows_covered_at_most_once(self, synthetic_df):
        """No row should appear in more than one fold's validation set.

        In forward_only=True mode, early folds are skipped when there is
        insufficient training history, so early-date rows may have coverage=0.
        The hard invariant is coverage <= 1 (no leakage between folds).
        """
        n = len(synthetic_df)
        val_coverage = np.zeros(n, dtype=int)
        for fold in self._cv(n_splits=4).split(synthetic_df):
            val_coverage[fold.val_idx] += 1
        assert (val_coverage <= 1).all(), (
            "Some rows appear in multiple val sets — fold val sets overlap."
        )
        assert val_coverage.sum() > 0, "No rows covered by any validation fold."

    def test_early_folds_skipped_when_insufficient_history(self):
        """Folds with fewer training dates than min_train_dates are skipped."""
        # 12 unique dates, n_splits=4 → each test chunk ≈ 3 dates
        # Fold 0 test = dates [0,1,2], train = [] (nothing before) → skipped
        df = pl.DataFrame({
            "date_id": list(range(12)) * 2,
            "time_id": [0] * 24,
            "symbol_id": [0] * 12 + [1] * 12,
        })
        cv = PurgedGroupKFold(
            n_splits=4, purge_days=1, embargo_days=1,
            forward_only=True, min_train_dates=5,
        )
        folds = list(cv.split(df))
        # Some early folds should be skipped (not all 4 should appear)
        assert len(folds) < 4 or all(
            len(f.train_dates) >= 5 for f in folds
        ), "Folds with < 5 training dates should be skipped"

    def test_raises_with_too_few_dates(self):
        tiny_df = pl.DataFrame({
            "date_id": [0, 1, 2],
            "time_id": [0, 0, 0],
            "symbol_id": [0, 0, 0],
        })
        cv = PurgedGroupKFold(n_splits=5)
        with pytest.raises(ValueError, match="Not enough unique dates"):
            list(cv.split(tiny_df))


class TestPurgedGroupKFoldLPStyle:
    """
    Tests for forward_only=False (López de Prado style).
    Here we only check that train ∩ val = {} — the temporal ordering
    constraint does NOT apply in this mode by design.
    """

    def _cv(self) -> PurgedGroupKFold:
        return PurgedGroupKFold(
            n_splits=3, purge_days=2, embargo_days=2,
            forward_only=False, min_train_dates=0,
        )

    def test_no_train_val_date_overlap(self, synthetic_df):
        for fold in self._cv().split(synthetic_df):
            overlap = np.intersect1d(fold.train_dates, fold.val_dates)
            assert len(overlap) == 0

    def test_purge_zone_absent_from_train(self, synthetic_df):
        purge_days = 2
        cv = PurgedGroupKFold(
            n_splits=3, purge_days=purge_days, embargo_days=0,
            forward_only=False, min_train_dates=0,
        )
        for fold in cv.split(synthetic_df):
            min_val = int(fold.val_dates.min())
            purge_zone = set(range(min_val - purge_days, min_val))
            bad = purge_zone & set(fold.train_dates.tolist())
            assert len(bad) == 0


class TestWalkForwardSplit:
    def test_expanding_train_window(self, synthetic_df):
        """Each fold's training set should be larger than the previous."""
        cv = WalkForwardSplit(min_train_days=5, test_window=3, step_days=2)
        folds = list(cv.split(synthetic_df))
        assert len(folds) >= 2, "Expected at least 2 walk-forward folds"
        for i in range(1, len(folds)):
            assert len(folds[i].train_dates) >= len(folds[i - 1].train_dates)

    def test_no_train_val_overlap(self, synthetic_df):
        cv = WalkForwardSplit(min_train_days=5, test_window=3, step_days=2, embargo_days=1)
        for fold in cv.split(synthetic_df):
            assert len(np.intersect1d(fold.train_dates, fold.val_dates)) == 0

    def test_max_train_before_min_val(self, synthetic_df):
        cv = WalkForwardSplit(min_train_days=5, test_window=3, step_days=2, embargo_days=1)
        for fold in cv.split(synthetic_df):
            assert fold.train_dates.max() < fold.val_dates.min()


class TestSimpleSplit:
    def test_temporal_order(self, synthetic_df):
        train_df, val_df = train_val_date_split(synthetic_df, val_fraction=0.2)
        assert train_df[S.DATE_ID].max() < val_df[S.DATE_ID].min()

    def test_sizes(self, synthetic_df):
        train_df, val_df = train_val_date_split(synthetic_df, val_fraction=0.2)
        total = len(train_df) + len(val_df)
        assert total <= len(synthetic_df)
        assert len(val_df) > 0
        assert len(train_df) > 0
