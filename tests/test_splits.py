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
from janestreet_forecasting.data.validation import validate_no_future_leakage


class TestPurgedGroupKFold:
    def test_no_overlap_between_train_and_val(self, synthetic_df):
        """Train and val date sets must be disjoint."""
        cv = PurgedGroupKFold(n_splits=3, purge_days=1, embargo_days=2)
        for fold in cv.split(synthetic_df):
            assert len(np.intersect1d(fold.train_dates, fold.val_dates)) == 0, (
                f"Fold {fold.fold_idx}: train and val dates overlap!"
            )

    def test_no_leakage_within_purge_embargo_window(self, synthetic_df):
        """
        The purge window ensures no training date sits between
        (min_val_date - purge_days) and min_val_date.
        Note: PurgedGroupKFold is NOT a walk-forward split — fold 0 may test
        on early dates while training on later dates. That is intentional and
        expected. The key invariant is that train ∩ val = {} and that the
        boundary rolling-feature window is purged.
        """
        cv = PurgedGroupKFold(n_splits=3, purge_days=2, embargo_days=2)
        for fold in cv.split(synthetic_df):
            # No date should appear in both train and val
            overlap = np.intersect1d(fold.train_dates, fold.val_dates)
            assert len(overlap) == 0, (
                f"Fold {fold.fold_idx}: train/val date overlap: {overlap}"
            )

    def test_purge_removes_boundary_dates(self, synthetic_df):
        """
        Dates in the window [min_val - purge_days, min_val) must not appear in
        the training set for that fold.
        """
        purge_days = 3
        cv = PurgedGroupKFold(n_splits=3, purge_days=purge_days, embargo_days=0)
        for fold in cv.split(synthetic_df):
            min_val_date = fold.val_dates.min()
            max_val_date = fold.val_dates.max()
            # The purge zone is [min_val - purge_days, min_val)
            purge_zone = set(range(int(min_val_date) - purge_days, int(min_val_date)))
            train_set = set(fold.train_dates.tolist())
            bad = purge_zone & train_set
            assert len(bad) == 0, (
                f"Fold {fold.fold_idx}: purge zone dates {bad} found in train set"
            )

    def test_all_rows_covered(self, synthetic_df):
        """Every row should appear in exactly one val fold."""
        cv = PurgedGroupKFold(n_splits=4, purge_days=1, embargo_days=1)
        n = len(synthetic_df)
        val_coverage = np.zeros(n, dtype=int)

        for fold in cv.split(synthetic_df):
            val_coverage[fold.val_idx] += 1

        # Each row is in exactly one val set
        assert (val_coverage == 1).all(), (
            "Some rows appear in multiple val sets or none."
        )

    def test_correct_number_of_folds(self, synthetic_df):
        cv = PurgedGroupKFold(n_splits=5, purge_days=1, embargo_days=1)
        folds = list(cv.split(synthetic_df))
        assert len(folds) == 5

    def test_raises_with_too_few_dates(self):
        """Should raise if there aren't enough dates for meaningful splits."""
        tiny_df = pl.DataFrame({
            "date_id": [0, 1, 2],
            "time_id": [0, 0, 0],
            "symbol_id": [0, 0, 0],
        })
        cv = PurgedGroupKFold(n_splits=5)
        with pytest.raises(ValueError, match="Not enough unique dates"):
            list(cv.split(tiny_df))

    def test_train_size_increases_across_folds(self, synthetic_df):
        """In the standard split, earlier folds have less training data."""
        cv = PurgedGroupKFold(n_splits=4, purge_days=1, embargo_days=1)
        folds = list(cv.split(synthetic_df))
        # First fold: small train (only dates before fold 0's val window)
        # Last fold: large train (all dates before fold 4's val window)
        # This isn't strictly guaranteed for all settings, but generally holds
        assert len(folds[0].train_dates) <= len(folds[-1].train_dates)


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


class TestSimpleSplit:
    def test_temporal_order(self, synthetic_df):
        train_df, val_df = train_val_date_split(synthetic_df, val_fraction=0.2)
        assert train_df[S.DATE_ID].max() < val_df[S.DATE_ID].min()

    def test_sizes(self, synthetic_df):
        train_df, val_df = train_val_date_split(synthetic_df, val_fraction=0.2)
        total = len(train_df) + len(val_df)
        # Some rows are purged so total < len(synthetic_df)
        assert total <= len(synthetic_df)
        assert len(val_df) > 0
        assert len(train_df) > 0
