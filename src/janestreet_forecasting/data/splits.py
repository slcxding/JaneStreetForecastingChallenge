"""
Time-aware cross-validation splits for financial time-series.

Standard k-fold CV is invalid for financial data because:
  1. Train and test sets can overlap in time — autocorrelated errors leak
     through, making OOF scores over-optimistic.
  2. Even non-overlapping adjacent folds suffer from autocorrelation between
     the last training rows and first test rows.

We implement:

  PurgedGroupKFold
  ────────────────
  Groups by `date_id` so that all rows from the same date are in the same
  fold.  Before training, a configurable number of dates at the fold boundary
  are *purged* from the train set.  After the test fold, an *embargo* of
  additional dates is excluded from any future training fold.

  Illustration (5 folds, purge=2, embargo=1):

  Date:  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15
  Fold1: TR TR TR TR TR TR TR TR TR [P] [P]  TS  TS  TS  [E]
  Fold2: TR TR TR TR TR TR [P] [P] TS  TS   TS  [E] ...

  [P] = purged (excluded from train)
  [E] = embargo (excluded from train in that fold)
  TS  = test set

  WalkForwardSplit
  ─────────────────
  A simpler expanding-window split that adds one fold's worth of data to the
  training set each iteration.  Useful for final model training diagnostics.

Reference: "Advances in Financial Machine Learning" (López de Prado, 2018),
Chapter 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np
import polars as pl
from loguru import logger

from janestreet_forecasting.data import schemas as S


@dataclass
class FoldIndices:
    """Integer row indices for a single CV fold."""

    fold_idx: int
    train_idx: np.ndarray      # row positions in the full DataFrame
    val_idx: np.ndarray
    train_dates: np.ndarray    # unique date_ids in train
    val_dates: np.ndarray      # unique date_ids in val


class PurgedGroupKFold:
    """
    K-Fold cross-validator that groups by date and applies purge + embargo.

    Args:
        n_splits:    Number of folds.
        purge_days:  Number of dates immediately before the test fold to remove
                     from training.  These dates likely share information with
                     the test period (e.g. via rolling features with a window
                     that spans the boundary).
        embargo_days: Number of dates immediately after the test fold to
                     exclude from *subsequent* training folds. Prevents leakage
                     from information that becomes available after prediction.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 5,
        embargo_days: int = 10,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be ≥ 2.")
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(
        self,
        df: pl.DataFrame,
        date_col: str = S.DATE_ID,
    ) -> Generator[FoldIndices, None, None]:
        """
        Yield FoldIndices for each split.

        Args:
            df:       Full DataFrame sorted by date_col.
            date_col: Column used to group rows into dates.

        Yields:
            FoldIndices with train/val row positions and dates.
        """
        dates = df[date_col].to_numpy()
        unique_dates = np.unique(dates)
        n_dates = len(unique_dates)

        if n_dates < self.n_splits * 2:
            raise ValueError(
                f"Not enough unique dates ({n_dates}) for {self.n_splits} folds "
                "with meaningful train/test splits."
            )

        # Assign each unique date to a fold
        date_fold = np.array_split(unique_dates, self.n_splits)

        for fold_idx, test_dates in enumerate(date_fold):
            test_date_set = set(test_dates.tolist())
            test_date_min = test_dates.min()
            test_date_max = test_dates.max()

            # Purge: exclude dates just before the test window
            purge_date_set = set(
                unique_dates[
                    (unique_dates >= test_date_min - self.purge_days)
                    & (unique_dates < test_date_min)
                ].tolist()
            )

            # Embargo: exclude dates just after the test window
            embargo_date_set = set(
                unique_dates[
                    (unique_dates > test_date_max)
                    & (unique_dates <= test_date_max + self.embargo_days)
                ].tolist()
            )

            excluded = test_date_set | purge_date_set | embargo_date_set

            train_mask = ~np.isin(dates, list(excluded))
            val_mask = np.isin(dates, list(test_date_set))

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            train_dates = np.unique(dates[train_mask])
            val_dates = test_dates

            logger.debug(
                "Fold {}/{}: train_dates={} val_dates={} purged={} embargoed={}",
                fold_idx + 1, self.n_splits,
                len(train_dates), len(val_dates),
                len(purge_date_set), len(embargo_date_set),
            )

            yield FoldIndices(
                fold_idx=fold_idx,
                train_idx=train_idx,
                val_idx=val_idx,
                train_dates=train_dates,
                val_dates=val_dates,
            )

    def get_n_splits(self) -> int:
        return self.n_splits


class WalkForwardSplit:
    """
    Expanding-window walk-forward split.

    Each fold uses all dates up to a cutoff as training, and the next
    `test_window` dates as validation.  The training window grows by
    `step_days` each fold.

    Useful for:
      - Simulating a live deployment scenario
      - Evaluating model degradation over time
      - Final production model diagnostics
    """

    def __init__(
        self,
        min_train_days: int = 200,
        test_window: int = 50,
        step_days: int = 50,
        embargo_days: int = 5,
    ) -> None:
        self.min_train_days = min_train_days
        self.test_window = test_window
        self.step_days = step_days
        self.embargo_days = embargo_days

    def split(
        self,
        df: pl.DataFrame,
        date_col: str = S.DATE_ID,
    ) -> Generator[FoldIndices, None, None]:
        dates = df[date_col].to_numpy()
        unique_dates = np.unique(dates)

        start = self.min_train_days
        fold_idx = 0

        while start + self.test_window <= len(unique_dates):
            train_dates = unique_dates[: start - self.embargo_days]
            val_dates = unique_dates[start: start + self.test_window]

            train_mask = np.isin(dates, train_dates)
            val_mask = np.isin(dates, val_dates)

            yield FoldIndices(
                fold_idx=fold_idx,
                train_idx=np.where(train_mask)[0],
                val_idx=np.where(val_mask)[0],
                train_dates=train_dates,
                val_dates=val_dates,
            )

            start += self.step_days
            fold_idx += 1


def train_val_date_split(
    df: pl.DataFrame,
    val_fraction: float = 0.2,
    purge_days: int = 5,
    date_col: str = S.DATE_ID,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Simple chronological train/val split for quick experiments.

    The last `val_fraction` of unique dates are held out for validation.
    `purge_days` dates before the split are removed from training.

    This is NOT for final model selection — use PurgedGroupKFold for that.
    """
    unique_dates = df[date_col].unique().sort()
    n_val = max(1, int(len(unique_dates) * val_fraction))
    n_train = len(unique_dates) - n_val

    if n_train <= purge_days:
        raise ValueError(
            f"After purging {purge_days} days, no training dates remain."
        )

    train_cutoff = unique_dates[n_train - purge_days - 1]
    val_start = unique_dates[n_train]

    train_df = df.filter(pl.col(date_col) <= train_cutoff)
    val_df = df.filter(pl.col(date_col) >= val_start)

    logger.info(
        "Train/val split: train_dates={}..{} val_dates={}..{}",
        unique_dates[0], train_cutoff,
        val_start, unique_dates[-1],
    )
    return train_df, val_df
