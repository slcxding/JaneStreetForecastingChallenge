"""
Time-aware cross-validation splits for financial time-series.

Standard k-fold CV is invalid for financial data because:
  1. Autocorrelation: adjacent rows share information (the market at t is
     correlated with t-1). Random splitting creates leakage between folds.
  2. Rolling-feature overlap: a 10-period rolling mean at time t uses data
     from t-9 through t. If t-5 is in validation, its rolling feature computed
     from training data already contains validation-period information.

We implement:

  PurgedGroupKFold (forward_only=True — default)
  ─────────────────────────────────────────────
  Test folds are ordered chronologically. Each fold trains ONLY on dates that
  come BEFORE the test fold (plus purge/embargo exclusions).

  Illustration (4 folds, purge=2, embargo=1):

  Date:  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15
  Fold0: TS TS TS TS              (no train — not enough history)
  Fold1: TR TR TR TR TR [P]  TS  TS  TS  TS [E]
  Fold2: TR TR TR TR TR TR TR TR [P]  TS  TS  TS  TS [E]
  Fold3: TR TR .... [P]  TS  TS  TS  TS [E]

  NOTE: fold 0 is SKIPPED when forward_only=True and there is insufficient
  training history. The `min_train_dates` parameter controls the threshold.

  forward_only=False: uses all non-excluded dates as training regardless of
  whether they come before or after the test fold. This is the original
  López de Prado formulation and gives more training data per fold, but
  allows the model to see future market regimes when predicting past ones.
  Appropriate ONLY for cross-validation as a hyperparameter selection proxy,
  NOT for temporal realism.

  WalkForwardSplit
  ─────────────────
  Expanding-window walk-forward split. Each fold adds one step of training
  data. The most temporally honest split — use for final model diagnostics.

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
        n_splits:        Number of folds.
        purge_days:      Dates immediately before each test fold to remove from
                         training.  Set to >= your largest rolling window so the
                         boundary features don't bleed across the fold edge.
        embargo_days:    Dates immediately after each test fold to exclude from
                         training of that fold.
        forward_only:    If True (default), each fold trains ONLY on dates
                         strictly before the test window.  This is temporally
                         honest.  If False, all non-excluded dates are used as
                         training (the original LP formulation — more training
                         data but allows future regimes to inform past predictions).
        min_train_dates: Minimum training dates required for a fold to be
                         yielded.  Folds with less history are skipped.
                         Only relevant when forward_only=True.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 5,
        embargo_days: int = 10,
        forward_only: bool = True,
        min_train_dates: int = 30,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be ≥ 2.")
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.forward_only = forward_only
        self.min_train_dates = min_train_dates

    def split(
        self,
        df: pl.DataFrame,
        date_col: str = S.DATE_ID,
    ) -> Generator[FoldIndices, None, None]:
        """
        Yield FoldIndices for each split.

        When forward_only=True, test folds are ordered chronologically and
        each fold trains only on data that precedes it.  Folds with
        insufficient training history (< min_train_dates) are silently skipped
        — callers should assert they received at least one fold.

        Args:
            df:       Full DataFrame.  Does NOT need to be pre-sorted.
            date_col: Column used to group rows into dates.

        Yields:
            FoldIndices with train/val row positions and dates.
        """
        dates = df[date_col].to_numpy()
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)

        if n_dates < self.n_splits * 2:
            raise ValueError(
                f"Not enough unique dates ({n_dates}) for {self.n_splits} folds "
                "with meaningful train/test splits."
            )

        # Divide sorted unique dates into n_splits consecutive chunks.
        # Chunk 0 = earliest dates (fold 0 test set), etc.
        date_chunks = np.array_split(unique_dates, self.n_splits)

        yielded = 0
        for fold_idx, test_dates in enumerate(date_chunks):
            test_date_min = int(test_dates.min())
            test_date_max = int(test_dates.max())

            # Purge: dates immediately before the test window
            purge_mask = (unique_dates >= test_date_min - self.purge_days) & (
                unique_dates < test_date_min
            )
            purge_date_set = set(unique_dates[purge_mask].tolist())

            # Embargo: dates immediately after the test window
            embargo_mask = (unique_dates > test_date_max) & (
                unique_dates <= test_date_max + self.embargo_days
            )
            embargo_date_set = set(unique_dates[embargo_mask].tolist())

            test_date_set = set(test_dates.tolist())
            excluded = test_date_set | purge_date_set | embargo_date_set

            if self.forward_only:
                # Training is restricted to dates strictly BEFORE the test window.
                # The purge removes boundary dates; the embargo is irrelevant here
                # (no dates after the test fold enter training anyway).
                train_date_candidates = unique_dates[unique_dates < test_date_min]
                train_date_candidates = train_date_candidates[
                    ~np.isin(train_date_candidates, list(excluded))
                ]
            else:
                # Use all non-excluded dates regardless of temporal position.
                train_date_candidates = unique_dates[
                    ~np.isin(unique_dates, list(excluded))
                ]

            if len(train_date_candidates) < self.min_train_dates:
                logger.debug(
                    "Fold {} skipped: only {} training dates (need {})",
                    fold_idx, len(train_date_candidates), self.min_train_dates,
                )
                continue

            train_mask = np.isin(dates, train_date_candidates)
            val_mask = np.isin(dates, list(test_date_set))

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            train_dates_out = np.sort(np.unique(dates[train_mask]))

            logger.debug(
                "Fold {}/{}: train={} dates, val={} dates | purged={} embargoed={}",
                fold_idx + 1, self.n_splits,
                len(train_dates_out), len(test_dates),
                len(purge_date_set), len(embargo_date_set),
            )

            yielded += 1
            yield FoldIndices(
                fold_idx=fold_idx,
                train_idx=train_idx,
                val_idx=val_idx,
                train_dates=train_dates_out,
                val_dates=test_dates,
            )

        if yielded == 0:
            raise ValueError(
                f"PurgedGroupKFold yielded 0 folds. "
                f"Increase n_dates, reduce min_train_dates ({self.min_train_dates}), "
                f"or reduce purge_days/embargo_days."
            )

    def get_n_splits(self) -> int:
        return self.n_splits


class WalkForwardSplit:
    """
    Expanding-window walk-forward split.

    Each fold uses all dates up to a cutoff as training, and the next
    `test_window` dates as validation.  The training window grows by
    `step_days` each fold.

    This is the most temporally honest CV strategy.  Use it for:
      - Final model diagnostics before deployment
      - Detecting model degradation over time
      - Simulating production deployment
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
        unique_dates = np.sort(np.unique(dates))

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
    `purge_days` dates before the split point are removed from training.

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
