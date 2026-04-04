"""
Data quality validation.

Run after loading raw data and before any feature engineering.
Returns a structured report so callers can decide whether to fail hard or log
a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
from loguru import logger

from janestreet_forecasting.data import schemas as S


@dataclass
class ValidationReport:
    """Results of a data quality check run."""

    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False
        logger.error(msg)

    def raise_if_failed(self) -> None:
        """Raise ValueError with all errors if any checks failed."""
        if not self.passed:
            raise ValueError("Data validation failed:\n" + "\n".join(self.errors))

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Validation {status} — {len(self.warnings)} warnings, {len(self.errors)} errors"]
        for w in self.warnings:
            lines.append(f"  WARN  {w}")
        for e in self.errors:
            lines.append(f"  ERROR {e}")
        return "\n".join(lines)


def validate_dataframe(df: pl.DataFrame, strict: bool = True) -> ValidationReport:
    """
    Run all data quality checks on a materialised training DataFrame.

    Args:
        df:     The DataFrame to validate.
        strict: If True, errors raise immediately. If False, collect all issues.

    Returns:
        ValidationReport with all findings.
    """
    report = ValidationReport()

    _check_required_columns(df, report)
    _check_temporal_monotonicity(df, report)
    _check_missing_rates(df, report)
    _check_weight_validity(df, report)
    _check_target_distribution(df, report)

    logger.info(report.summary())

    if strict:
        report.raise_if_failed()

    return report


def validate_no_future_leakage(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    date_col: str = S.DATE_ID,
) -> None:
    """
    Assert that the maximum training date is strictly less than the minimum
    validation date (after accounting for purge/embargo).

    This is a hard check — we always raise on failure.
    """
    max_train = train_df[date_col].max()
    min_val = val_df[date_col].min()

    if max_train is None or min_val is None:
        raise ValueError("Empty train or validation split.")

    if max_train >= min_val:
        raise ValueError(
            f"Look-ahead leakage detected! "
            f"Max train {date_col}={max_train} >= min val {date_col}={min_val}. "
            "This means training data contains dates from the validation period."
        )
    logger.debug(
        "Leakage check passed: max_train_date={} < min_val_date={}",
        max_train, min_val,
    )


# ---------------------------------------------------------------------------
# Private check functions
# ---------------------------------------------------------------------------


def _check_required_columns(df: pl.DataFrame, report: ValidationReport) -> None:
    required = set(S.INDEX_COLS + [S.WEIGHT, S.TARGET_COL])
    missing = required - set(df.columns)
    if missing:
        report.add_error(f"Missing required columns: {sorted(missing)}")
    else:
        logger.debug("Required columns: OK")


def _check_temporal_monotonicity(df: pl.DataFrame, report: ValidationReport) -> None:
    """date_id should be non-decreasing in the raw file."""
    if S.DATE_ID not in df.columns:
        return
    dates = df[S.DATE_ID].to_numpy()
    if (dates[1:] < dates[:-1]).any():
        report.add_error(
            "date_id is not monotonically non-decreasing. "
            "Ensure training data is sorted by date_id before use."
        )
    else:
        logger.debug("Temporal monotonicity: OK")


def _check_missing_rates(
    df: pl.DataFrame,
    report: ValidationReport,
    warn_threshold: float = 0.30,
    error_threshold: float = 0.80,
) -> None:
    """Warn on high missingness; error on near-complete columns."""
    n = len(df)
    if n == 0:
        report.add_error("DataFrame is empty.")
        return

    for col in S.FEATURE_COLS:
        if col not in df.columns:
            continue
        null_rate = df[col].null_count() / n
        if null_rate >= error_threshold:
            report.add_error(
                f"{col} has {null_rate:.1%} missing values (threshold={error_threshold:.0%})."
            )
        elif null_rate >= warn_threshold:
            report.add_warning(
                f"{col} has {null_rate:.1%} missing values (threshold={warn_threshold:.0%})."
            )


def _check_weight_validity(df: pl.DataFrame, report: ValidationReport) -> None:
    """Weights must be non-negative and not all zero."""
    if S.WEIGHT not in df.columns:
        return
    weights = df[S.WEIGHT]
    if (weights < 0).sum() > 0:
        report.add_error("Negative weights found.")
    if weights.sum() == 0:
        report.add_error("All weights are zero — metric will be undefined.")
    elif weights.null_count() > 0:
        report.add_warning(f"{weights.null_count()} null weights found.")


def _check_target_distribution(df: pl.DataFrame, report: ValidationReport) -> None:
    """Target should be roughly zero-centred with finite variance."""
    if S.TARGET_COL not in df.columns:
        return
    target = df[S.TARGET_COL].drop_nulls()
    if len(target) == 0:
        report.add_error("Target column is entirely null.")
        return
    mean = target.mean()
    std = target.std()
    if std == 0 or std is None:
        report.add_error("Target has zero variance — something is wrong.")
    elif abs(mean) > 1.0:  # type: ignore[operator]
        report.add_warning(
            f"Target mean={mean:.4f} is far from zero. "
            "Expected ~0 for financial return targets."
        )
