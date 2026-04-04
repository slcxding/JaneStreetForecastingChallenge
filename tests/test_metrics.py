"""
Tests for the competition metric implementation.

The competition metric is a sum of weighted Pearson correlations per date.
We test:
  1. Perfect predictions score = number of unique dates
  2. Zero predictions score = 0
  3. Sign-flipped predictions score = negative (sum of -1 per date)
  4. Single date
  5. Numerical stability (zero variance target)
"""

import numpy as np
import pytest

from janestreet_forecasting.modeling.metrics import (
    competition_score,
    mean_competition_score,
    weighted_r2,
    weighted_rmse,
    compute_all_metrics,
)


def _make_inputs(n_dates=5, n_per_date=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(np.arange(n_dates), n_per_date)
    y_true = rng.standard_normal(n_dates * n_per_date).astype(np.float32)
    weights = np.ones(n_dates * n_per_date, dtype=np.float32)
    return y_true, weights, dates


class TestCompetitionScore:
    def test_perfect_predictions_score_equals_n_dates(self):
        """Perfect predictions: each per-date term = 1, total = n_dates."""
        n_dates = 5
        y_true, weights, dates = _make_inputs(n_dates=n_dates)
        score = competition_score(y_true, y_true, weights, dates)
        assert abs(score - n_dates) < 1e-5, f"Expected {n_dates}, got {score}"

    def test_zero_predictions_score_is_zero(self):
        """Zero predictions: numerator = 0, so score = 0."""
        y_true, weights, dates = _make_inputs(n_dates=5)
        y_pred = np.zeros_like(y_true)
        score = competition_score(y_true, y_pred, weights, dates)
        assert abs(score) < 1e-5

    def test_sign_flipped_predictions(self):
        """Negating predictions: correlation per date = -1, total = -n_dates."""
        n_dates = 4
        y_true, weights, dates = _make_inputs(n_dates=n_dates)
        # Avoid near-zero targets which could give undefined correlation
        y_true = y_true + 1.0
        y_pred = -y_true
        score = competition_score(y_true, y_pred, weights, dates)
        assert score < 0, "Negated predictions should give negative score"
        assert abs(score + n_dates) < 1e-4

    def test_single_date(self):
        n = 20
        y_true = np.random.randn(n).astype(np.float32)
        y_pred = y_true * 0.5 + np.random.randn(n).astype(np.float32) * 0.1
        weights = np.ones(n, dtype=np.float32)
        dates = np.zeros(n, dtype=np.int32)
        score = competition_score(y_true, y_pred, weights, dates)
        assert -1.0 <= score <= 1.0, "Single-date score should be in [-1, 1]"

    def test_degenerate_zero_variance_target(self):
        """When all y_true are constant, the score should be 0 (not NaN/inf)."""
        n = 10
        y_true = np.ones(n, dtype=np.float32)
        y_pred = np.random.randn(n).astype(np.float32)
        weights = np.ones(n, dtype=np.float32)
        dates = np.zeros(n, dtype=np.int32)
        score = competition_score(y_true, y_pred, weights, dates)
        assert np.isfinite(score)

    def test_mean_score_normalises_by_n_dates(self):
        n_dates = 6
        y_true, weights, dates = _make_inputs(n_dates=n_dates)
        total = competition_score(y_true, y_true, weights, dates)
        mean = mean_competition_score(y_true, y_true, weights, dates)
        assert abs(mean - total / n_dates) < 1e-5


class TestWeightedR2:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        assert abs(weighted_r2(y, y, w) - 1.0) < 1e-5

    def test_mean_prediction(self):
        """Predicting the (weighted) mean gives R² = 0."""
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        y_mean = np.full_like(y, y.mean())
        assert abs(weighted_r2(y, y_mean, w)) < 1e-5

    def test_negative_r2(self):
        """Predictions worse than the mean give R² < 0."""
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        y_bad = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        assert weighted_r2(y, y_bad, w) < 0


class TestWeightedRMSE:
    def test_zero_error(self):
        y = np.ones(5, dtype=np.float32)
        w = np.ones(5, dtype=np.float32)
        assert abs(weighted_rmse(y, y, w)) < 1e-6

    def test_known_value(self):
        y_true = np.array([0.0, 0.0], dtype=np.float32)
        y_pred = np.array([1.0, 1.0], dtype=np.float32)
        w = np.ones(2, dtype=np.float32)
        assert abs(weighted_rmse(y_true, y_pred, w) - 1.0) < 1e-5


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        y_true, weights, dates = _make_inputs()
        y_pred = y_true * 0.8
        metrics = compute_all_metrics(y_true, y_pred, weights, dates)
        expected_keys = {
            "competition_score", "mean_competition_score",
            "weighted_r2", "weighted_rmse", "weighted_mae",
        }
        assert expected_keys == set(metrics.keys())

    def test_all_values_finite(self):
        y_true, weights, dates = _make_inputs()
        y_pred = np.random.randn(len(y_true)).astype(np.float32)
        metrics = compute_all_metrics(y_true, y_pred, weights, dates)
        for k, v in metrics.items():
            assert np.isfinite(v), f"Metric {k} is not finite: {v}"
