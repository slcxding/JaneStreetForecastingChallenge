"""
Prediction calibration.

Financial regression targets are typically well-calibrated by GBDTs, but
calibration can help if the model predictions have systematic bias.

Provided calibrators:
  - TemporalBiasCorrector: removes systematic trend bias across time
  - IsotonicCalibrator: sklearn's isotonic regression for rank preservation
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """
    Wrap sklearn's IsotonicRegression for post-hoc calibration.

    Isotonic regression fits a monotone non-decreasing function to the OOF
    predictions. It preserves rank order (important for correlation metrics)
    while correcting systematic bias.

    Use with care: isotonic regression can overfit on small validation sets.
    Only apply if you see a clear calibration gap in diagnostics.
    """

    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self._fitted = False

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self._model.fit(y_pred, y_true)
        self._fitted = True
        logger.debug("IsotonicCalibrator fitted on {} samples", len(y_pred))
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self._model.predict(y_pred).astype(np.float32)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    @classmethod
    def load(cls, path: Path) -> "IsotonicCalibrator":
        obj = cls()
        with open(path, "rb") as f:
            obj._model = pickle.load(f)
        obj._fitted = True
        return obj


class TemporalBiasCorrector:
    """
    Remove systematic time-varying bias from predictions.

    Computes the mean residual (y_true - y_pred) per date_id in the OOF set,
    smooths it with a rolling average, and subtracts from future predictions.

    This addresses regime shifts where the model systematically over- or
    under-predicts during certain market conditions.
    """

    def __init__(self, smooth_window: int = 20) -> None:
        self.smooth_window = smooth_window
        self._bias_by_date: dict[int, float] = {}
        self._global_bias: float = 0.0
        self._fitted = False

    def fit(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        dates: np.ndarray,
    ) -> "TemporalBiasCorrector":
        residuals = y_true - y_pred
        unique_dates = np.unique(dates)
        raw_bias = {
            int(d): float(residuals[dates == d].mean()) for d in unique_dates
        }

        # Smooth the bias estimates
        sorted_dates = sorted(raw_bias.keys())
        raw_values = np.array([raw_bias[d] for d in sorted_dates], dtype=np.float64)

        if len(raw_values) >= self.smooth_window:
            from numpy.lib.stride_tricks import sliding_window_view
            padded = np.pad(raw_values, (self.smooth_window - 1, 0), mode="edge")
            smoothed = sliding_window_view(padded, self.smooth_window).mean(axis=-1)
        else:
            smoothed = raw_values

        self._bias_by_date = dict(zip(sorted_dates, smoothed.tolist()))
        self._global_bias = float(residuals.mean())
        self._fitted = True
        logger.debug(
            "TemporalBiasCorrector fitted: mean_bias={:.6f}", self._global_bias
        )
        return self

    def transform(self, y_pred: np.ndarray, dates: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        biases = np.array([
            self._bias_by_date.get(int(d), self._global_bias) for d in dates
        ])
        return (y_pred + biases).astype(np.float32)
