"""
Custom training objectives for LightGBM and XGBoost.

The competition metric is not directly differentiable in a convenient form,
so we use proxy losses during training and evaluate the true metric separately.

Provided objectives:
  - Huber loss: robust to outliers (financial returns can be fat-tailed)
  - Weighted MSE: standard regression with sample weights
  - Pearson correlation loss: a differentiable proxy for the competition metric

Note: LightGBM and XGBoost use different function signatures for custom
objectives. We provide both versions.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# LightGBM objective interface: (y_pred, dataset) -> (grad, hess)
# ---------------------------------------------------------------------------


def huber_objective_lgbm(
    y_pred: np.ndarray, dataset, delta: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """
    Huber loss for LightGBM.

    Behaves like MSE for small errors (|e| < delta) and like MAE for large
    errors.  More robust to outliers than pure MSE.

    Args:
        y_pred:  Current predictions.
        dataset: LightGBM Dataset (provides labels and weights).
        delta:   Huber transition threshold.  Tune based on target scale.
    """
    y_true = dataset.get_label()
    weights = dataset.get_weight()
    if weights is None:
        weights = np.ones_like(y_true)

    residual = y_pred - y_true
    abs_res = np.abs(residual)

    # Gradient and Hessian of Huber loss
    grad = np.where(abs_res <= delta, residual, delta * np.sign(residual))
    hess = np.where(abs_res <= delta, 1.0, delta / (abs_res + 1e-8))

    return weights * grad, weights * hess


def pearson_correlation_objective_lgbm(
    y_pred: np.ndarray, dataset
) -> tuple[np.ndarray, np.ndarray]:
    """
    Differentiable proxy for the competition's correlation metric.

    We maximise weighted Pearson correlation, which requires computing
    gradients through the correlation formula.

    Use with care: this objective is more complex to tune than MSE and can
    be numerically unstable.  Start with MSE and switch if it helps.
    """
    y_true = dataset.get_label()
    weights = dataset.get_weight()
    if weights is None:
        weights = np.ones_like(y_true)

    n = len(y_true)
    w_sum = weights.sum()

    # Weighted means
    y_mean = np.sum(weights * y_true) / w_sum
    p_mean = np.sum(weights * y_pred) / w_sum

    y_centered = y_true - y_mean
    p_centered = y_pred - p_mean

    # Covariance terms
    cov = np.sum(weights * y_centered * p_centered)
    var_y = np.sum(weights * y_centered ** 2)
    var_p = np.sum(weights * p_centered ** 2)

    denom = np.sqrt(var_y * var_p) + 1e-8

    # Gradient of correlation w.r.t. y_pred
    # d(corr)/d(p_i) = w_i * [y_centered_i * denom - cov * var_y * p_centered_i / (var_p * denom)] / denom^2
    grad = weights * (
        y_centered / denom
        - cov * p_centered / (var_p * denom + 1e-8)
    )
    # Negate because LightGBM minimises, but we want to maximise correlation
    grad = -grad

    # Hessian approximation (constant diagonal for stability)
    hess = weights * np.ones(n)

    return grad, hess


# ---------------------------------------------------------------------------
# XGBoost objective interface: (y_pred, dmatrix) -> (grad, hess)
# XGBoost uses the same signature as LightGBM for custom objectives
# ---------------------------------------------------------------------------


def huber_objective_xgb(
    y_pred: np.ndarray, dmatrix, delta: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Huber loss for XGBoost. Same logic as LightGBM version."""
    y_true = dmatrix.get_label()
    residual = y_pred - y_true
    abs_res = np.abs(residual)
    grad = np.where(abs_res <= delta, residual, delta * np.sign(residual))
    hess = np.where(abs_res <= delta, 1.0, delta / (abs_res + 1e-8))
    return grad, hess
