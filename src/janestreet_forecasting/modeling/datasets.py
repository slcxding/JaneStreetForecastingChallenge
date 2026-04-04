"""
Dataset preparation utilities.

Converts Polars DataFrames into the format expected by each model library.
Keeping this logic here prevents it from leaking into train_*.py files.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl

from janestreet_forecasting.data import schemas as S


def to_numpy_arrays(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = S.TARGET_COL,
    weight_col: str = S.WEIGHT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (X, y, weights) NumPy arrays from a Polars DataFrame.

    Args:
        df:           Source DataFrame.
        feature_cols: Ordered list of feature column names.
        target_col:   Target column name.
        weight_col:   Weight column name.

    Returns:
        (X, y, weights) as float32 NumPy arrays.
    """
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Feature columns not found in DataFrame: {missing}")

    X = df.select(feature_cols).to_numpy(allow_copy=True).astype(np.float32)
    y = df[target_col].to_numpy().astype(np.float32)
    w = df[weight_col].to_numpy().astype(np.float32)

    return X, y, w


def make_lgbm_dataset(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = S.TARGET_COL,
    weight_col: str = S.WEIGHT,
    reference=None,
):
    """
    Create a LightGBM Dataset from a Polars DataFrame.

    Args:
        reference: Another Dataset to use as reference (for val sets).
    """
    import lightgbm as lgb

    X, y, w = to_numpy_arrays(df, feature_cols, target_col, weight_col)
    return lgb.Dataset(
        data=X,
        label=y,
        weight=w,
        feature_name=list(feature_cols),
        reference=reference,
        free_raw_data=False,
    )


def make_xgb_dmatrix(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = S.TARGET_COL,
    weight_col: str = S.WEIGHT,
):
    """Create an XGBoost DMatrix from a Polars DataFrame."""
    import xgboost as xgb

    X, y, w = to_numpy_arrays(df, feature_cols, target_col, weight_col)
    return xgb.DMatrix(data=X, label=y, weight=w, feature_names=list(feature_cols))


def make_catboost_pool(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = S.TARGET_COL,
    weight_col: str = S.WEIGHT,
):
    """Create a CatBoost Pool from a Polars DataFrame."""
    from catboost import Pool

    X, y, w = to_numpy_arrays(df, feature_cols, target_col, weight_col)
    return Pool(data=X, label=y, weight=w, feature_names=list(feature_cols))
