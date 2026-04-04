"""
Base Transformer protocol and abstract class.

We define the feature interface using two approaches:

1. `TransformerProtocol` — a structural type used for static type checking.
   Any class with `fit` and `transform` matching these signatures satisfies it
   without explicitly inheriting.  This keeps feature code clean and doesn't
   force a dependency on sklearn.

2. `BaseTransformer` — an optional abstract base class with shared helpers
   (e.g., is_fitted check, output column tracking).  Use this when you want
   inheritance-based organisation.

Design note: we deliberately do NOT inherit from sklearn's TransformerMixin.
The sklearn API adds bloat (fit_transform, set_params, get_params) that is
largely irrelevant for our use case.  We add a `to_sklearn_transformer()`
adapter when sklearn compatibility is needed (e.g., in a Pipeline).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class TransformerProtocol(Protocol):
    """Structural type for any feature transformer."""

    def fit(self, df: pl.DataFrame, **kwargs: Any) -> "TransformerProtocol":
        """Learn parameters from training data. Return self."""
        ...

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply learned transformation. Must not modify df in-place."""
        ...

    @property
    def feature_names_out(self) -> list[str]:
        """Names of columns added by transform()."""
        ...


class BaseTransformer(ABC):
    """
    Abstract base class for feature transformers.

    Subclasses must implement:
      - _fit(df) — learn any statistics needed from training data
      - _transform(df) — apply the transformation
      - feature_names_out (property) — list of output column names
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._output_cols: list[str] = []

    def fit(self, df: pl.DataFrame, **kwargs: Any) -> "BaseTransformer":
        """Fit on training data and return self."""
        self._fit(df, **kwargs)
        self._fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply transformation. Raises if not yet fitted.

        Returns a new DataFrame with added columns (original columns preserved).
        Does NOT modify df in-place.
        """
        self._assert_fitted()
        return self._transform(df)

    def fit_transform(self, df: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
        """Convenience: fit then transform on the same data."""
        return self.fit(df, **kwargs).transform(df)

    @property
    def feature_names_out(self) -> list[str]:
        return self._output_cols

    @abstractmethod
    def _fit(self, df: pl.DataFrame, **kwargs: Any) -> None: ...

    @abstractmethod
    def _transform(self, df: pl.DataFrame) -> pl.DataFrame: ...

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before calling transform(). "
                "Call fit() on training data first."
            )
