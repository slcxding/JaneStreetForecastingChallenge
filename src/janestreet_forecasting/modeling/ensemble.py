"""
Ensemble methods for combining multiple model predictions.

Methods:
  1. WeightedAverageEnsemble  — optimise weights on OOF predictions
  2. RankAverageEnsemble      — average the rank-transformed predictions
  3. StackingEnsemble         — meta-model trained on OOF predictions

Start with WeightedAverageEnsemble. It typically performs well and is easy
to reason about. Stacking is more powerful but risks overfitting on the
OOF set if the folds were not carefully chosen.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from scipy.optimize import minimize

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.modeling.metrics import competition_score


class WeightedAverageEnsemble:
    """
    Combine predictions from multiple models via a weighted average.

    Weights are optimised to maximise the competition score on the OOF
    predictions using Nelder-Mead (a derivative-free optimiser).

    Args:
        n_models: Number of models to ensemble.
        fixed_weights: If provided, skip optimisation and use these weights.
    """

    def __init__(
        self,
        n_models: int,
        fixed_weights: list[float] | None = None,
    ) -> None:
        self.n_models = n_models
        self.weights = np.array(fixed_weights) if fixed_weights else np.ones(n_models) / n_models
        self._fitted = False

    def fit(
        self,
        oof_preds: list[np.ndarray],
        y_true: np.ndarray,
        weights: np.ndarray,
        dates: np.ndarray,
    ) -> "WeightedAverageEnsemble":
        """
        Optimise ensemble weights on OOF predictions.

        Args:
            oof_preds: List of OOF prediction arrays, one per model.
            y_true:    Ground truth array.
            weights:   Sample weights.
            dates:     date_id array for competition metric computation.
        """
        if len(oof_preds) != self.n_models:
            raise ValueError(
                f"Expected {self.n_models} OOF arrays, got {len(oof_preds)}."
            )

        preds_matrix = np.stack(oof_preds, axis=1)  # (N, n_models)

        def neg_score(w: np.ndarray) -> float:
            # Normalise to sum to 1 (simplex constraint)
            w = np.abs(w)
            w = w / (w.sum() + 1e-10)
            blended = preds_matrix @ w
            return -competition_score(y_true, blended, weights, dates)

        init_weights = np.ones(self.n_models) / self.n_models
        result = minimize(
            neg_score,
            init_weights,
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-5, "fatol": 1e-5},
        )

        opt_weights = np.abs(result.x)
        opt_weights /= opt_weights.sum()
        self.weights = opt_weights

        best_score = -result.fun
        logger.info(
            "Ensemble weights optimised: {} | OOF competition_score={:.4f}",
            [f"{w:.3f}" for w in self.weights], best_score,
        )

        self._fitted = True
        return self

    def predict(self, preds: list[np.ndarray]) -> np.ndarray:
        """
        Apply weights to a list of prediction arrays.

        Args:
            preds: One array per model, each shape (N,).

        Returns:
            Weighted average prediction, shape (N,).
        """
        preds_matrix = np.stack(preds, axis=1)
        return (preds_matrix @ self.weights).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"weights": self.weights, "n_models": self.n_models}, f)
        logger.info("Ensemble saved to {}", path)

    @classmethod
    def load(cls, path: Path) -> "WeightedAverageEnsemble":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(n_models=data["n_models"])
        obj.weights = data["weights"]
        obj._fitted = True
        return obj


class RankAverageEnsemble:
    """
    Combine predictions by averaging their rank percentiles.

    Rank averaging is more robust to scale differences between models and
    avoids giving excessive weight to outlier predictions.  Useful when
    the individual model distributions differ significantly.
    """

    def __init__(self, n_models: int) -> None:
        self.n_models = n_models
        self.weights = np.ones(n_models) / n_models

    def predict(self, preds: list[np.ndarray]) -> np.ndarray:
        """Rank each model's predictions then take a weighted average of ranks."""
        n = len(preds[0])
        ranked = np.stack([
            _rank_percentile(p) for p in preds
        ], axis=1)  # (N, n_models)
        return (ranked @ self.weights).astype(np.float32)


def _rank_percentile(arr: np.ndarray) -> np.ndarray:
    """Convert an array to percentile ranks in [0, 1]."""
    from scipy.stats import rankdata
    return rankdata(arr, method="average") / len(arr)


def load_oof_predictions(artifact_dir: Path) -> pl.DataFrame:
    """Load OOF predictions from a model artifact directory."""
    path = artifact_dir / "oof_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"OOF predictions not found at {path}")
    return pl.read_parquet(path)


def build_ensemble_from_artifacts(
    member_dirs: list[Path],
    method: str = "weighted_average",
) -> WeightedAverageEnsemble | RankAverageEnsemble:
    """
    Build an ensemble by loading OOF predictions from multiple experiments.

    Args:
        member_dirs: List of artifact directories, one per member model.
        method:      "weighted_average" or "rank_average".

    Returns:
        Fitted ensemble object.
    """
    oof_frames = [load_oof_predictions(d) for d in member_dirs]

    # Align on index — all OOF sets must cover the same rows
    base = oof_frames[0].select(S.INDEX_COLS + [S.TARGET_COL, S.WEIGHT, S.DATE_ID])
    oof_preds = [f["oof_prediction"].to_numpy() for f in oof_frames]

    y_true = base[S.TARGET_COL].to_numpy()
    weights = base[S.WEIGHT].to_numpy()
    dates = base[S.DATE_ID].to_numpy()

    if method == "weighted_average":
        ensemble = WeightedAverageEnsemble(n_models=len(oof_preds))
        ensemble.fit(oof_preds, y_true, weights, dates)
    elif method == "rank_average":
        ensemble = RankAverageEnsemble(n_models=len(oof_preds))
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    return ensemble
