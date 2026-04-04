"""
Kaggle API adapter.

Converts our PredictorPipeline into the format expected by the Kaggle
evaluation harness.  The competition uses a generator-based API where:

  1. You call `env = kaggle_evaluation.make_env()`
  2. You iterate: `for test, sample_prediction in env.iter_test()`
  3. For each batch, you call `env.predict(your_predictions_df)`

The prediction DataFrame must have:
  - `row_id` column matching the test batch
  - `responder_6` column with your predictions

This module provides `KagglePredictor` — a thin wrapper that connects our
`PredictorPipeline` to this interface.

Usage in a Kaggle notebook submission:
    ```python
    from janestreet_forecasting.inference.kaggle_adapter import KagglePredictor
    from janestreet_forecasting.inference.predict import PredictorPipeline

    predictor = PredictorPipeline.load_lgbm(Path("artifacts/exp001_lgbm_baseline"))
    kaggle_pred = KagglePredictor(predictor)
    kaggle_pred.run()
    ```

NOTE: This module imports `kaggle_evaluation` which is only available inside
the Kaggle notebook environment.  The class stubs are clearly marked so this
module can be imported and tested locally without the Kaggle library.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.inference.predict import PredictorPipeline


class KagglePredictor:
    """
    Wraps a PredictorPipeline for submission to the Kaggle API.

    Args:
        predictor:     Fitted PredictorPipeline.
        target_col:    Output column name (competition requires "responder_6").
    """

    def __init__(
        self,
        predictor: PredictorPipeline,
        target_col: str = S.TARGET_COL,
    ) -> None:
        self.predictor = predictor
        self.target_col = target_col

    def predict_batch(
        self,
        test_df: pd.DataFrame,
        lags_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Process one batch from the Kaggle API and return predictions.

        Args:
            test_df:  Pandas DataFrame from `env.iter_test()`.
            lags_df:  Optional lag DataFrame provided by the API (previous
                      time step's responders, already in test_df as lag_0 cols).

        Returns:
            Pandas DataFrame with columns [row_id, responder_6].
        """
        # Convert to Polars for our pipeline
        batch_pl = pl.from_pandas(test_df)

        # Generate predictions
        preds = self.predictor.predict(batch_pl)
        self.predictor.update_state(batch_pl)

        # Format output for Kaggle API
        out = pd.DataFrame({
            "row_id": test_df["row_id"].values,
            self.target_col: preds.astype(np.float64),
        })
        return out

    def run(self) -> None:
        """
        Execute the full competition prediction loop.

        This method requires the `kaggle_evaluation` library which is only
        available inside the Kaggle notebook environment.

        STUB: The actual submission code goes here. Uncomment the import and
        loop body when running inside Kaggle.
        """
        try:
            import kaggle_evaluation.jane_street_inference_server as js_server  # type: ignore[import]

            inference_server = js_server.JSInferenceServer(self.predict_batch)

            if inference_server.competition_api.is_competition_host:
                inference_server.serve()
            else:
                inference_server.run_local_gateway()

        except ImportError:
            # Running outside Kaggle — this is expected during local dev
            logger.warning(
                "kaggle_evaluation not available. "
                "This is expected when running locally. "
                "Use LocalInferenceSimulator for local testing."
            )

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: Path,
        model_type: str = "lgbm",
        **kwargs,
    ) -> "KagglePredictor":
        """
        Convenience constructor: load from artifact directory.

        Args:
            artifact_dir: Path to experiment artifact directory.
            model_type:   "lgbm" | "xgb" | "catboost".
        """
        loaders = {
            "lgbm": PredictorPipeline.load_lgbm,
            "xgb": PredictorPipeline.load_xgb,
            "catboost": PredictorPipeline.load_catboost,
        }
        if model_type not in loaders:
            raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(loaders)}")

        predictor = loaders[model_type](artifact_dir, **kwargs)
        return cls(predictor=predictor)
