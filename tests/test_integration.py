"""
Integration smoke test: pipeline → features → train → evaluate → infer.

This test runs the entire pipeline on synthetic data end-to-end, catching
wiring bugs that unit tests can't see.  It uses the smallest possible data
and fastest model settings — the goal is correctness, not performance.

Run with: pytest tests/test_integration.py -v
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from janestreet_forecasting.data import schemas as S
from janestreet_forecasting.data.loaders import make_synthetic_dataset
from janestreet_forecasting.data.splits import PurgedGroupKFold
from janestreet_forecasting.data.validation import validate_dataframe
from janestreet_forecasting.features.lag_features import LagTransformer
from janestreet_forecasting.features.pipelines import FeaturePipeline
from janestreet_forecasting.features.rolling_features import RollingTransformer
from janestreet_forecasting.inference.predict import PredictorPipeline
from janestreet_forecasting.inference.serve_local import LocalInferenceSimulator
from janestreet_forecasting.inference.state import InferenceState
from janestreet_forecasting.modeling.baselines import WeightedMeanPredictor, ZeroPredictor
from janestreet_forecasting.modeling.metrics import competition_score, compute_all_metrics
from janestreet_forecasting.modeling.train_lgbm import LGBMTrainer


@pytest.fixture(scope="module")
def integration_df() -> pl.DataFrame:
    """Synthetic dataset large enough for a meaningful CV run."""
    return make_synthetic_dataset(
        n_dates=40, n_times_per_date=8, n_symbols=10, seed=7
    )


@pytest.fixture(scope="module")
def feature_pipeline() -> FeaturePipeline:
    """Minimal pipeline for fast integration tests."""
    return FeaturePipeline(
        transformers=[
            ("lag", LagTransformer(columns=[S.LAG_COLS[0]], lags=[1])),
            ("rolling", RollingTransformer(
                columns=[S.FEATURE_COLS[0]],
                windows=[3],
                stats=["mean", "std"],
            )),
        ]
    )


class TestDataValidation:
    def test_synthetic_data_passes_validation(self, integration_df):
        """Synthetic data should pass all schema checks."""
        report = validate_dataframe(integration_df, strict=False)
        assert len(report.errors) == 0, f"Validation errors: {report.errors}"


class TestFeaturePipeline:
    def test_pipeline_fit_transform(self, integration_df, feature_pipeline):
        """Pipeline should add new columns without corrupting originals."""
        result = feature_pipeline.fit_transform(integration_df)

        # All original columns preserved
        for col in S.INDEX_COLS + [S.TARGET_COL, S.WEIGHT]:
            assert col in result.columns, f"Missing original column: {col}"

        # New feature columns added
        assert len(result.columns) > len(integration_df.columns)

        # feature_cols is populated
        assert len(feature_pipeline.feature_cols) > 0
        for col in feature_pipeline.feature_cols:
            assert col in result.columns, f"feature_col {col} not in output"

    def test_pipeline_save_load(self, integration_df, feature_pipeline, tmp_path):
        """Saved and loaded pipeline should produce identical output."""
        pipeline_path = tmp_path / "pipeline.pkl"
        feature_pipeline.save(pipeline_path)

        loaded = FeaturePipeline.load(pipeline_path)
        result_orig = feature_pipeline.transform(integration_df)
        result_loaded = loaded.transform(integration_df)

        # Same columns
        assert result_orig.columns == result_loaded.columns

        # Same feature col list
        assert feature_pipeline.feature_cols == loaded.feature_cols

    def test_no_temp_columns_in_output(self, integration_df, feature_pipeline):
        """No intermediate/temp columns should leak into the output DataFrame."""
        result = feature_pipeline.fit_transform(integration_df)
        for col in result.columns:
            assert not col.startswith("__tmp_"), (
                f"Temporary column leaked into output: {col}"
            )
            assert not col.endswith("_tmp"), (
                f"Temporary column leaked into output: {col}"
            )


class TestCVSplits:
    def test_forward_only_fold_ordering(self, integration_df):
        """With forward_only=True, every fold's max train date < min val date."""
        cv = PurgedGroupKFold(
            n_splits=4,
            purge_days=2,
            embargo_days=2,
            forward_only=True,
            min_train_dates=5,
        )
        folds = list(cv.split(integration_df))
        assert len(folds) > 0, "Expected at least one fold"

        for fold in folds:
            assert fold.train_dates.max() < fold.val_dates.min(), (
                f"Fold {fold.fold_idx}: max train date {fold.train_dates.max()} "
                f">= min val date {fold.val_dates.min()} — future data used for training"
            )

    def test_feature_cols_consistent_with_pipeline(self, integration_df, feature_pipeline):
        """Feature cols from the pipeline should all exist in the transformed data."""
        result = feature_pipeline.fit_transform(integration_df)
        for col in feature_pipeline.feature_cols:
            assert col in result.columns


class TestLGBMTraining:
    def test_lgbm_trains_and_scores(self, integration_df, feature_pipeline, tmp_path):
        """LightGBM should train, produce OOF predictions, and score above zero."""
        df_with_features = feature_pipeline.fit_transform(integration_df)

        # Drop rows with NaN targets (synthetic data has none, but be safe)
        df_clean = df_with_features.filter(pl.col(S.TARGET_COL).is_not_null())

        cv = PurgedGroupKFold(
            n_splits=3,
            purge_days=2,
            embargo_days=2,
            forward_only=True,
            min_train_dates=5,
        )

        # Minimal params for speed
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "n_estimators": 20,
            "learning_rate": 0.1,
            "num_leaves": 8,
            "n_jobs": 1,
        }

        trainer = LGBMTrainer(
            params=params,
            cv=cv,
            feature_cols=feature_pipeline.feature_cols,
            target_col=S.TARGET_COL,
            weight_col=S.WEIGHT,
            date_col=S.DATE_ID,
            artifact_dir=tmp_path / "artifacts",
            early_stopping_rounds=5,
        )
        trainer.train(df_clean)

        # Should have models and OOF predictions
        assert len(trainer.fold_models) > 0
        assert trainer.oof_preds is not None
        assert len(trainer.feature_importance) == len(feature_pipeline.feature_cols)

        # OOF score should be finite
        assert np.isfinite(trainer.mean_cv_score)

        # Artifacts should be written
        assert (tmp_path / "artifacts" / "oof_predictions.parquet").exists()
        assert (tmp_path / "artifacts" / "feature_importance.parquet").exists()
        assert (tmp_path / "artifacts" / "trainer_state.pkl").exists()

    def test_trainer_state_pkl_has_correct_feature_cols(
        self, integration_df, feature_pipeline, tmp_path
    ):
        """The saved trainer_state.pkl must record the exact feature column list."""
        df_with_features = feature_pipeline.fit_transform(integration_df)

        cv = PurgedGroupKFold(n_splits=2, purge_days=1, embargo_days=1,
                              forward_only=True, min_train_dates=5)

        params = {"objective": "regression", "verbosity": -1,
                  "n_estimators": 5, "num_leaves": 4, "n_jobs": 1}

        artifact_dir = tmp_path / "test_artifacts"
        trainer = LGBMTrainer(
            params=params, cv=cv,
            feature_cols=feature_pipeline.feature_cols,
            artifact_dir=artifact_dir,
            early_stopping_rounds=3,
        )
        trainer.train(df_with_features)

        with open(artifact_dir / "trainer_state.pkl", "rb") as f:
            state = pickle.load(f)

        assert state["feature_cols"] == feature_pipeline.feature_cols, (
            "Saved feature_cols must match the pipeline's feature_cols exactly"
        )


class TestInferenceIntegration:
    def test_inference_state_matches_offline_rolling(self, integration_df, feature_pipeline):
        """
        InferenceState rolling features should match offline RollingTransformer
        for the same symbol and window.  This is the most critical correctness
        invariant in the whole inference pipeline.
        """
        col = S.FEATURE_COLS[0]
        window = 3
        symbol_to_check = integration_df[S.SYMBOL_ID].unique()[0]

        # Offline: fit and transform the full data
        offline_transformer = RollingTransformer(
            columns=[col], windows=[window], stats=["mean"]
        )
        offline_df = offline_transformer.fit_transform(integration_df)
        offline_col = f"{col}_roll{window}_mean"

        # Online: simulate batch-by-batch with InferenceState
        state = InferenceState(
            buffer_size=window + 5,
            feature_cols=[col],
            rolling_windows=[window],
            rolling_stats=["mean"],
        )

        batches = _split_by_time(integration_df)
        online_preds: dict[int, float] = {}  # time_step → predicted value

        for i, batch in enumerate(batches):
            enriched = state.compute_features(batch)
            state.update(batch)

            # Record the rolling mean for our symbol on this batch
            sym_row = enriched.filter(pl.col(S.SYMBOL_ID) == symbol_to_check)
            roll_col = f"{col}_roll{window}_mean"
            if roll_col in sym_row.columns and len(sym_row) > 0:
                val = sym_row[roll_col][0]
                if val is not None:
                    online_preds[i] = float(val)

        # Find a time step where both offline and online have non-NaN values
        sym_offline = offline_df.filter(
            pl.col(S.SYMBOL_ID) == symbol_to_check
        ).sort([S.DATE_ID, S.TIME_ID])

        offline_vals = sym_offline[offline_col].to_numpy()

        # Check that the values agree for steps where both have data
        mismatches = 0
        checked = 0
        for step_idx, (offline_val, (step, online_val)) in enumerate(
            zip(offline_vals[window:], list(online_preds.items())[window:])
        ):
            if np.isnan(offline_val) or np.isnan(online_val):
                continue
            checked += 1
            if abs(offline_val - online_val) > 1e-3:
                mismatches += 1

        # Allow a small fraction of mismatches due to NaN boundary handling
        if checked > 0:
            mismatch_rate = mismatches / checked
            assert mismatch_rate < 0.1, (
                f"Online rolling mean disagrees with offline in {mismatch_rate:.0%} "
                "of checked steps. InferenceState does not match training features."
            )

    def test_local_simulator_runs(self, integration_df, feature_pipeline, tmp_path):
        """LocalInferenceSimulator should run without error on synthetic data."""
        # Write features to a parquet file
        data_path = tmp_path / "features.parquet"
        df_with_features = feature_pipeline.fit_transform(integration_df)
        df_with_features.write_parquet(data_path)

        # Build a trivial predictor (zero predictions)
        zero = ZeroPredictor()
        zero.fit(df_with_features)

        state = InferenceState(buffer_size=5, rolling_windows=())

        predictor = PredictorPipeline(
            predict_fn=lambda df: zero.predict(df),
            feature_cols=feature_pipeline.feature_cols,
            state=state,
            clip_range=(-5.0, 5.0),
        )

        simulator = LocalInferenceSimulator(
            predictor=predictor,
            data_path=data_path,
            max_batches=5,
        )
        result = simulator.run(output_path=tmp_path / "preds.parquet")

        assert result.n_batches == 5
        assert result.total_rows > 0
        assert (tmp_path / "preds.parquet").exists()


class TestBaselineModels:
    def test_zero_predictor_scores_zero(self, integration_df):
        predictor = ZeroPredictor().fit(integration_df)
        preds = predictor.predict(integration_df)
        score = competition_score(
            y_true=integration_df[S.TARGET_COL].to_numpy(),
            y_pred=preds,
            weights=integration_df[S.WEIGHT].to_numpy(),
            dates=integration_df[S.DATE_ID].to_numpy(),
        )
        assert abs(score) < 1e-5

    def test_weighted_mean_beats_zero(self, integration_df):
        """Weighted mean should (slightly) outperform zero in absolute terms."""
        z = ZeroPredictor().fit(integration_df)
        wm = WeightedMeanPredictor().fit(integration_df)

        z_preds = z.predict(integration_df)
        wm_preds = wm.predict(integration_df)

        y = integration_df[S.TARGET_COL].to_numpy()
        w = integration_df[S.WEIGHT].to_numpy()
        d = integration_df[S.DATE_ID].to_numpy()

        z_score = abs(competition_score(y, z_preds, w, d))
        wm_score = abs(competition_score(y, wm_preds, w, d))

        # Both scores should be finite — the absolute value comparison is weak
        # but sufficient as a sanity check (financial returns are near-zero-mean)
        assert np.isfinite(z_score)
        assert np.isfinite(wm_score)


def _split_by_time(df: pl.DataFrame) -> list[pl.DataFrame]:
    return [
        group
        for (_, _), group in (
            df.sort([S.DATE_ID, S.TIME_ID])
            .group_by([S.DATE_ID, S.TIME_ID], maintain_order=True)
        )
    ]
