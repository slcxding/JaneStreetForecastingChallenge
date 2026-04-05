# Jane Street Real-Time Market Data Forecasting

End-to-end ML engineering for the [Kaggle Jane Street forecasting competition](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting). Predicts `responder_6` (an anonymised financial return) for 20 instruments across 47M rows of intraday market data.

Built as a portfolio project demonstrating production ML practices: temporal correctness, offline/online feature parity, reproducible experiments, and a clean path from raw data to Kaggle submission.

**Stack**: Python 3.10 · Polars · LightGBM / XGBoost / CatBoost · Optuna · MLflow · Typer · Pydantic

---

## What makes this non-trivial

**Temporal CV done correctly.** Standard k-fold leaks future data through rolling features and autocorrelation. This repo implements `PurgedGroupKFold` with `forward_only=True`: each fold trains only on dates strictly before the test window, with a purge buffer to prevent rolling-feature bleed at boundaries.

```
Date:  0  1  2  3  4  5  6  7  8  9  10  11  12  13
Fold1: TR TR TR TR [P  P] TS TS TS [E]
Fold2: TR TR TR TR TR TR  TR TR [P  P] TS TS TS [E]
       TR=train  [P]=purged  TS=test  [E]=embargoed
```

**Online/offline feature parity.** The Kaggle API sends one time step at a time. `InferenceState` maintains a per-symbol ring buffer so rolling features computed at inference time are numerically identical to those computed offline during training. Verified by an integration test.

**Feature columns sourced from the fitted pipeline.** The trained model's input features come from `feature_pipeline.pkl`, not from exclusion logic applied to the DataFrame. This prevents accidentally including intermediate columns that a buggy transformer left behind.

---

## Repository layout

```
src/janestreet_forecasting/
├── data/           schemas, loaders (Polars lazy), validation, CV splits
├── features/       lag, rolling, cross-sectional transformers + pipeline
├── modeling/       LightGBM / XGBoost / CatBoost trainers, ensemble, metrics
├── inference/      InferenceState (ring buffer), PredictorPipeline, local simulator
├── tuning/         Optuna hyperparameter search (resumable via SQLite)
├── evaluation/     backtest, diagnostics, report generation
└── cli/            Typer CLI: js-prepare / js-features / js-train / js-evaluate / js-tune / js-infer
configs/            YAML experiment configs (composable, with override system)
tests/              62 tests, all synthetic data, ~5 s
```

---

## Workflow

```bash
make env                # install + pre-commit
make prepare-data       # validate raw parquet → data/interim/
make build-features     # fit feature pipeline → data/processed/
make train-lgbm         # 5-fold PurgedGroupKFold CV → artifacts/exp001_lgbm_baseline/
make evaluate EXP=configs/experiments/exp001_lgbm_baseline.yaml
make infer-local EXP=configs/experiments/exp001_lgbm_baseline.yaml  # latency test
make test               # 62 tests
```

Data download (requires Kaggle API key):
```bash
make data-download
```

---

## Model stack

| Model | Config | Notes |
|---|---|---|
| LightGBM | `exp001_lgbm_baseline` | Primary model; handles NaN natively |
| XGBoost | `exp002_xgb_baseline` | Diversity from LightGBM |
| CatBoost | `exp003_catboost_baseline` | Different split algorithm |
| Weighted ensemble | `exp004_ensemble` | Weights optimised on OOF via Nelder-Mead |

Feature engineering: per-symbol lag features, rolling mean/std/min/max/zscore (windows 5/10/20), cross-sectional rank normalisation.

---

## Scoring metric

Weighted, date-averaged Pearson correlation:

```
score = (1/T) Σ_t  [ Σ_i(w_i · r_i · p_i) / sqrt( Σ_i(w_i·r_i²) · Σ_i(w_i·p_i²) ) ]
```

Zero predictions → 0.0. Perfect predictions → 1.0. Implemented in `modeling/metrics.py` and used as the CV objective.

---

## Configuration

Experiments compose four sub-configs and override specific values:

```yaml
# configs/experiments/exp001_lgbm_baseline.yaml
experiment_id: exp001_lgbm_baseline
data_config:     configs/data/default.yaml
features_config: configs/features/default.yaml
model_config:    configs/models/lgbm_baseline.yaml
train_config:    configs/train/default.yaml

overrides:
  model:
    params:
      n_estimators: 500
```

The resolved config is frozen into `artifacts/{exp_id}/experiment_config.yaml` on every run — full reproducibility from a single file.

---

## Testing

```bash
make test       # 62 tests, ~5 s, no competition data required
make test-cov   # + HTML coverage in reports/coverage/
make check      # lint + typecheck + test
```

Key correctness tests:
- `test_max_train_date_strictly_before_min_val_date` — forward-only CV
- `test_inference_state_matches_offline_rolling` — online/offline feature parity
- `test_no_temp_columns_in_output` — pipeline doesn't leak intermediate columns
- `test_trainer_state_pkl_has_correct_feature_cols` — model input features match pipeline

---

## Environment variables (`.env`)

```bash
JS_MAX_ROWS=500000        # cap rows for fast local iteration
JS_LOG_LEVEL=INFO
JS_LOG_JSON=false         # true for CI / log aggregators
JS_MLFLOW_TRACKING_URI=  # empty = local file store
KAGGLE_USERNAME=
KAGGLE_KEY=
```

---

## Extending

The architecture is designed for extension. See the **How to Extend** section in [ARCHITECTURE.md](ARCHITECTURE.md) for concrete starting points covering:

- Deep learning sequence models (LSTM with persistent hidden state)
- Online learning (sliding window retraining, EWM-weighted samples)
- Regime detection (GMM clustering, HMM)
- Feature store (content-addressed Parquet cache)
- Better ensembling (stacking, Bayesian combination, snapshot ensembling)
