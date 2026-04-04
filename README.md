# Jane Street Real-Time Market Data Forecasting

A production-quality research codebase for the [Kaggle Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) competition.

Built as a **portfolio project** demonstrating end-to-end ML engineering for financial time-series: clean architecture, temporal correctness, reproducibility, and a clear path from baseline to competitive models.

---

## Table of Contents

1. [Competition Context](#competition-context)
2. [Architecture](#architecture)
3. [Setup](#setup)
4. [Workflow](#workflow)
5. [Configuration System](#configuration-system)
6. [Key Design Decisions](#key-design-decisions)
7. [Preventing Target Leakage](#preventing-target-leakage)
8. [Model Stack](#model-stack)
9. [Inference Architecture](#inference-architecture)
10. [Experiment Tracking](#experiment-tracking)
11. [Testing](#testing)
12. [Future Improvements](#future-improvements)

---

## Competition Context

The competition provides anonymised market data and asks participants to predict `responder_6` (a financial return variable) for each instrument at each time step.

**Key properties:**
- **Real-time API**: The Kaggle evaluation harness sends one batch (one `time_id`) at a time. Predictions must be submitted before the next batch arrives.
- **Temporal structure**: Training data is ordered by `(date_id, time_id)`. Future information must never be used.
- **Evaluation metric**: Sum of per-date weighted Pearson correlations between predictions and `responder_6`.
- **Lag features**: The API provides `responder_*_lag_0` (previous time step's responder values) as legitimate model inputs.

**Metric formula:**

```
score = Σ_t [ Σ_i(w_i · r_i · p_i) / sqrt(Σ_i(w_i · r_i²) · Σ_i(w_i · p_i²)) ]
```

where `t` = date, `i` = instrument, `w` = weight, `r` = actual `responder_6`, `p` = prediction. A perfect score = number of unique dates.

---

## Architecture

```
.
├── src/janestreet_forecasting/
│   ├── data/           ← loaders, schemas, validation, CV splits
│   ├── features/       ← lag, rolling, cross-sectional transformers
│   ├── modeling/       ← LightGBM, XGBoost, CatBoost, ensemble, metrics
│   ├── evaluation/     ← backtest, diagnostics, report generation
│   ├── inference/      ← InferenceState, PredictorPipeline, local simulator
│   ├── tuning/         ← Optuna hyperparameter search
│   └── cli/            ← Typer CLI entrypoints (js-* commands)
├── configs/            ← YAML experiment configs
├── notebooks/          ← EDA and interpretation only
├── tests/              ← pytest test suite
├── artifacts/          ← model artifacts (gitignored)
├── reports/            ← evaluation reports (gitignored)
└── data/               ← raw, interim, processed, external (gitignored)
```

### Data Flow

```
data/raw/train.parquet
    ↓ js-prepare  (validate + type-check)
data/interim/train.parquet
    ↓ js-features  (fit feature pipeline, save pipeline.pkl)
data/processed/features.parquet + feature_pipeline.pkl
    ↓ js-train  (PurgedGroupKFold CV)
artifacts/{exp_id}/
    ├── lgbm_fold{0..N}.txt        ← fold models
    ├── oof_predictions.parquet    ← for ensembling + evaluation
    ├── feature_importance.parquet
    └── experiment_config.yaml     ← frozen copy for reproducibility
    ↓ js-evaluate
reports/{exp_id}/
    ├── predictions.parquet
    ├── per_date_metrics.parquet
    ├── feature_importance.png
    └── report.md
    ↓ js-infer serve-local
reports/{exp_id}/local_inference_predictions.parquet
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Kaggle API key](https://www.kaggle.com/docs/api) for data download

### Install

```bash
git clone https://github.com/yourusername/JaneStreetForecastingChallenge.git
cd JaneStreetForecastingChallenge

# Install package + dev dependencies + pre-commit hooks
make env
```

### Download Data

```bash
# Requires KAGGLE_USERNAME and KAGGLE_KEY in environment or .env
make data-download
```

Or manually place `train.parquet` in `data/raw/`.

### Environment Variables (`.env`)

```bash
# Optional — cap row loading for fast local iteration
JS_MAX_ROWS=500000

# Logging
JS_LOG_LEVEL=INFO
JS_LOG_JSON=false

# Kaggle credentials
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

---

## Workflow

### Step 1 — Validate and Prepare Data

```bash
make prepare-data
```

Reads `data/raw/train.parquet`, validates schema and data quality, drops the non-target responder columns (leakage prevention), and writes `data/interim/train.parquet`.

### Step 2 — Build Features

```bash
make build-features
```

Fits the feature pipeline on training data (lag features, rolling statistics, cross-sectional ranks) and writes `data/processed/features.parquet`. The fitted pipeline is saved for use at inference time.

### Step 3 — Train a Baseline Model

```bash
make train-lgbm
```

Runs 5-fold `PurgedGroupKFold` CV, saves fold models, OOF predictions, and feature importance to `artifacts/exp001_lgbm_baseline/`.

### Step 4 — Evaluate

```bash
make evaluate EXP=configs/experiments/exp001_lgbm_baseline.yaml
```

Generates a full report in `reports/exp001_lgbm_baseline/` including per-date metrics, feature importance chart, and worst-symbol breakdown.

### Step 5 — Tune Hyperparameters (Optional)

```bash
make tune EXP=configs/experiments/exp001_lgbm_baseline.yaml
```

Runs 100 Optuna trials and writes best params to `artifacts/exp001_lgbm_baseline/best_params.yaml`. Resume interrupted runs with `--resume`.

### Step 6 — Train All Baselines and Ensemble

```bash
make train-lgbm train-xgb train-catboost
make train-ensemble
```

### Step 7 — Local Inference Simulation

```bash
make infer-local EXP=configs/experiments/exp001_lgbm_baseline.yaml
```

Simulates the Kaggle batch API locally, tracking inference state across batches. Reports per-batch latency and overall metrics.

### Step 8 — Open Notebooks

```bash
make notebook    # Start Jupyter at notebooks/
make mlflow-ui   # Start MLflow UI at http://localhost:5000
```

---

## Configuration System

Experiments are fully described by YAML files in `configs/experiments/`. An experiment config composes four sub-configs and can selectively override any value:

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
  train:
    cv:
      n_splits: 5

tags:
  model: lightgbm
  stage: baseline
```

The experiment config is always copied into the artifact directory — you can reproduce any run from a single YAML file.

---

## Key Design Decisions

### Why Polars?

Polars is 3–8× faster than pandas on large tabular data because:
- Apache Arrow columnar memory layout (cache-friendly, SIMD-friendly)
- Lazy evaluation: filters push down to the Parquet reader, avoiding loading unused data
- Native multi-threading without GIL contention

We use Polars for all data loading and feature engineering, converting to NumPy at the model training boundary. The single copy cost is worth the upstream speed on a 47M-row dataset.

### Why PurgedGroupKFold?

Standard K-fold is **invalid** for financial time-series because autocorrelated errors leak through adjacent folds, making OOF scores overly optimistic.

Our implementation:
- Groups all rows from the same `date_id` together
- **Purges** the `purge_days=5` dates immediately before each test fold from training (prevents rolling feature leakage at the boundary)
- **Embargoes** the `embargo_days=10` dates immediately after each test fold (prevents autocorrelation from future fold training data)

```
Date:  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15
Fold1: ── Train ────────────── [P] [P]  Test Test Test [E]
Fold2: ── Train ────── [P] [P]  Test Test Test [E]  ──────

[P] = purged   [E] = embargoed
```

Reference: *Advances in Financial Machine Learning*, López de Prado (2018), Ch. 7.

### Why Protocol-based Transformers?

Feature transformers implement a `Protocol` (structural typing) rather than inheriting from sklearn's `TransformerMixin`. This:
- Keeps feature code clean — no `get_params`, `set_params`, `fit_transform` boilerplate
- Avoids coupling feature engineering to sklearn's NumPy-array contract
- Makes adding new transformers frictionless

### InferenceState Design

`InferenceState` solves the critical real-time feature engineering problem:

```python
# Correct inference loop order — always compute before updating
for batch in kaggle_api.iter_test():
    features = state.compute_features(batch)  # Uses past history
    predictions = model.predict(features)
    state.update(batch)                        # Update AFTER predicting
    api.predict(predictions)
```

The state maintains a per-symbol ring buffer of the last `buffer_size` observations. Rolling/lag features are computed from this buffer at each step, exactly mirroring offline training.

---

## Preventing Target Leakage

| Risk | Prevention |
|---|---|
| Rolling features fit on full dataset | Feature pipeline **fit on train dates only**, applied forward |
| Non-target responders used as features | `js-prepare` drops `responder_{0-5,7,8}` before feature engineering |
| CV fold overlap | `PurgedGroupKFold` with `purge_days=5, embargo_days=10` |
| Test in `validate_no_future_leakage()` | Asserts max(train_date) < min(val_date) for every fold |

---

## Model Stack

| Model | Config | Notes |
|---|---|---|
| LightGBM | `configs/models/lgbm_baseline.yaml` | Fast, handles NaN natively, typically strongest single model |
| XGBoost | `configs/models/xgb_baseline.yaml` | Good diversity from LightGBM; use `tree_method: hist` for speed |
| CatBoost | `configs/models/catboost_baseline.yaml` | Native NaN handling, different split algorithm |
| Ensemble | `configs/experiments/exp004_ensemble.yaml` | Weights optimised on OOF via Nelder-Mead |

Each trainer follows the same interface:

```python
trainer = LGBMTrainer(params=..., cv=..., feature_cols=..., artifact_dir=...)
trainer.train(df)  # Runs full CV loop
trainer.mean_cv_score  # Overall OOF score
```

---

## Inference Architecture

```
┌─────────────────────────────────────────────────────┐
│ PredictorPipeline                                    │
│                                                      │
│  ┌─────────────────┐   ┌──────────────────────────┐ │
│  │  InferenceState │   │  predict_fn              │ │
│  │                 │   │  (avg of N fold models)  │ │
│  │  Per-symbol     │   │                          │ │
│  │  ring buffer    │   │  LightGBM / XGB / CB     │ │
│  └────────┬────────┘   └───────────┬──────────────┘ │
│           │                        │                │
│  compute_features(batch)  →  predict(X)             │
│  update(batch)  ←── called after prediction         │
└─────────────────────────────────────────────────────┘
```

`LocalInferenceSimulator` feeds data batch by batch to `PredictorPipeline`, measuring per-batch latency. `KaggleAdapter` wraps the same pipeline for Kaggle notebook submission with a clean two-line integration.

---

## Experiment Tracking

**Local (always on)**: Each run writes to `artifacts/{experiment_id}/` with a frozen config copy. Full reproducibility from the config file alone.

**MLflow (optional)**: Set `JS_MLFLOW_TRACKING_URI` or leave empty for local file-store. Start with `make mlflow-ui`.

---

## Testing

```bash
make test        # Full test suite (~30 tests, uses only synthetic data)
make test-cov    # With HTML coverage report in reports/coverage/
make check       # lint + typecheck + test
```

The test suite focuses on the highest-risk correctness invariants:

| File | What it tests |
|---|---|
| `test_splits.py` | Temporal CV: no overlap, no future leakage, purge/embargo |
| `test_features.py` | No look-ahead in lag/rolling features, correct column names |
| `test_metrics.py` | Competition metric: perfect=N, zero=0, sign-flip=negative |
| `test_inference_state.py` | Buffer management, rolling correctness, warm-up |

All tests run without competition data (synthetic data from `make_synthetic_dataset()`).

---

## Future Improvements

**Feature Engineering**
- EWM features (already implemented in `EWMTransformer`, add to default pipeline)
- Multi-symbol features: cross-instrument correlations, index-level signals
- Feature selection via permutation importance or SHAP

**Models**
- MLP baseline with batch normalisation and dropout
- Temporal Fusion Transformer (variable-length time-series)
- Multi-task learning: predict all 9 responders jointly with a shared backbone
- Online learning: incremental model update with each new day's data

**Validation**
- Walk-forward split with expanding window (already implemented in `WalkForwardSplit`)
- Monthly out-of-sample tests to simulate production degradation

**Inference**
- ONNX export for portability across environments
- Model quantisation for faster batch inference
- Async state updates for higher throughput

**Infrastructure**
- GitHub Actions CI pipeline (ruff + mypy + pytest on every push)
- DVC for data versioning
- Hydra config management for large grid searches
