# Jane Street Real-Time Market Data Forecasting

A production-quality research codebase for the [Kaggle Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) competition.

Built as a **learning-oriented portfolio project**: every design decision is documented, every module has a clear reason for existing, and the architecture is designed to be extended — not just run once.

---

## Table of Contents

1. [The Competition](#the-competition)
2. [Quick Start](#quick-start)
3. [Data Layout](#data-layout)
4. [Architecture Overview](#architecture-overview)
5. [Module Reference](#module-reference)
6. [The Inference Problem](#the-inference-problem-offline-vs-online)
7. [Validation Strategy](#validation-strategy-why-standard-kfold-is-wrong)
8. [Configuration System](#configuration-system)
9. [Testing Philosophy](#testing-philosophy)
10. [How to Extend This Project](#how-to-extend-this-project)
11. [Makefile Reference](#makefile-reference)

---

## The Competition

### What You're Predicting

Jane Street provides **anonymised market data** and asks you to predict `responder_6` — a financial return variable — for each of 20 instruments at each intraday time step.

**Real data shape (verified against downloaded files):**

| File | Rows | Key Columns |
|---|---|---|
| `train.parquet` (10 partitions) | **47.1M** | `date_id` (0–1689), `time_id` (0–848), `symbol_id` (0–19), 79 features, 9 responders |
| `lag.parquet` (Hive by `date_id`) | — | Previous day's responders: `responder_*_lag_1` |
| `test.parquet` (Hive by `date_id`) | — | Same as train minus responders; adds `row_id`, `is_scored` |
| `features.csv` | 79 | `feature`, `tag_0`..`tag_16` (anonymous metadata tags) |
| `responders.csv` | 9 | `responder`, `tag_0`..`tag_4` |

The 79 features and 9 responders are **fully anonymised**. Jane Street deliberately withholds semantics to prevent overfitting to domain knowledge. The feature metadata (`features.csv`) provides boolean tags grouping related features — a starting point for targeted feature engineering.

### The Scoring Metric

The competition uses a **weighted, date-averaged Pearson correlation**:

```
score = (1/T) × Σ_t  [  Σ_i(w_i · r_i · p_i)  /  sqrt( Σ_i(w_i·r_i²) · Σ_i(w_i·p_i²) )  ]
```

Where:
- `t` = date, `i` = instrument
- `w` = sample weight, `r` = actual `responder_6`, `p` = your prediction
- Each date contributes equally (average, not sum) — so late dates are not upweighted
- **A perfect score = 1.0** (one date with perfect correlation = 1/T)
- **Zero predictions → score = 0.0** (not negative, but useless)

**Intuition**: the metric rewards *ranking* instruments correctly within each date more than predicting exact magnitudes. A model that always predicts the right sign scores well even if magnitudes are off. This means cross-sectional rank features (which date matters most?) are often more valuable than time-series features (what is the absolute level?).

### The Real-Time API Constraint

At evaluation time, Kaggle sends one batch at a time — **one `time_id`** across all 20 symbols. Your notebook must return predictions before the next batch arrives. This creates a strict engineering constraint:

```
# Kaggle's evaluation harness, simplified
for test_df, lags_df, sample_sub in iter_test():
    # test_df: current time step's features (20 rows)
    # lags_df: previous time step's responder_*_lag_0
    # You have ~500ms to return predictions
    sample_sub["responder_6"] = your_predict_fn(test_df, lags_df)
    env.predict(sample_sub)
```

**This changes everything about how you engineer features.** You cannot sort the full dataset and compute rolling means. You need to maintain a rolling buffer of past observations and update it incrementally. This is what `InferenceState` solves — see [The Inference Problem](#the-inference-problem-offline-vs-online).

---

## Quick Start

```bash
# 1. Install
git clone <this-repo> && cd JaneStreetForecastingChallenge
make env

# 2. Prepare data (assumes data/ already downloaded)
make prepare-data          # Validate + write data/interim/train.parquet

# 3. Build features
make build-features        # Fit pipeline, write data/processed/features.parquet

# 4. Train LightGBM baseline
make train-lgbm            # 5-fold PurgedGroupKFold CV → artifacts/exp001_lgbm_baseline/

# 5. Evaluate
make evaluate EXP=configs/experiments/exp001_lgbm_baseline.yaml

# 6. Simulate inference locally
make infer-local EXP=configs/experiments/exp001_lgbm_baseline.yaml
```

Without competition data, run tests on synthetic data:

```bash
make test    # 62 tests, all synthetic, ~5 seconds
```

---

## Data Layout

```
data/
├── raw/
│   ├── train.parquet/              ← Hive-partitioned, 47.1M rows total
│   │   ├── partition_id=0/part-0.parquet   # ~1.9M rows, dates 0–169
│   │   ├── partition_id=1/part-0.parquet
│   │   └── ...  (10 partitions total)
│   │
│   ├── lag.parquet/                ← Hive-partitioned by date_id
│   │   ├── date_id=0/part-0.parquet        # Previous day's responders
│   │   └── ...
│   │
│   ├── test.parquet/               ← Same structure, no responder columns
│   │   └── date_id=0/part-0.parquet
│   │
│   ├── features.csv                ← Feature metadata (tags only, no names)
│   ├── responders.csv              ← Responder metadata
│   └── sample_submission.csv       ← row_id, responder_6 (all zeros)
│
├── interim/                        ← After js-prepare (validated, typed)
│   └── train.parquet
│
└── processed/                      ← After js-features (engineered)
    ├── features.parquet
    └── feature_pipeline.pkl        ← Fitted pipeline (needed at inference time)
```

### Column-Level Details

**train.parquet** columns (92 total):

| Column(s) | Type | Notes |
|---|---|---|
| `date_id` | Int16 | 0–1689; each value = one trading day |
| `time_id` | Int16 | 0–848; intraday step within the date |
| `symbol_id` | Int8 | 0–19; exactly 20 instruments |
| `weight` | Float32 | Non-negative; row's contribution to the score |
| `feature_00`..`feature_78` | Float32 | 79 anonymised features, ~5% missing |
| `responder_0`..`responder_8` | Float32 | 9 return variables; `responder_6` = target |

**lag.parquet** columns (12 total):

| Column(s) | Notes |
|---|---|
| `date_id`, `time_id`, `symbol_id` | Join keys |
| `responder_0_lag_1`..`responder_8_lag_1` | Previous calendar day's responders |

**Why a separate lag file?** Because at inference time you don't have yesterday's responders in the current batch — the API provides them separately. Mirroring this separation in the training data forces you to think about the join explicitly, preventing accidental leakage.

---

## Architecture Overview

### Component Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE PIPELINE                                 │
│                                                                         │
│  data/raw/           data/interim/       data/processed/                │
│  train.parquet  ──►  train.parquet  ──►  features.parquet              │
│  lag.parquet    ──►  (validated,    ──►  feature_pipeline.pkl          │
│                       typed)                     │                      │
│                                                  │                      │
│                                    ┌─────────────▼──────────────────┐  │
│                                    │  PurgedGroupKFold CV            │  │
│                                    │  ┌────────┐  ┌────────┐        │  │
│                                    │  │ Fold 1 │  │ Fold 2 │  ...   │  │
│                                    │  │ Train  │  │ Train  │        │  │
│                                    │  │ Val    │  │ Val    │        │  │
│                                    │  └────────┘  └────────┘        │  │
│                                    └──────────────┬─────────────────┘  │
│                                                   │                     │
│                                    ┌──────────────▼──────────────────┐ │
│                                    │  artifacts/{exp_id}/            │ │
│                                    │  lgbm_fold0.txt .. foldN.txt   │ │
│                                    │  oof_predictions.parquet        │ │
│                                    │  feature_importance.parquet     │ │
│                                    │  trainer_state.pkl              │ │
│                                    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        ONLINE INFERENCE                                 │
│                                                                         │
│   Kaggle API (or LocalInferenceSimulator)                               │
│                                                                         │
│   ┌────────────┐    ┌──────────────────────────────────────────────┐   │
│   │  Batch     │    │  PredictorPipeline                           │   │
│   │  t=0       │───►│                                              │   │
│   │  (20 rows) │    │  ┌─────────────────┐  ┌──────────────────┐  │   │
│   └────────────┘    │  │  InferenceState │  │  predict_fn      │  │   │
│                     │  │                 │  │  (avg of N fold  │  │   │
│   ┌────────────┐    │  │  per-symbol     │  │   models)        │  │   │
│   │  Batch     │───►│  │  ring buffer    │  └────────┬─────────┘  │   │
│   │  t=1       │    │  │  (maxlen=30)    │           │            │   │
│   └────────────┘    │  └────────┬────────┘           │            │   │
│        ↑            │           │                     │            │   │
│    next batch       │  compute_features(batch)        │predict(X)  │   │
│                     │  update(batch)  ←── after pred  │            │   │
│                     └─────────────────────────────────┴────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow (CLI commands)

```
js-prepare    ──► js-features ──► js-train ──► js-evaluate
                                      │
                                      └──► js-tune (optional Optuna search)
                                      └──► js-infer (local simulation)
```

---

## Module Reference

Every module below has a **Why it exists** section explaining the problem it solves, not just what it does.

### `data/schemas.py` — The Single Source of Truth

**Why it exists**: Column names (`"date_id"`, `"responder_6"`, etc.) appear in dozens of places. If they're scattered as string literals, a rename or typo creates silent failures — you get empty DataFrames instead of errors. `schemas.py` centralises every column name as a typed constant. All other modules import from here.

```python
from janestreet_forecasting.data import schemas as S

df.filter(pl.col(S.DATE_ID) > 100)      # not pl.col("date_id")
target = df[S.TARGET_COL]               # not df["responder_6"]
```

Also documents the actual data structure with real row counts and dtypes — acts as living documentation that's always co-located with the code that uses it.

---

### `data/loaders.py` — Lazy Parquet Loading

**Why it exists**: The training set is 47M rows. Loading it all into memory before applying filters wastes RAM and time. Polars' `scan_parquet` + predicate push-down reads only the row groups you need from disk. The loaders provide a consistent interface that abstracts over Hive-partitioned directories vs single files.

Key design choices:
- **Returns `LazyFrame`**, not `DataFrame`. Callers call `.collect()` when they actually need the data. This lets the query planner optimise across chained operations.
- **`date_range` filter** pushes down to Parquet row groups — on a 47M-row dataset, filtering to one year (~7M rows) reads only those row groups, not the full file.
- **`load_lag()`** is separate from `load_train()` — this mirrors the actual API contract where lag data arrives as a separate DataFrame. Keeping them separate during training prevents accidental joins that would introduce leakage.

```python
# Efficient: reads only partition_id=3–5, filters, never touches others
lf = load_train("data/raw/train.parquet", date_range=(500, 1000))
df = lf.collect()   # materialise only when needed
```

---

### `data/validation.py` — Schema Enforcement

**Why it exists**: Parquet files don't enforce schemas — columns can be missing, renamed, or have wrong types without raising errors until training. `validate_dataframe` catches these issues immediately after loading, before hours of feature engineering produce wrong results. Returns a `ValidationReport` rather than raising — callers decide whether to abort or warn.

---

### `data/splits.py` — Time-Aware Cross-Validation

**Why it exists**: Standard k-fold CV is **invalid for financial time-series**. See [Validation Strategy](#validation-strategy-why-standard-kfold-is-wrong) for the full explanation.

Three splitters:

| Class | Use case | Key property |
|---|---|---|
| `PurgedGroupKFold(forward_only=True)` | Production CV | Trains only on dates before each test fold |
| `PurgedGroupKFold(forward_only=False)` | HP search proxy | More training data per fold; not temporally honest |
| `WalkForwardSplit` | Final diagnostics | Strictest; expanding window, never looks forward |

The `FoldIndices` dataclass returned by `.split()` carries both row indices (for NumPy indexing) and date arrays (for debugging temporal boundaries). The date arrays make it easy to confirm visually that train never overlaps val.

---

### `features/base.py` — Transformer Interface

**Why it exists**: Feature transformers need a consistent interface so `FeaturePipeline` can compose them generically. We use Python's `Protocol` (structural typing) rather than abstract base class inheritance. This means you can implement a transformer as a plain Python class that just has `fit()`, `transform()`, and `feature_names_out` — no need to inherit from anything or implement `get_params`/`set_params`.

```python
# A valid transformer — no inheritance required
class MyTransformer:
    def fit(self, df: pl.DataFrame, **kwargs) -> "MyTransformer":
        ...
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        ...

    @property
    def feature_names_out(self) -> list[str]:
        return [...]
```

`BaseTransformer` ABC is provided for convenience — it adds `fit_transform()` and validates that `transform()` isn't called before `fit()`. Use it or not.

---

### `features/lag_features.py` — Additional Lag Features

**Why it exists**: The Kaggle API provides `responder_*_lag_1` (previous day's responders). But you might also want `responder_6_lag_2` (2 days back) or `responder_6_lag_5`. `LagTransformer` computes these additional lags from the `InferenceState` buffer at inference time, and from the sorted DataFrame at training time.

**Critical correctness constraint**: Lags must be computed **per symbol**, not per row. In a sorted DataFrame, the row before symbol A at time t is symbol B at time t — not symbol A at time t-1. The transformer uses Polars' `.over(symbol_id)` window to enforce per-symbol grouping.

```python
# Correct — shifts within each symbol's group
pl.col(col).shift(lag).over(S.SYMBOL_ID).alias(f"{col}_lag{lag}")

# WRONG — shifts across symbol boundaries (produces garbage)
pl.col(col).shift(lag).alias(...)  # ← this is a bug
```

---

### `features/rolling_features.py` — Rolling Statistics

**Why it exists**: A model that sees only one time step cannot learn trends. Rolling statistics (`mean`, `std`, `min`, `max`, `zscore` over windows of 5, 10, 20 steps) give the model a "memory" of recent history. This is the primary bridge between the static GBDT model and the time-series nature of the data.

**Design decisions**:
- `zscore` is computed with inline expressions (no temp columns). An earlier version created `__tmp_mean_*` columns that leaked into the output DataFrame and became accidental model features.
- `"skew"` was intentionally removed from supported stats — the naive implementation computed `x - rolling_mean` (deviation), not third-moment skewness. A mislabeled feature causes silent model degradation.
- Windows are configured per-experiment in `configs/features/default.yaml` so ablation studies don't require code changes.

**Online/offline parity**: the rolling mean at step t in `RollingTransformer` includes the current row (standard Polars semantics). `InferenceState` must match: its `compute_features()` combines the past buffer with the current batch's values before computing rolling stats. This is validated by `test_inference_state_matches_offline_rolling` in the integration test suite.

---

### `features/cross_features.py` — Cross-Sectional Features

**Why it exists**: At each time step, you have 20 symbols. Cross-sectional rank features answer "is this symbol's feature_00 high or low relative to the other symbols right now?" This is often more predictive than absolute feature values because:
1. Anonymised features have no inherent scale — ranks are scale-free
2. Market factors are often cross-sectional: momentum works by buying the top decile of performers, not buying everything that went up

`CrossSectionalRankTransformer` ranks each feature within each `(date_id, time_id)` group and optionally z-scores the ranks. The z-score converts ordinal ranks to a roughly normal distribution, which many models (especially MLPs) handle better than uniform ranks.

**Warning**: Cross-sectional features computed at inference time require all 20 symbols to be present in the batch. If the API ever sends a partial batch, ranks will be computed on fewer symbols — a silent correctness bug. The transformer does not validate batch completeness; add this check if robustness matters.

---

### `features/pipelines.py` — Pipeline Composition and Serialisation

**Why it exists**: Feature engineering has a **training/serving gap problem**. You fit statistics (e.g., rolling means, rank distributions) on training data and must reproduce them exactly at inference time. A plain function cannot be serialised with its state. `FeaturePipeline` serialises the entire fitted pipeline (all transformers + fitted state) to a pickle file that the inference system loads.

The `feature_cols` property is the **authoritative list of model input features** — it comes from the fitted pipeline, not from "all columns except index/target". This distinction matters because leaked temporary columns (from bugs like the old zscore implementation) would be silently included if you derived feature_cols by exclusion.

```python
# Correct — load from saved pipeline
pipeline = FeaturePipeline.load("data/processed/feature_pipeline.pkl")
feature_cols = pipeline.feature_cols   # exactly what the model was trained on

# Wrong — would include any leaked temp columns
feature_cols = [c for c in df.columns if c not in {"date_id", "symbol_id", ...}]
```

---

### `modeling/metrics.py` — The Actual Competition Metric

**Why it exists**: Training on RMSE and evaluating on a different metric is a common mistake. `competition_score()` implements the exact weighted, date-averaged Pearson correlation used by Kaggle. OOF CV scores computed with this function are directly comparable to the leaderboard.

The implementation decomposes by date (one Pearson correlation per date, then average) rather than computing one Pearson over all rows. This matters: a model that scores perfectly on half the dates and fails on the other half gets 0.5, not 0 — but a model that predicts the wrong sign on all rows of the last date (which has high weight rows) should still score close to 0 on that date.

---

### `modeling/train_lgbm.py` / `train_xgb.py` / `train_catboost.py` — Trainers

**Why separate files?**: Each GBDT library has its own API for early stopping, custom objectives, callbacks, and model serialisation. Keeping them separate makes each file self-contained and easy to replace. The interface is identical — `Trainer.train(df)` — so `cli/train.py` dispatches by `model_type` in the config without knowing implementation details.

**What `LGBMTrainer.train()` does**:
1. Calls `cv.split(df)` to get fold indices
2. For each fold: extracts `X_train, y_train, X_val, y_val` as NumPy arrays
3. Trains LightGBM with early stopping on the val set
4. Accumulates OOF predictions (one row per val row, in original order)
5. After all folds: saves models, OOF predictions, feature importance, and `trainer_state.pkl`

`trainer_state.pkl` contains `feature_cols` — the exact column list used during training. This is the handshake between training and inference: the inference system loads this pkl to know what columns to request.

---

### `modeling/ensemble.py` — Ensemble with OOF-Optimised Weights

**Why it exists**: Multiple models (LightGBM, XGBoost, CatBoost) trained on the same folds will make correlated errors, but their errors don't fully overlap. Averaging reduces variance. The optimal weights are found by minimising OOF loss over the training set using Nelder-Mead simplex optimisation with a simplex constraint (weights sum to 1, all non-negative).

```
w* = argmin_{w ≥ 0, Σw=1} OOF_score(Σ_i w_i · oof_preds_i)
```

**Why OOF and not validation?** Because OOF predictions were made on held-out data — the model hasn't seen these rows during training, so OOF scores are out-of-sample. Using a separate held-out validation set for weight optimisation wastes data; using training predictions would overfit dramatically.

---

### `modeling/baselines.py` — Zero and Weighted-Mean Predictors

**Why it exists**: Every ML project needs baseline scores to calibrate results. `ZeroPredictor` gives score=0.0 (by design of the metric). `WeightedMeanPredictor` predicts the training-weighted mean target for all rows. If your LightGBM model underperforms `WeightedMeanPredictor`, something is wrong.

Also used in tests: `ZeroPredictor` makes the `test_local_simulator_runs` test fast (no model loading) while still exercising the full inference pipeline.

---

### `inference/state.py` — The Online Feature Store

**Why it exists**: See [The Inference Problem](#the-inference-problem-offline-vs-online) below. This is the most architecturally important module in the codebase.

---

### `inference/predict.py` — PredictorPipeline

**Why it exists**: Bundles three things that must always travel together:
1. **`predict_fn`**: a function that takes a feature DataFrame and returns predictions (typically averages N fold models)
2. **`InferenceState`**: the rolling buffer that computes features from history
3. **Clip range**: predictions are clipped to `(-5, 5)` to prevent extreme predictions from exploding the weighted metric

The `predict(batch)` method handles the correct order: compute features (using past history) → predict → update state (record current batch). This order cannot be wrong if callers use `PredictorPipeline.predict()` instead of calling the components directly.

---

### `inference/serve_local.py` — LocalInferenceSimulator

**Why it exists**: The Kaggle inference environment is opaque — you can only test your notebook by submitting. `LocalInferenceSimulator` replays the real data batch by batch (one `time_id` at a time), calling your predictor exactly as Kaggle would, and measures per-batch latency. If your batch latency > ~500ms, you'll time out on Kaggle before fixing it.

Also used in `test_local_simulator_runs` to verify the full inference wiring is correct — a test that exercises `InferenceState → PredictorPipeline → LocalInferenceSimulator` is the closest thing to a staging environment for the Kaggle submission.

---

### `inference/kaggle_adapter.py` — The Submission Shim

**Why it exists**: The Kaggle submission notebook needs a two-line integration:

```python
from janestreet_forecasting.inference.kaggle_adapter import KaggleAdapter
adapter = KaggleAdapter.from_artifacts("artifacts/exp001_lgbm_baseline")

for test_df, lags_df, sample_sub in iter_test():
    sample_sub["responder_6"] = adapter.predict(test_df, lags_df)
    env.predict(sample_sub)
```

The adapter loads the fitted pipeline, the model artifacts, and the trainer state, then wraps `PredictorPipeline`. This is the only Kaggle-specific code — everything else is framework-agnostic.

---

### `tuning/optuna_runner.py` — Hyperparameter Search

**Why it exists**: LightGBM has 30+ hyperparameters and the optimal values depend on data size, feature count, and the target distribution. Manual grid search misses interactions (a larger `num_leaves` requires a smaller `learning_rate`). Optuna's TPE sampler models the hyperparameter → score function as a probabilistic model and samples from regions likely to improve, using far fewer trials than grid search.

Key implementation choices:
- **SQLite storage** (`optuna_study.db`): interrupted runs resume automatically. Optuna loads the study from the database and continues where it left off.
- **`MedianPruner`**: kills unpromising trials early (after 5 CV folds with worse median score than past trials). Saves hours on large searches.
- **Dict copy before mutation**: the params dict is copied per trial before `n_estimators` is extracted. The original was mutated with `.pop()` and then `.update()` — not atomic, causing race conditions with `n_jobs > 1`.

---

### `cli/` — Typer CLI Commands

**Why Typer?** Typer generates `--help` automatically, validates argument types, and keeps CLIs readable without boilerplate. Each command is a standalone `@app.command()` function that imports only what it needs — `cli/train.py` doesn't import `optuna`, `cli/tune.py` doesn't import `lightgbm`.

| Command | Module | What it does |
|---|---|---|
| `js-prepare` | `cli/prepare_data.py` | Validate raw data → `data/interim/` |
| `js-features` | `cli/build_features.py` | Fit pipeline → `data/processed/` |
| `js-train` | `cli/train.py` | PurgedGroupKFold CV → `artifacts/` |
| `js-evaluate` | `cli/evaluate.py` | Score OOF + generate report |
| `js-tune` | `cli/tune.py` | Optuna search, resumable |
| `js-infer` | `cli/infer.py` | Local inference simulation |

---

## The Inference Problem: Offline vs Online

This is the hardest architectural challenge in this codebase, and getting it wrong causes silent model degradation.

### The Problem

During **offline training**, you compute a rolling 10-step mean of `feature_00` for symbol 3:

```python
# Polars rolling_mean — fast, vectorised, includes current row
df.with_columns(
    pl.col("feature_00")
      .rolling_mean(window_size=10, min_samples=1)
      .over("symbol_id")
      .alias("feature_00_roll10_mean")
)
```

During **online inference**, you receive 20 rows (one `time_id`). You cannot run `rolling_mean` on 20 rows and get a 10-step mean — you only have this one time step in memory.

### The Solution: InferenceState

`InferenceState` maintains a per-symbol ring buffer (`collections.deque(maxlen=buffer_size)`) of past observations. At each time step:

```
┌──────────────────────────────────────────────────┐
│  InferenceState._buffers[symbol_id=3]            │
│                                                  │
│  deque(maxlen=30):                               │
│  [row_t-9, row_t-8, row_t-7, ..., row_t-1]      │
│              (oldest)              (newest)      │
└──────────────────────────────────────────────────┘

compute_features(batch_t):
  for each symbol:
    col_history = buffer[-9:]          # last 9 buffered values
    col_with_current = history + [current_value]  # include current row
    rolling_mean_10 = mean(col_with_current)      # exactly 10 values

update(batch_t):
  for each symbol:
    buffer.append(current_row)         # evicts oldest if buffer full
```

### The Critical Invariant

The `compute_features()` implementation **includes the current batch's values** in the rolling window. This matches the offline `rolling_mean` which includes the current row. If you exclude the current row online, every rolling feature is off by one step — the model sees systematically different features than it was trained on.

This parity is explicitly tested:

```python
# tests/test_integration.py
def test_inference_state_matches_offline_rolling():
    # Train offline transformer on full data
    # Simulate online step-by-step
    # Assert rolling mean values agree within 1e-3
```

### The Ordering Rule

```python
# Always in this order — never reversed:
enriched = state.compute_features(batch)  # uses buffer (past data)
predictions = model.predict(enriched)     # make prediction
state.update(batch)                       # add current to buffer
```

Reversing this would make step t's features use step t's data in the buffer — equivalent to using the future in training, which is leakage.

---

## Validation Strategy: Why Standard KFold is Wrong

### The Problem with Standard KFold

Standard k-fold shuffles rows randomly and puts some in train, some in val. For financial data:

1. **Autocorrelation leakage**: market at time t is correlated with t-1. If t is in train and t-1 is in val, the val row's rolling feature was computed from t — which is in train. The model "cheats" by seeing training data through the feature's history.

2. **Rolling feature overlap**: a 10-period rolling mean at time t uses data from t-9..t. If any of those steps appear in both train and val, information leaks.

3. **Distribution shift**: models trained on shuffled data learn "what is the average return" rather than "what is the return given recent history". Real-time prediction requires the latter.

### Our Solution: PurgedGroupKFold with forward_only=True

```
Date:  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19
       ────────────────────────────────────────────────────────────────────────
Fold0: Skip (insufficient training history)
Fold1: [TR TR TR TR] [P  P] [TS TS TS TS] [E  E] ──────────────────────────
Fold2: [TR TR TR TR  TR TR  TR TR TR TR] [P  P] [TS TS TS TS] [E  E] ───────
Fold3: [TR TR TR TR  TR TR  TR TR TR TR  TR TR  TR TR TR TR] [P  P] [TS TS]

TR = train   [P] = purged (adjacent to test)   TS = test   [E] = embargo
```

**Purge** (`purge_days=5`): dates immediately before the test fold are excluded from training. A 5-period rolling mean at the first test date uses 5 steps of "recent" data — if those are in training, the model has effectively seen what's happening near the test boundary. Purging removes this overlap.

**Embargo** (`embargo_days=10`): dates immediately after the test fold are excluded from training in future folds. Market autocorrelation means test-fold events influence subsequent returns; allowing the model to train on those subsequent dates would leak information about the test period.

**forward_only=True**: each fold trains only on dates that come **chronologically before** the test fold. Without this, fold 0 could test on dates 0–4 while training on dates 5–19 — the model would have seen the future market regimes during training. With `forward_only=True`, early folds are skipped when there is insufficient training history (controlled by `min_train_dates`).

**Reference**: Advances in Financial Machine Learning, López de Prado (2018), Chapter 7.

---

## Configuration System

Experiments compose four sub-configs and can override any value:

```yaml
# configs/experiments/exp001_lgbm_baseline.yaml
experiment_id: exp001_lgbm_baseline

data_config:     configs/data/default.yaml
features_config: configs/features/default.yaml
model_config:    configs/models/lgbm_baseline.yaml
train_config:    configs/train/default.yaml

overrides:                      # ← selectively override sub-configs
  model:
    params:
      n_estimators: 1000
      num_leaves: 127
  train:
    cv:
      n_splits: 7

tags:
  model: lightgbm
  stage: iteration_2
```

The override system uses deep merge (`_deep_merge` in `cli/train.py`): nested dicts are merged recursively, scalar values are replaced. This lets you change `params.n_estimators` without repeating the entire model config.

**The frozen config**: every run copies the full resolved config into `artifacts/{exp_id}/experiment_config.yaml`. You can reproduce any past run from just this file — no reconstruction of what flags were passed to which command.

---

## Testing Philosophy

Tests focus on the highest-risk correctness invariants — not line coverage.

```bash
make test       # 62 tests, ~5 seconds, all synthetic data
make test-cov   # + HTML coverage report in reports/coverage/
```

### What We Test and Why

| Test file | Critical invariant |
|---|---|
| `test_splits.py` | `max(train_dates) < min(val_dates)` for every fold in `forward_only=True` mode. **A bug here means training on future data — all CV scores are meaningless.** |
| `test_features.py` | Per-symbol correctness of lag/rolling; no cross-symbol contamination; `__tmp_*` columns don't leak |
| `test_integration.py::TestInferenceIntegration::test_inference_state_matches_offline_rolling` | Online `InferenceState` rolling features match offline `RollingTransformer` values. **A bug here means inference features differ from training features — a silent model quality regression.** |
| `test_integration.py::TestLGBMTraining::test_trainer_state_pkl_has_correct_feature_cols` | Saved `feature_cols` in trainer_state.pkl match pipeline's `feature_cols`. **A mismatch means inference uses different columns than training.** |
| `test_metrics.py` | Competition score: perfect=1.0, zero=0.0, sign-flip=negative |

### What We Don't Test

- Real data loading (uses synthetic data only — competition data isn't in CI)
- Model accuracy (we test that training completes and scores are finite, not that they're competitive)
- Inference latency (measured by `LocalInferenceSimulator`, not a test assertion)

---

## How to Extend This Project

The architecture is designed to make extensions low-friction. Each extension area below explains the concept, the key design challenge, and a concrete starting point.

---

### 1. Deep Learning Sequence Models

**Concept**: GBDTs see each row independently (even with rolling features). A sequence model (LSTM, Transformer) sees the last T steps of a symbol's history and learns temporal patterns that rolling statistics miss.

**The key challenge**: the Kaggle API sends one time step at a time. You can't batch symbols and time steps together as you would for offline training. Your sequence model must maintain **hidden state** across calls, just like `InferenceState`.

**Starting point** — add a new transformer and trainer:

```python
# src/janestreet_forecasting/modeling/train_lstm.py
import torch
import torch.nn as nn

class SymbolLSTM(nn.Module):
    """Per-symbol LSTM. Hidden state persists across time steps at inference."""
    def __init__(self, input_size: int, hidden_size: int = 64, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, state=None):
        # x: (batch_size=n_symbols, seq_len=1, features)
        out, new_state = self.lstm(x, state)
        return self.head(out[:, -1, :]), new_state  # return state for next step


class LSTMInferenceState:
    """Carries LSTM hidden state across inference batches (replaces InferenceState)."""
    def __init__(self, model: SymbolLSTM):
        self.model = model
        self._states: dict[int, tuple] = {}  # symbol → (h, c) tensors

    def predict(self, batch: pl.DataFrame) -> np.ndarray:
        predictions = []
        for symbol in batch["symbol_id"].to_list():
            x = self._batch_to_tensor(batch, symbol)       # (1, 1, n_features)
            state = self._states.get(symbol)
            pred, new_state = self.model(x, state)
            self._states[symbol] = new_state
            predictions.append(pred.item())
        return np.array(predictions)
```

**Training**: use `PurgedGroupKFold` with a custom `LSTMTrainer` that formats batches as sequences `(symbol, time, features)`. Sequence length = 30 (matching `InferenceState.buffer_size`). Use teacher forcing for fast convergence.

**Architecture options**:
- **LSTM/GRU**: simpler, faster, good for short sequences
- **Temporal Fusion Transformer (TFT)**: interpretable attention weights, variable-length history
- **N-BEATS / N-HiTS**: strong on univariate series; requires adapting for multi-variate
- **PatchTST**: treats time-series as patches (like Vision Transformer); emerging SOTA

---

### 2. Online Learning

**Concept**: financial markets change regimes. A model trained on 2018–2022 data may underperform in 2024 because the signal-to-noise ratio, factor exposures, and correlations have shifted. Online learning updates model weights as new data arrives, adapting continuously.

**Key challenge**: updating tree-based models (LightGBM, XGBoost) incrementally is hard — they don't support gradient updates. Options:

**Option A — Sliding window retraining** (easiest, already supported):

```python
# Retrain every N days using only the last K days of data
def sliding_window_retrain(date_id: int, window_days: int = 180):
    lf = load_train(
        "data/raw/train.parquet",
        date_range=(date_id - window_days, date_id - 1)
    )
    df = lf.collect()
    trainer = LGBMTrainer(...)
    trainer.train(df)
    trainer.save(f"artifacts/online/model_at_date_{date_id}.pkl")
```

**Option B — Online gradient boosting** (new models):

```python
# river: sklearn-compatible online learning library
from river import ensemble, linear_model, optim

model = ensemble.AdaBoostClassifier(
    model=linear_model.LogisticRegression(optimizer=optim.SGD(lr=0.01)),
    n_models=10,
)

# At each time step:
for batch in iter_batches():
    x = {col: batch[col][0] for col in feature_cols}
    y = batch["responder_6"][0]
    model.learn_one(x, y > 0)  # online update
    pred = model.predict_proba_one(x)
```

**Option C — EWM-weighted training** (recommended for this competition):

```python
# Weight recent data more heavily in the loss function
# LightGBM accepts per-sample weights
decay = 0.995  # exponential decay rate
days_old = current_date - df["date_id"]
sample_weights = decay ** days_old  # recent data weighted higher
trainer = LGBMTrainer(params=..., weight_col="dynamic_weight")
df = df.with_columns(
    (pl.lit(decay) ** (current_date - pl.col("date_id"))).alias("dynamic_weight")
)
```

**Add to `InferenceState`**: a `daily_update()` method that triggers retraining after each day's data is received, replacing the models in `PredictorPipeline` without restarting the inference server.

---

### 3. Regime Detection

**Concept**: markets behave differently in different "regimes" — trending vs mean-reverting, high-volatility vs low-volatility, risk-on vs risk-off. A single model trained over all regimes learns the average behaviour, missing regime-specific patterns. Regime-conditioned models or a gating network that routes to the right model can improve.

**Starting point — unsupervised regime clustering**:

```python
# src/janestreet_forecasting/features/regime_features.py
from sklearn.mixture import GaussianMixture

class RegimeDetector:
    """
    Detects market regimes from cross-sectional feature statistics.

    Fits a Gaussian Mixture Model on daily aggregates (mean, std, skew of
    cross-sectional feature distributions). At inference time, assigns each
    day to a regime.

    The regime label can be used as:
    - A categorical feature (tell the model what regime it's in)
    - A routing key (pick the model trained on this regime)
    - A weight adjustment (upweight in-regime training samples)
    """
    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=random_state)

    def fit(self, df: pl.DataFrame) -> "RegimeDetector":
        daily_stats = self._compute_daily_stats(df)
        self.gmm.fit(daily_stats)
        return self

    def predict_regime(self, df: pl.DataFrame) -> np.ndarray:
        daily_stats = self._compute_daily_stats(df)
        return self.gmm.predict(daily_stats)

    def _compute_daily_stats(self, df: pl.DataFrame) -> np.ndarray:
        # Aggregate features cross-sectionally per date
        return (
            df.group_by("date_id")
            .agg([
                pl.col(c).mean().alias(f"{c}_mean")
                for c in S.FEATURE_COLS[:10]  # subset for speed
            ] + [
                pl.col(c).std().alias(f"{c}_std")
                for c in S.FEATURE_COLS[:10]
            ])
            .sort("date_id")
            .drop("date_id")
            .to_numpy()
        )
```

**Using regime labels**:

```python
# Option 1: regime as a feature
df = df.with_columns(pl.Series("regime", regime_detector.predict_regime(df)))

# Option 2: train separate models per regime
for regime_id in range(n_regimes):
    regime_df = df.filter(pl.col("regime") == regime_id)
    trainer = LGBMTrainer(...)
    trainer.train(regime_df)

# Option 3: upweight in-regime samples (soft routing)
current_regime_probs = regime_detector.gmm.predict_proba(current_daily_stats)
df = df.with_columns(
    pl.Series("regime_weight", current_regime_probs[df["date_id"].to_numpy()])
)
```

**Hidden Markov Models** (HMM) are a more principled approach: they model regime transitions probabilistically, giving you both the current regime and the probability of switching. `hmmlearn` implements this with a clean sklearn-style API.

---

### 4. Feature Store Ideas

**Concept**: As you iterate on features, you recompute the same lag/rolling features repeatedly. A feature store pre-computes and caches expensive features so experiments can reuse them without waiting for re-computation. For a 47M-row dataset, this is the difference between a 2-hour iteration cycle and a 10-minute one.

**Minimal feature store** — tagged Parquet cache:

```python
# src/janestreet_forecasting/features/store.py
import hashlib, json
from pathlib import Path
import polars as pl

class FeatureStore:
    """
    Content-addressed Parquet cache for computed features.

    Each feature set is identified by a hash of its computation config.
    If a config was already computed, load from cache. Otherwise compute,
    cache, and return.

    Usage:
        store = FeatureStore("data/feature_store/")
        df = store.get_or_compute(
            config={"rolling_windows": [5, 10], "columns": ["feature_00"]},
            compute_fn=lambda: rolling_transformer.fit_transform(raw_df),
        )
    """
    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def get_or_compute(
        self,
        config: dict,
        compute_fn,
        force: bool = False,
    ) -> pl.DataFrame:
        key = self._hash_config(config)
        cache_path = self.store_dir / f"{key}.parquet"

        if cache_path.exists() and not force:
            return pl.read_parquet(cache_path)

        df = compute_fn()
        df.write_parquet(cache_path)
        # Save config alongside for debuggability
        (self.store_dir / f"{key}.json").write_text(json.dumps(config, indent=2))
        return df

    def _hash_config(self, config: dict) -> str:
        canonical = json.dumps(config, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

**Next level — column-level cache with merge**:

Instead of caching whole DataFrames, cache individual feature groups (lag features, rolling features, cross-sectional features separately). When you change rolling windows from `[5, 10]` to `[5, 10, 20]`, you only need to compute the `roll20_*` columns — the `roll5_*` and `roll10_*` columns are loaded from cache.

**Production feature stores** (Feast, Tecton, Hopsworks) add:
- Point-in-time joins: look up feature values as of a specific timestamp without future leakage
- Online serving: low-latency key-value lookup for inference (replaces `InferenceState`)
- Data quality monitoring: alert when features drift from historical distributions
- Feature lineage: track which features came from which computation

---

### 5. Better Ensembling

**Concept**: the current ensemble uses a fixed linear combination of model predictions (`Σ w_i · p_i`). More sophisticated ensembling can capture non-linear interactions between models, regime-dependent weighting, and uncertainty-aware combination.

**Current baseline** (already implemented):

```python
# OOF-optimised linear weights via Nelder-Mead
w* = argmin_{w ≥ 0, Σw=1} OOF_competition_score(Σ w_i · oof_i)
```

**Stacking (Level 2 model)**:

```python
# src/janestreet_forecasting/modeling/stacking.py
class StackingEnsemble:
    """
    Train a meta-model on OOF predictions from base models.

    The meta-model learns WHEN each base model is right, not just a fixed
    average. A base model that's good in high-volatility regimes and bad
    in trending regimes will get high meta-model weight in the former.
    """
    def fit(
        self,
        oof_preds: dict[str, np.ndarray],  # {"lgbm": [...], "xgb": [...]}
        y_true: np.ndarray,
        weights: np.ndarray,
        # Optionally include regime features so meta-model is regime-aware
        meta_features: np.ndarray | None = None,
    ):
        X_meta = np.column_stack(list(oof_preds.values()))
        if meta_features is not None:
            X_meta = np.column_stack([X_meta, meta_features])

        # Use a simple LightGBM meta-model — fast, handles non-linearity
        # CV the meta-model too to avoid overfitting on OOF
        self.meta_model = lgb.train(
            params={"objective": "regression", "num_leaves": 7, "verbosity": -1},
            train_set=lgb.Dataset(X_meta, label=y_true, weight=weights),
        )

    def predict(self, base_preds: dict[str, np.ndarray]) -> np.ndarray:
        X_meta = np.column_stack(list(base_preds.values()))
        return self.meta_model.predict(X_meta)
```

**Bayesian Model Combination**:

```python
# Each model's weight is a distribution, not a point estimate
# Uses the posterior uncertainty to down-weight models that are uncertain
from scipy.special import softmax

class BayesianModelCombiner:
    def fit(self, oof_preds, y_true):
        # Log-likelihood of each model
        log_likelihoods = np.array([
            -np.mean((pred - y_true) ** 2)
            for pred in oof_preds.values()
        ])
        self.weights = softmax(log_likelihoods)  # higher LL → higher weight

    def predict(self, preds):
        return sum(w * p for w, p in zip(self.weights, preds.values()))
```

**Snapshot ensembling**: train one model and save checkpoints at different epochs. Final prediction = average of checkpoints. This is free diversity — no need to train separate models. LightGBM supports this via `callbacks`:

```python
def _snapshot_callback(n_snapshots=5):
    """Save a model snapshot every N/total boosting rounds."""
    snapshots = []
    def callback(env):
        if env.iteration % (env.end_iteration // n_snapshots) == 0:
            snapshots.append(env.model.copy())
    callback.snapshots = snapshots
    return callback
```

**Multi-task learning as implicit ensemble**:

Train a single model with 9 output heads (one per responder). The shared backbone learns common structure; each head specialises. Predictions from the `responder_6` head may outperform a single-task model because the shared backbone is regularised by signal from the other 8 responders — many of which are correlated with `responder_6`.

```python
# LightGBM doesn't support multi-output natively — use scikit-multilearn
# or a neural network head
import torch
import torch.nn as nn

class MultiTaskHead(nn.Module):
    def __init__(self, backbone_dim: int, n_tasks: int = 9):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(backbone_dim, 1) for _ in range(n_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=1)
```

---

## Makefile Reference

```bash
make env            # pip install -e ".[dev]" + pre-commit install
make data-download  # kaggle competitions download (requires KAGGLE_KEY)
make prepare-data   # js-prepare → data/interim/
make build-features # js-features → data/processed/
make train-lgbm     # js-train --experiment configs/experiments/exp001_lgbm_baseline.yaml
make train-xgb      # js-train --experiment configs/experiments/exp002_xgb_baseline.yaml
make train-catboost # js-train --experiment configs/experiments/exp003_catboost_baseline.yaml
make train-ensemble # js-train --experiment configs/experiments/exp004_ensemble.yaml
make evaluate       # js-evaluate --experiment $EXP
make tune           # js-tune --experiment $EXP --n-trials 100
make infer-local    # js-infer serve-local --experiment $EXP
make notebook       # jupyter lab notebooks/
make mlflow-ui      # mlflow ui --port 5000
make test           # pytest tests/ -v
make test-cov       # pytest --cov=src --cov-report=html
make lint           # ruff check src/ tests/
make typecheck      # mypy src/
make check          # lint + typecheck + test
make clean          # remove artifacts/, reports/, __pycache__
```
