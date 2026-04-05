# Architecture & Design Decisions

Technical deep-dive into why the codebase is structured as it is, and how to extend it.

---

## Data

### Why Polars?

Polars uses Apache Arrow's columnar memory layout and a lazy query planner that pushes filters down to the Parquet reader. On this dataset (47M rows, 92 columns), filtering to a date range reads only the relevant row groups — not the full file. The single copy to NumPy at the model training boundary is worth the upstream speed.

### Why lazy loading?

All loaders return `LazyFrame`. Callers call `.collect()` only when they need the data. This lets the query planner fuse operations across chained transforms before touching disk.

### Data files

| File | Rows | Key columns |
|---|---|---|
| `train.parquet` (10 Hive partitions) | 47.1M | `date_id` 0–1689, `time_id` 0–848, `symbol_id` 0–19, 79 features, 9 responders |
| `lag.parquet` (Hive by `date_id`) | — | `responder_*_lag_1` — previous calendar day's responders |
| `test.parquet` (Hive by `date_id`) | — | Same as train minus responders; adds `row_id`, `is_scored` |
| `features.csv` | 79 rows | Anonymous boolean tags (`tag_0`..`tag_16`) grouping related features |
| `responders.csv` | 9 rows | Anonymous boolean tags (`tag_0`..`tag_4`) grouping responders |

The lag file is separate because the Kaggle API provides it separately at inference time. Keeping training data structured the same way makes the join explicit and prevents accidental leakage.

---

## Features

### Transformer protocol

Transformers implement a `Protocol` (structural typing), not inheritance from sklearn. This keeps feature code clean — no `get_params`/`set_params`/`fit_transform` boilerplate — and avoids coupling Polars DataFrames to sklearn's NumPy contract.

```python
class MyTransformer:
    def fit(self, df: pl.DataFrame, **kwargs) -> "MyTransformer": ...
    def transform(self, df: pl.DataFrame) -> pl.DataFrame: ...

    @property
    def feature_names_out(self) -> list[str]: ...
```

### Pipeline serialisation

`FeaturePipeline` saves all fitted transformers to a pickle file. At inference time this file is loaded to reproduce the exact same features. Critically, `feature_cols` is read from the saved pipeline — not derived by excluding index/target columns from the DataFrame. Exclusion-based selection would silently include any intermediate columns leaked by a buggy transformer.

### Rolling feature correctness

`RollingTransformer` uses Polars `.rolling_mean(...).over(symbol_id)` — this includes the current row in the window. `InferenceState.compute_features()` matches this by combining the ring buffer (past rows) with the current batch's values before computing rolling stats. This parity is enforced by `tests/test_integration.py::test_inference_state_matches_offline_rolling`.

`"skew"` is deliberately absent from supported stats. The naive implementation computed `x - rolling_mean` (a deviation, not skewness). A mislabeled feature causes silent model degradation.

---

## Validation strategy

### Why standard k-fold is invalid

1. **Rolling-feature overlap**: a 10-period rolling mean at time t uses data from t−9..t. If any of those steps are in train, information leaks through the feature into val.
2. **Autocorrelation**: adjacent rows share information. Random splitting creates leakage.
3. **Distribution shift**: models trained on shuffled data learn average behaviour, not conditional predictions given recent history.

### PurgedGroupKFold

Groups rows by `date_id`. For each fold:

- **Purge** (`purge_days=5`): exclude the N dates immediately before the test window from training. A 5-step rolling mean at the first test date reaches back 5 steps — those are in the purge zone.
- **Embargo** (`embargo_days=10`): exclude dates immediately after the test window from training in future folds. Market autocorrelation means test-period events influence subsequent returns.
- **`forward_only=True`** (default): each fold trains only on dates before the test window. Without this, fold 0 could test on early dates while training on later (future) ones.

Reference: *Advances in Financial Machine Learning*, López de Prado (2018), Ch. 7.

---

## The inference problem

The Kaggle API sends one `time_id` at a time. Rolling features cannot be computed from a single row — you need history. `InferenceState` solves this with a per-symbol `collections.deque(maxlen=buffer_size)`.

```
# Correct order — must never be reversed
enriched  = state.compute_features(batch)  # use history to build features
preds     = model.predict(enriched)         # predict
state.update(batch)                         # then add current row to history
```

Reversing this makes features at step t include step t's data — equivalent to target leakage.

---

## Ensemble

`WeightedAverageEnsemble` optimises weights on OOF predictions via Nelder-Mead:

```
w* = argmin_{w ≥ 0, Σw=1}  −competition_score(Σ_i w_i · oof_i)
```

OOF predictions are used rather than a held-out validation set because they cover the full training period without wasting data. A stacking meta-model (next extension step) can learn richer combinations when the base models have systematically different failure modes.

---

## How to extend

### Deep learning sequence models

GBDTs see each row independently even with rolling features. An LSTM sees the last T steps and learns temporal patterns rolling statistics miss.

The key constraint: the Kaggle API sends one time step at a time. Your sequence model must maintain **hidden state** across calls, just like `InferenceState`.

```python
class SymbolLSTM(nn.Module):
    def forward(self, x: Tensor, state=None):
        # x: (n_symbols, seq_len=1, n_features)
        out, new_state = self.lstm(x, state)
        return self.head(out[:, -1, :]), new_state  # return state for next step

class LSTMInferenceState:
    def __init__(self, model):
        self._states: dict[int, tuple] = {}  # symbol_id → (h, c)

    def predict(self, batch):
        for symbol in batch["symbol_id"]:
            x = to_tensor(batch, symbol)
            pred, self._states[symbol] = self.model(x, self._states.get(symbol))
```

Architecture options: LSTM/GRU (simple, fast), TFT (interpretable attention), PatchTST (ViT for time-series).

### Online learning

Markets change regimes. A model trained on 2018–2022 data may underperform in 2024.

**Option A — sliding window retraining** (already supported by `load_train(date_range=...)`):
retrain every N days using only the last K days of data.

**Option B — EWM-weighted samples** (recommended):
```python
decay = 0.995
df = df.with_columns(
    (pl.lit(decay) ** (current_date - pl.col("date_id"))).alias("sample_weight")
)
```
Pass `sample_weight` as the LightGBM weight column — recent data is valued more without discarding older data entirely.

**Option C — online gradient boosting** via the `river` library for truly incremental updates.

### Regime detection

```python
from sklearn.mixture import GaussianMixture

class RegimeDetector:
    """Cluster market days by cross-sectional feature distribution."""
    def fit(self, df: pl.DataFrame) -> "RegimeDetector":
        daily_stats = df.group_by("date_id").agg([
            pl.col(c).mean().alias(f"{c}_mean") for c in S.FEATURE_COLS[:10]
        ] + [
            pl.col(c).std().alias(f"{c}_std") for c in S.FEATURE_COLS[:10]
        ]).sort("date_id").drop("date_id").to_numpy()
        self.gmm = GaussianMixture(n_components=3).fit(daily_stats)
        return self
```

Use regime labels as features, routing keys (train separate models per regime), or to upweight in-regime training samples.

HMMs (`hmmlearn`) are more principled: they model regime transitions probabilistically and give you a probability of being in each regime at each step.

### Feature store

As you iterate on features, you recompute the same lag/rolling features repeatedly. A content-addressed cache cuts iteration time from hours to minutes:

```python
class FeatureStore:
    def get_or_compute(self, config: dict, compute_fn) -> pl.DataFrame:
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        cache_path = self.store_dir / f"{key}.parquet"
        if cache_path.exists():
            return pl.read_parquet(cache_path)
        df = compute_fn()
        df.write_parquet(cache_path)
        return df
```

Cache at the feature-group level (lag features, rolling features separately) so changing rolling windows only recomputes the new window — not all rolling features.

### Better ensembling

**Stacking**: train a LightGBM meta-model on OOF predictions from base models. The meta-model learns when each base model is right, not a fixed average.

```python
X_meta = np.column_stack([oof_lgbm, oof_xgb, oof_catboost])
meta_model = lgb.train(
    {"objective": "regression", "num_leaves": 7},
    lgb.Dataset(X_meta, label=y_oof, weight=w_oof),
)
```

**Snapshot ensembling**: save LightGBM checkpoints at different boosting rounds and average them. Free diversity — no extra training time.

**Multi-task learning**: train a neural network with 9 output heads (one per responder). The shared backbone is regularised by all 9 signals; predictions from the `responder_6` head benefit from related return series.
