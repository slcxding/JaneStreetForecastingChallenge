"""
Jane Street Real-Time Market Data Forecasting
=============================================
Research codebase for Kaggle competition modelling.

Package layout:
  data/       — loaders, schemas, splits
  features/   — lag, rolling, cross-sectional transformers
  modeling/   — train scripts for LightGBM / XGBoost / CatBoost
  evaluation/ — backtest, diagnostics, reports
  inference/  — real-time state management and local simulator
  tuning/     — Optuna hyperparameter search
  cli/        — Typer entrypoints (exposed as js-* commands)
"""

__version__ = "0.1.0"
