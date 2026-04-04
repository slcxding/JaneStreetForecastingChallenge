# =============================================================================
# Jane Street Forecasting — Makefile
# Common workflow commands. Run `make help` to see all targets.
# =============================================================================

PYTHON   := python
PIP      := pip
SRC      := src
TESTS    := tests
EXP      ?= configs/experiments/exp001_lgbm_baseline.yaml

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

.PHONY: install
install:  ## Install package + all dependencies (editable mode)
	$(PIP) install -e ".[dev]"

.PHONY: install-pre-commit
install-pre-commit:  ## Install pre-commit hooks
	pre-commit install

.PHONY: env
env: install install-pre-commit  ## Full environment setup (install + pre-commit)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

.PHONY: data-download
data-download:  ## Download competition data via Kaggle CLI (requires Kaggle API key)
	kaggle competitions download -c jane-street-real-time-market-data-forecasting -p data/raw/
	cd data/raw && unzip -o jane-street-real-time-market-data-forecasting.zip && rm *.zip

.PHONY: prepare-data
prepare-data:  ## Validate and prepare raw data → data/interim/
	js-prepare --config configs/data/default.yaml

.PHONY: build-features
build-features:  ## Build feature set → data/processed/
	js-features --config configs/features/default.yaml

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

.PHONY: train
train:  ## Train model defined in EXP config (default: LightGBM baseline)
	js-train --experiment $(EXP)

.PHONY: train-lgbm
train-lgbm:  ## Train LightGBM baseline
	js-train --experiment configs/experiments/exp001_lgbm_baseline.yaml

.PHONY: train-xgb
train-xgb:  ## Train XGBoost baseline
	js-train --experiment configs/experiments/exp002_xgb_baseline.yaml

.PHONY: train-catboost
train-catboost:  ## Train CatBoost baseline
	js-train --experiment configs/experiments/exp003_catboost_baseline.yaml

.PHONY: train-ensemble
train-ensemble:  ## Build weighted ensemble from existing model artifacts
	js-train --experiment configs/experiments/exp004_ensemble.yaml

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

.PHONY: evaluate
evaluate:  ## Run backtest evaluation on validation set
	js-evaluate --experiment $(EXP)

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

.PHONY: tune
tune:  ## Run Optuna hyperparameter search for EXP
	js-tune --experiment $(EXP) --n-trials 50

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

.PHONY: infer-local
infer-local:  ## Simulate local inference loop (mirrors Kaggle API)
	js-infer serve-local --experiment $(EXP) --data data/processed/test.parquet

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

.PHONY: lint
lint:  ## Run ruff linter
	ruff check $(SRC) $(TESTS)

.PHONY: format
format:  ## Auto-format with ruff
	ruff format $(SRC) $(TESTS)
	ruff check --fix $(SRC) $(TESTS)

.PHONY: typecheck
typecheck:  ## Run mypy type checker
	mypy $(SRC)

.PHONY: test
test:  ## Run test suite
	pytest $(TESTS) -v

.PHONY: test-cov
test-cov:  ## Run tests with coverage report
	pytest $(TESTS) --cov=$(SRC) --cov-report=term-missing --cov-report=html:reports/coverage

.PHONY: check
check: lint typecheck test  ## Run all quality checks

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

.PHONY: report
report:  ## Generate evaluation report for EXP
	js-evaluate --experiment $(EXP) --report reports/

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

.PHONY: clean
clean:  ## Remove Python build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ reports/coverage/

.PHONY: mlflow-ui
mlflow-ui:  ## Launch MLflow tracking UI (default port 5000)
	mlflow ui --backend-store-uri artifacts/mlruns

.PHONY: notebook
notebook:  ## Launch Jupyter notebook server
	jupyter notebook notebooks/

.PHONY: help
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'
