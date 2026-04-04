"""
Global runtime settings, loaded from environment variables.

These are *runtime* settings (credentials, compute resources, flags), not
experiment configuration.  Experiment hyperparameters live in YAML configs.

Usage:
    from janestreet_forecasting.settings import settings
    print(settings.log_level)
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with sane defaults for local development."""

    model_config = SettingsConfigDict(
        env_prefix="JS_",       # env var JS_LOG_LEVEL overrides log_level
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Logging
    log_level: str = Field("INFO", description="Loguru log level")
    log_json: bool = Field(False, description="Emit JSON-structured logs (for prod/CI)")

    # Compute
    n_jobs: int = Field(-1, description="Parallelism. -1 = all cores.")
    device: str = Field("cpu", description="Training device: cpu | cuda | mps")
    seed: int = Field(42, description="Global random seed")

    # MLflow
    mlflow_tracking_uri: str = Field(
        "",
        description="MLflow tracking URI. Empty string = local file store.",
    )
    mlflow_experiment_name: str = Field(
        "janestreet-forecasting",
        description="MLflow experiment name",
    )

    # Data
    max_rows: int | None = Field(
        None,
        description="Cap rows loaded (useful for fast iteration). None = load all.",
    )

    # Kaggle
    kaggle_username: str = Field("", description="Kaggle username (for API downloads)")
    kaggle_key: str = Field("", description="Kaggle API key")


# Singleton — import this object throughout the codebase
settings = Settings()
