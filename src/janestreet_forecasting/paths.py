"""
Centralised path resolution for the project.

All paths are derived from PROJECT_ROOT so the package works regardless
of where you run it from. Import these constants instead of constructing
paths inline throughout the codebase.
"""

from pathlib import Path

# The project root is two levels up from this file (src/janestreet_forecasting/paths.py)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"

ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# MLflow tracking URI — local by default, overridable via env
MLFLOW_TRACKING_URI: str = f"file://{ARTIFACTS_DIR / 'mlruns'}"


def get_experiment_dir(experiment_id: str) -> Path:
    """Return the artifact directory for a specific experiment run."""
    return ARTIFACTS_DIR / experiment_id


def ensure_dirs() -> None:
    """Create all standard directories if they don't exist."""
    dirs = [
        RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR,
        ARTIFACTS_DIR, REPORTS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
