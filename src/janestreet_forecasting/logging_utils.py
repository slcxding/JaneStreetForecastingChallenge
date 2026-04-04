"""
Logging configuration via Loguru.

Call `setup_logging()` once at the start of any CLI entrypoint.
The rest of the codebase just does `from loguru import logger`.

Design choice: Loguru over stdlib logging because:
  - Zero boilerplate — no handler/formatter/level dance
  - Structured JSON mode is a one-liner
  - Exception tracing is excellent out of the box
"""

import sys

from loguru import logger

from janestreet_forecasting.settings import settings


def setup_logging(level: str | None = None, json: bool | None = None) -> None:
    """
    Configure the global Loguru logger.

    Args:
        level: Override JS_LOG_LEVEL from settings.
        json:  Override JS_LOG_JSON from settings. Set True in CI/production.
    """
    effective_level = level or settings.log_level
    emit_json = json if json is not None else settings.log_json

    logger.remove()  # Remove Loguru's default handler

    if emit_json:
        # Machine-readable — useful for log aggregators (Datadog, CloudWatch, etc.)
        logger.add(
            sys.stderr,
            level=effective_level,
            serialize=True,
        )
    else:
        # Human-readable with colour
        logger.add(
            sys.stderr,
            level=effective_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
                "<level>{message}</level>"
            ),
            colorize=True,
        )


def get_logger(name: str):  # type: ignore[return]
    """Return a logger bound to the given module name."""
    return logger.bind(module=name)
