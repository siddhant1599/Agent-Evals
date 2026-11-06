"""The only logger factory in the package.

``print()`` is never used in the package. Log lines are for humans; structured numbers go to
``metrics.csv`` through the writer in ``run.py`` so that ``analysis/`` never has to parse prose.

Handlers are attached to the package's root logger exactly once. Module loggers are children of
it, so ``get_logger(__name__)`` works before or after configuration.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import LoggingConfig

__all__ = ["get_logger", "configure_logging", "reset_logging", "PACKAGE_LOGGER"]

#: Root logger name for the package. Every module logger is a child of this.
PACKAGE_LOGGER = "agent_evals"

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a module.

    Args:
        name: Normally ``__name__``.

    Returns:
        A logger under the package root, so it inherits the configured handlers.
    """
    if name == PACKAGE_LOGGER or name.startswith(f"{PACKAGE_LOGGER}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER}.{name}")


def configure_logging(cfg: LoggingConfig, log_file: Path | None = None) -> logging.Logger:
    """Attach console and file handlers to the package logger.

    Calling this twice replaces the handlers rather than doubling them, so a second run in the
    same process does not write every line twice.

    Args:
        cfg: Levels for each destination.
        log_file: Where to write ``run.log``. None means console only, which is what the test
            suite uses.

    Returns:
        The configured package root logger.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    reset_logging()

    console_level = getattr(logging, cfg.console_level.upper())
    file_level = getattr(logging, cfg.file_level.upper())
    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def reset_logging() -> None:
    """Remove and close every handler on the package logger.

    Used between runs in one process, and between tests. Closing matters on Windows, where an
    open file handle blocks the temporary directory from being removed.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
