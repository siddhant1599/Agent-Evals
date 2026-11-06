"""Load secrets and instance settings from a local ``.env`` file into the environment.

Only secrets, instance paths and the vLLM endpoint URL belong in the environment -- never an
experimental parameter, which lives in ``src/configs/``. Values are never logged, never placed in
an error message, and never written to a manifest: the only thing this module reports is the
*names* of the variables it set.

Standard library only. A ``.env`` file is ``KEY=VALUE`` lines; blank lines and ``#`` comments are
skipped, an optional ``export`` prefix is accepted, and matching surrounding quotes are stripped.
"""

from __future__ import annotations

import os
from pathlib import Path

from .logging_utils import get_logger

__all__ = ["EnvFileError", "load_env_file"]

logger = get_logger(__name__)


class EnvFileError(ValueError):
    """Raised for a malformed ``.env`` line. The message names the line, never its value."""


def load_env_file(path: Path, override: bool = False) -> tuple[str, ...]:
    """Set environment variables from a ``.env`` file.

    Args:
        path: The file to read. A missing file is not an error: nothing is set.
        override: Replace variables already present in the environment. Off by default, so a
            value exported in the shell wins over the file.

    Returns:
        The names of the variables that were set, in file order. Keys whose value is empty are
        skipped, so an unfilled placeholder such as ``HF_TOKEN=`` leaves the variable unset.

    Raises:
        EnvFileError: If a non-comment line is not ``KEY=VALUE`` with a valid identifier key.
    """
    if not path.is_file():
        return ()
    set_keys: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parsed = _parse_line(raw, path.name, lineno)
        if parsed is None:
            continue
        key, value = parsed
        if not value or (key in os.environ and not override):
            continue
        os.environ[key] = value
        set_keys.append(key)
    logger.info("loaded %s variable(s) from %s: %s", len(set_keys), path.name, ", ".join(set_keys))
    return tuple(set_keys)


def _parse_line(raw: str, filename: str, lineno: int) -> tuple[str, str] | None:
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    if "=" not in line:
        raise EnvFileError(f"{filename}:{lineno}: expected KEY=VALUE")
    key, value = (part.strip() for part in line.split("=", 1))
    if not key.isidentifier():
        raise EnvFileError(f"{filename}:{lineno}: invalid variable name")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return key, value
