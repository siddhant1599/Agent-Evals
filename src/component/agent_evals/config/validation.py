"""The config error type and the field checks shared by several config groups."""

from __future__ import annotations

import re
from typing import Any


class ConfigError(ValueError):
    """Raised for an unknown key, a missing required field, or a value out of range."""


_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _check_text(group: str, obj: Any, names: tuple[str, ...]) -> None:
    for name in names:
        value = getattr(obj, name)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{group}.{name} must be a non-empty string")


def _check_digest(group: str, obj: Any, names: tuple[str, ...], pattern: re.Pattern[str]) -> None:
    for name in names:
        value = getattr(obj, name)
        if not isinstance(value, str) or not pattern.fullmatch(value):
            raise ConfigError(
                f"{group}.{name} must be a lowercase hex digest of the pinned artifact"
            )


def _check_positive(group: str, obj: Any, names: tuple[str, ...]) -> None:
    for name in names:
        value = getattr(obj, name)
        if type(value) is not int or value < 1:
            raise ConfigError(f"{group}.{name} must be an integer >= 1, got {value!r}")
