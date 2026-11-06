"""The only config parser: YAML composition, ``--set`` overrides, and conversion back to dicts."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from .experiment import ExperimentConfig
from .validation import ConfigError

T = TypeVar("T")


def _is_optional(annotation: Any) -> bool:
    return get_origin(annotation) in (Union, UnionType) and type(None) in get_args(annotation)


def _non_none_arg(annotation: Any) -> Any:
    return next(a for a in get_args(annotation) if a is not type(None))


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Convert a YAML or --set value to the annotated type.

    Args:
        value: The raw value from YAML or the command line.
        annotation: The dataclass field's type annotation.
        path: Dotted key path, for error messages.

    Returns:
        The value converted to the annotated type.

    Raises:
        ConfigError: If the value cannot be converted.
    """
    if _is_optional(annotation):
        if value is None or (isinstance(value, str) and value.lower() in ("none", "null", "")):
            return None
        annotation = _non_none_arg(annotation)

    if is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping")
        return _build(annotation, value, path)

    origin = get_origin(annotation)
    if origin is tuple:
        item_type, _ = get_args(annotation)
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list or comma-separated values")
        if item_type is str and any(not isinstance(v, str) for v in value):
            raise ConfigError(f"{path}: list entries must be strings")
        return tuple(_coerce(v, item_type, f"{path}[{i}]") for i, v in enumerate(value))

    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "yes", "1"):
            return True
        if isinstance(value, str) and value.lower() in ("false", "no", "0"):
            return False
        raise ConfigError(f"{path}: cannot read {value!r} as a boolean")

    for target in (int, float, str):
        if annotation is target:
            if annotation is int and isinstance(value, bool):
                raise ConfigError(f"{path}: expected int, got boolean {value!r}")
            if annotation is int and isinstance(value, float) and not value.is_integer():
                raise ConfigError(f"{path}: expected int, got fractional value {value!r}")
            try:
                return target(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{path}: cannot read {value!r} as {target.__name__}") from exc

    return value


def _build(cls: type, data: dict[str, Any], path: str = "") -> Any:
    """Recursively construct a frozen dataclass from a nested dict.

    Args:
        cls: The dataclass to construct.
        data: Mapping of field name to value.
        path: Dotted key path so far, for error messages.

    Returns:
        An instance of ``cls``.

    Raises:
        ConfigError: On an unknown key or a missing required field.
    """
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        prefix = f"{path}." if path else ""
        raise ConfigError(
            f"unknown config key(s): {', '.join(sorted(prefix + k for k in unknown))}. "
            f"Valid keys here: {', '.join(sorted(known))}"
        )

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        key_path = f"{path}.{f.name}" if path else f.name
        annotation = hints[f.name]
        if f.name in data:
            kwargs[f.name] = _coerce(data[f.name], annotation, key_path)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(
                f"missing required config key {key_path!r}. It has no default on purpose; "
                f"set it in a config file or pass --set {key_path}=<value>"
            )
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_tree(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load one YAML file and everything in its ``defaults:`` list, merged in order."""
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(p.name for p in (*seen, resolved))
        raise ConfigError(f"circular defaults: {chain}")
    if not resolved.exists():
        raise ConfigError(f"config file not found: {resolved}")

    raw = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{resolved.name}: top level must be a mapping")

    merged: dict[str, Any] = {}
    for stem in raw.pop("defaults", []) or []:
        parent = resolved.parent / f"{stem}.yaml"
        merged = _deep_merge(merged, _load_yaml_tree(parent, (*seen, resolved)))
    return _deep_merge(merged, raw)


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key.path=value`` strings to a nested dict.

    Args:
        data: The merged config dict.
        overrides: Strings of the form ``run.seed=1``. None or empty is a no-op.

    Returns:
        A new dict with the overrides applied.

    Raises:
        ConfigError: If an override is not of the form ``key=value``.
    """
    result = dict(data)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must be of the form key.path=value")
        key_path, value = item.split("=", 1)
        node = result
        parts = key_path.strip().split(".")
        for part in parts[:-1]:
            child = node.get(part)
            node[part] = dict(child) if isinstance(child, dict) else {}
            node = node[part]
        node[parts[-1]] = value.strip()
    return result


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    *,
    extra_paths: Sequence[str | Path] = (),
) -> ExperimentConfig:
    """Load, compose, override and validate a config.

    Args:
        path: Path to a YAML config. Its ``defaults:`` list is resolved relative to its own
            directory.
        overrides: ``key.path=value`` strings applied after composition.
        extra_paths: Further YAML configs, each composed like ``path`` and merged over it in
            order, before ``overrides``. The benchmark runner passes a topology's group file.

    Returns:
        A fully resolved, frozen ExperimentConfig.

    Raises:
        ConfigError: On a missing file, circular defaults, an unknown key, a missing required
            field, or a value that fails validation.
    """
    data = _load_yaml_tree(Path(path))
    for extra in extra_paths:
        data = _deep_merge(data, _load_yaml_tree(Path(extra)))
    return _build(ExperimentConfig, apply_overrides(data, overrides))


def load_group(path: str | Path, group: str, cls: type[T], overrides: list[str] | None = None) -> T:
    """Load and validate one top-level config group without building a whole run config.

    For tools that are not runs -- such as the backbone check -- and so have no ``run.owner`` to
    supply. Composition, overrides and validation are exactly those of :func:`load_config`.

    Args:
        path: Path to a YAML config. Its ``defaults:`` list is resolved as usual.
        group: Top-level key to load, e.g. ``"serving"``.
        cls: The dataclass that group builds into.
        overrides: ``key.path=value`` strings applied after composition.

    Returns:
        The validated group. A group absent from the files is built from its dataclass defaults.

    Raises:
        ConfigError: On a missing file, an unknown key, or a value that fails validation.
    """
    data = apply_overrides(_load_yaml_tree(Path(path)), overrides)
    section = data.get(group, {})
    if not isinstance(section, dict):
        raise ConfigError(f"{group}: expected a mapping")
    return _build(cls, section, group)


def to_dict(cfg: Any) -> Any:
    """Convert a config back to plain nested dicts, for writing ``config.resolved.yaml``.

    Args:
        cfg: A config dataclass instance, or any nested value inside one.

    Returns:
        Plain dicts, lists and scalars, safe for ``yaml.safe_dump``.
    """
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return {f.name: to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, (tuple, list)):
        return [to_dict(v) for v in cfg]
    if isinstance(cfg, Path):
        return str(cfg)
    return cfg
