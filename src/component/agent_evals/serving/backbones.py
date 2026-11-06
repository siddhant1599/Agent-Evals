"""Backbone specifications, read from ``model_registry.csv``.

The registry file is the source of truth for which backbones exist. Code never hardcodes a model
name: a backbone is selected by its registry ``name``, and every serving detail -- repository,
pinned revision, quantization, context length -- comes from its row.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["BackboneSpec", "ModelRegistryError", "load_model_registry", "select_backbone"]

_REQUIRED_COLUMNS = ("name", "hf_path", "hf_revision", "gated", "quantization", "max_model_len")


class ModelRegistryError(ValueError):
    """Raised for a malformed registry, an unknown backbone, or a value not yet measured."""


@dataclass(frozen=True, slots=True)
class BackboneSpec:
    """One row of ``model_registry.csv``.

    Attributes:
        name: The registry key passed as ``--backbone``.
        hf_path: HuggingFace repository.
        hf_revision: Pinned 40-character commit SHA.
        gated: Whether the repository needs an accepted license and an access token.
        quantization: The vLLM quantization method.
        max_model_len: Serving context length, or None until measured for this backbone.
    """

    name: str
    hf_path: str
    hf_revision: str
    gated: bool
    quantization: str
    max_model_len: int | None

    def require_max_model_len(self) -> int:
        """Return the measured serving context length.

        Returns:
            The context length recorded in the registry.

        Raises:
            ModelRegistryError: If it has not been measured for this backbone yet.
        """
        if self.max_model_len is None:
            raise ModelRegistryError(
                f"{self.name}: max_model_len has not been measured yet. Measure the longest prompt "
                "plus the completion cap with this backbone's own tokenizer and chat template, and "
                "record it in model_registry.csv before serving it"
            )
        return self.max_model_len


def load_model_registry(path: Path) -> dict[str, BackboneSpec]:
    """Read every backbone from the registry file.

    Args:
        path: Location of ``model_registry.csv``.

    Returns:
        Backbone name mapped to its specification, in file order.

    Raises:
        ModelRegistryError: If the file is missing, lacks a required column, repeats a name, or
            holds an invalid value.
    """
    if not path.is_file():
        raise ModelRegistryError(f"model registry not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in _REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ModelRegistryError(f"{path.name} is missing columns: {missing}")
        specs: dict[str, BackboneSpec] = {}
        for line, row in enumerate(reader, start=2):
            spec = _parse_row(row, f"{path.name}:{line}")
            if spec.name in specs:
                raise ModelRegistryError(f"{path.name}:{line}: duplicate backbone {spec.name!r}")
            specs[spec.name] = spec
    return specs


def select_backbone(registry: Mapping[str, BackboneSpec], name: str) -> BackboneSpec:
    """Look up one backbone by its registry name.

    Args:
        registry: As returned by :func:`load_model_registry`.
        name: The requested backbone.

    Returns:
        Its specification.

    Raises:
        ModelRegistryError: If the name is unknown. The message lists every available name.
    """
    try:
        return registry[name]
    except KeyError:
        available = ", ".join(registry) or "<empty registry>"
        raise ModelRegistryError(f"unknown backbone {name!r}. Available: {available}") from None


def _parse_row(row: Mapping[str, str | None], where: str) -> BackboneSpec:
    values = {column: (row.get(column) or "").strip() for column in _REQUIRED_COLUMNS}
    for column in ("name", "hf_path", "hf_revision", "quantization"):
        if not values[column]:
            raise ModelRegistryError(f"{where}: {column} is empty")
    if not re.fullmatch(r"[0-9a-f]{40}", values["hf_revision"]):
        raise ModelRegistryError(f"{where}: hf_revision must be a 40-character commit SHA")
    if values["gated"] not in ("true", "false"):
        raise ModelRegistryError(f"{where}: gated must be true or false")
    length = values["max_model_len"]
    if length and (not length.isdigit() or int(length) < 1):
        raise ModelRegistryError(f"{where}: max_model_len must be a positive integer or empty")
    return BackboneSpec(
        name=values["name"],
        hf_path=values["hf_path"],
        hf_revision=values["hf_revision"],
        gated=values["gated"] == "true",
        quantization=values["quantization"],
        max_model_len=int(length) if length else None,
    )
