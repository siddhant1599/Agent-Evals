"""XSum's column mapping; raw loading and all dataset policy live in DatasetAdapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..registry import DATASETS
from .base import DatasetAdapter


@DATASETS.register("tier1_xsum")
class XSumAdapter(DatasetAdapter):
    """Tier 1 summarization, with XSum's training split eligible for T5 training (D-047)."""

    TIER = 1
    TASK = "summarization"
    TRAINABLE_SPLITS = frozenset({"train"})

    def _map_row(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "id": row["id"],
            "source": row["document"],
            "evidence": row["document"],
            "reference": row["summary"],
            "question": None,
        }
