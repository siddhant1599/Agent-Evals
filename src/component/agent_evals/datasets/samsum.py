"""SAMSum's column mapping; raw loading and all dataset policy live in DatasetAdapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..registry import DATASETS
from .base import DatasetAdapter


@DATASETS.register("tier1_samsum")
class SAMSumAdapter(DatasetAdapter):
    """Tier 1 dialogue summarization, evaluation-only: no split is trainable (AGENTS.md §4.2).

    The dialogue, one ``Speaker: message`` turn per line, is both source and evidence; the
    summary becomes the reference unchanged.
    """

    TIER = 1
    TASK = "summarization"

    def _map_row(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "id": row["id"],
            "source": row["dialogue"],
            "evidence": row["dialogue"],
            "reference": row["summary"],
            "question": None,
        }
