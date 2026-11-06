"""CNN/DailyMail's column mapping; raw loading and all dataset policy live in DatasetAdapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..registry import DATASETS
from .base import DatasetAdapter


@DATASETS.register("tier1_cnndm")
class CnnDailyMailAdapter(DatasetAdapter):
    """Tier 1 summarization, evaluation-only: no split is trainable (AGENTS.md §4.2, D-035).

    Highlights become the reference unchanged, one highlight per line (D-038).
    """

    TIER = 1
    TASK = "summarization"

    def _map_row(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "id": row["id"],
            "source": row["article"],
            "evidence": row["article"],
            "reference": row["highlights"],
            "question": None,
        }
