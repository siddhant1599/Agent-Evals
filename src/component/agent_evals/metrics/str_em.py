"""Exact-match recall on ASQA: the share of sub-questions whose answer appears in the output.

ALCE's ``str_em`` (Gao et al. 2023, §3.2; ``eval.py``, ``compute_str_em`` and ``exact_presence``,
commit ``246c476``), following Stelmakh et al. 2022: a sub-question counts as answered when any of
its accepted short answers is a substring of the output, both normalized as in SQuAD (lowercase;
punctuation, articles and extra whitespace removed). The runner removes citation markers first
(D-126), as ALCE does. ALCE reports percentages; this metric reports the same value in [0, 1].
"""

from __future__ import annotations

import json
import re
import string

from ..registry import METRICS
from .base import Metric, MetricError

__all__ = ["StrEM", "normalize_answer"]

_PUNCTUATION = frozenset(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    """ALCE's ``normalize_answer`` (``utils.py``), itself SQuAD's, unchanged.

    Args:
        text: An output or an accepted answer.

    Returns:
        Lowercased, without punctuation or the words "a", "an", "the", whitespace collapsed.
    """
    lowered = text.lower()
    without_punctuation = "".join(ch for ch in lowered if ch not in _PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", without_punctuation).split())


@METRICS.register("str_em")
class StrEM(Metric):
    """Exact-match recall of ASQA's sub-question answers (D-120).

    Reads ``short_answers`` from ``Example.gold``: one list of accepted answers per sub-question.

    Output range: ``str_em`` in [0, 1], the fraction of sub-questions answered.
    """

    KEYS = ("str_em",)
    NEEDS_GOLD = True

    def _score_examples(
        self,
        predictions: list[str],
        references: list[str],
        evidence: list[str],
        gold: list[str | None],
    ) -> list[dict[str, float]]:
        rows = []
        for prediction, example_gold in zip(predictions, gold, strict=True):
            sub_questions = json.loads(example_gold).get("short_answers")
            if not sub_questions:
                raise MetricError("str_em needs gold short_answers for at least one sub-question")
            output = normalize_answer(prediction)
            answered = [
                any(normalize_answer(answer) in output for answer in accepted)
                for accepted in sub_questions
            ]
            rows.append({"str_em": sum(answered) / len(answered)})
        return rows
