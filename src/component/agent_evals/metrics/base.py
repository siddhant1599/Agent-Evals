"""The metric interface: resident scorer models load once, and every example gets its own scores.

Scores are kept per example because the paired design (D-002) compares topologies article by
article and bootstrapped confidence intervals resample examples. ``compute`` keeps the aggregate
interface as the mean over examples.

A blank generated output is never scored. Factual consistency is undefined for text that does not
exist, and any number a scorer returned for it would be an artefact of that scorer's internals. The
caller records a blank output as a generation failure instead.

``gold`` carries annotations a single reference string cannot hold (``Example.gold``, D-126). It is
optional: a metric that does not read it ignores it, and a metric that needs it says so with
``NEEDS_GOLD``.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from ..config import MetricsConfig

__all__ = ["Metric", "MetricError", "reference_texts"]


class MetricError(RuntimeError):
    """Raised for invalid metric input, or a score the result contract cannot accept."""


class Metric(ABC):
    """Base for every metric.

    Subclasses load resident models in ``__init__`` and implement ``_score_examples``. They declare
    the keys they emit in ``KEYS``; keys are stable forever once a run has used them.

    Args:
        cfg: Metric configuration. Parameters bind once, here.
    """

    KEYS: ClassVar[tuple[str, ...]] = ()
    #: True for a metric that cannot score without ``gold`` on every example.
    NEEDS_GOLD: ClassVar[bool] = False
    #: True for a metric that reads citation markers such as ``[1]``. On a dataset whose outputs
    #: carry citations, every other metric scores the output with the markers removed (D-126).
    READS_CITATIONS: ClassVar[bool] = False

    def __init__(self, cfg: MetricsConfig) -> None:
        self._device = cfg.device

    def score(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        evidence: Sequence[str],
        gold: Sequence[str | None] | None = None,
    ) -> list[dict[str, float]]:
        """Score every example.

        Args:
            predictions: Generated outputs. None may be blank.
            references: Gold references, aligned with ``predictions``.
            evidence: Grounding text each output is checked against, aligned with ``predictions``.
            gold: Each example's ``Example.gold`` (a JSON object string, or None), aligned with
                ``predictions``. None means no example has any.

        Returns:
            One flat dict per example holding exactly ``KEYS``, every value a finite float. Each
            subclass documents its output range.

        Raises:
            MetricError: On misaligned or non-string inputs, a blank prediction, missing gold for
                a metric that needs it, or a bad score.
        """
        golds = [None] * len(predictions) if gold is None else list(gold)
        self._check_inputs(predictions, references, evidence)
        self._check_gold(golds, len(predictions))
        rows = self._score_examples(list(predictions), list(references), list(evidence), golds)
        self._check_outputs(rows, len(predictions))
        return rows

    def compute(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        evidence: Sequence[str],
        gold: Sequence[str | None] | None = None,
    ) -> dict[str, float]:
        """Mean of each key over the examples.

        Args:
            predictions: Generated outputs.
            references: Gold references.
            evidence: Grounding text.
            gold: Each example's ``Example.gold``, or None.

        Returns:
            Flat dict of the mean per key.

        Raises:
            MetricError: As for :meth:`score`.
        """
        rows = self.score(predictions, references, evidence, gold)
        return {key: math.fsum(row[key] for row in rows) / len(rows) for key in self.KEYS}

    @abstractmethod
    def _score_examples(
        self,
        predictions: list[str],
        references: list[str],
        evidence: list[str],
        gold: list[str | None],
    ) -> list[dict[str, float]]:
        """Score validated inputs; one dict per example, in order. ``gold`` may hold None."""

    def _check_gold(self, gold: list[str | None], count: int) -> None:
        if len(gold) != count:
            raise MetricError(f"gold is misaligned: {count} predictions, {len(gold)} gold")
        if any(value is not None and not isinstance(value, str) for value in gold):
            raise MetricError("gold must hold JSON strings or None")
        if self.NEEDS_GOLD and any(value is None for value in gold):
            raise MetricError(f"{type(self).__name__} needs gold annotations on every example")

    def _check_inputs(
        self, predictions: Sequence[str], references: Sequence[str], evidence: Sequence[str]
    ) -> None:
        count = len(predictions)
        if count == 0:
            raise MetricError("no examples to score")
        if len(references) != count or len(evidence) != count:
            raise MetricError(
                f"inputs are misaligned: {count} predictions, {len(references)} references, "
                f"{len(evidence)} evidence"
            )
        for name, values in (
            ("predictions", predictions),
            ("references", references),
            ("evidence", evidence),
        ):
            if any(not isinstance(value, str) for value in values):
                raise MetricError(f"{name} must all be strings")
        blank = [i for i, value in enumerate(predictions) if not value.strip()]
        if blank:
            raise MetricError(
                f"{len(blank)} blank prediction(s), first at index {blank[0]}; blank outputs are "
                "recorded as generation failures, never scored"
            )

    def _check_outputs(self, rows: list[dict[str, float]], count: int) -> None:
        if len(rows) != count:
            raise MetricError(f"{type(self).__name__} gave {len(rows)} rows for {count} examples")
        expected = set(self.KEYS)
        for index, row in enumerate(rows):
            if set(row) != expected:
                raise MetricError(
                    f"{type(self).__name__} row {index} has keys {sorted(row)}, "
                    f"expected {sorted(expected)}"
                )
            for key, value in row.items():
                if not isinstance(value, float) or not math.isfinite(value):
                    raise MetricError(f"{type(self).__name__} row {index} {key} is {value!r}")


def reference_texts(reference: str, gold: str | None) -> list[str]:
    """The texts a reference-based metric scores against (D-123).

    Args:
        reference: The example's single reference.
        gold: The example's ``Example.gold``, or None.

    Returns:
        Every human answer in ``gold``'s ``long_answers`` when it lists any (ASQA has two);
        otherwise the single reference.

    Raises:
        MetricError: If ``long_answers`` is present but not a list of non-empty strings.
    """
    if gold is None:
        return [reference]
    answers = json.loads(gold).get("long_answers")
    if answers is None:
        return [reference]
    if not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
        raise MetricError("gold long_answers must be a non-empty list of non-empty strings")
    return list(answers)
