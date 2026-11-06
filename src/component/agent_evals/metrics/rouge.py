"""ROUGE-L of each output against its reference summary (D-026), or its best human answer."""

from __future__ import annotations

from ..config import MetricsConfig
from ..registry import METRICS
from .base import Metric, reference_texts

__all__ = ["RougeL"]


@METRICS.register("rouge_l")
class RougeL(Metric):
    """ROUGE-L precision, recall and F1 from ``rouge-score``.

    With several human answers in ``Example.gold`` (ASQA has two), the output is scored against
    each and the scores of the answer with the higher F1 are kept; on a tie, the later answer's,
    as ALCE's ``compute_rouge`` does (D-123). Otherwise it is scored against ``reference``.

    Output range: every key lies in [0, 1].

    Args:
        cfg: Metric configuration; reads ``cfg.rouge_l``.
    """

    KEYS = ("rouge_l_precision", "rouge_l_recall", "rouge_l_f1")

    def __init__(self, cfg: MetricsConfig) -> None:
        super().__init__(cfg)
        from rouge_score import rouge_scorer

        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=cfg.rouge_l.use_stemmer)

    def _score_examples(
        self,
        predictions: list[str],
        references: list[str],
        evidence: list[str],
        gold: list[str | None],
    ) -> list[dict[str, float]]:
        rows = []
        for prediction, reference, example_gold in zip(predictions, references, gold, strict=True):
            results = [
                self._scorer.score(text, prediction)["rougeL"]
                for text in reference_texts(reference, example_gold)
            ]
            # max() keeps the first maximum, so scanning in reverse keeps the later answer on a tie.
            result = max(reversed(results), key=lambda score: score.fmeasure)
            rows.append(
                {
                    "rouge_l_precision": float(result.precision),
                    "rouge_l_recall": float(result.recall),
                    "rouge_l_f1": float(result.fmeasure),
                }
            )
        return rows
