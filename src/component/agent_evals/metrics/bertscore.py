"""BERTScore of each output against its reference summary (D-019), or its best human answer."""

from __future__ import annotations

from ..config import MetricsConfig
from ..logging_utils import get_logger
from ..registry import METRICS
from .base import Metric, MetricError, reference_texts

__all__ = ["BertScore"]

logger = get_logger(__name__)

# ``bert-score`` picks special loaders when the model path contains these (``utils.get_model``).
_SPECIAL_LOADER_MARKERS = ("scibert", "t5")


@METRICS.register("bertscore")
class BertScore(Metric):
    """BERTScore precision, recall and F1 from ``bert-score``, no IDF and no rescaling.

    The pinned snapshot is passed to the library by local path, so the revision holds.

    With several human answers in ``Example.gold`` (ASQA has two), the output is scored against
    each and the scores of the answer with the higher F1 are kept, the same rule as ROUGE-L
    (D-123). The library's own multi-reference mode is not used: it takes the maximum of
    precision, recall and F1 separately (``BERTScorer.score``, ``max(dim=0)`` over the group), so
    the three could come from different answers.

    Output range: cosine-similarity based, in [-1, 1]; without baseline rescaling, values for
    English text sit near the top of that range.

    Args:
        cfg: Metric configuration; reads ``cfg.bertscore`` and ``cfg.device``.
    """

    KEYS = ("bertscore_precision", "bertscore_recall", "bertscore_f1")

    def __init__(self, cfg: MetricsConfig) -> None:
        super().__init__(cfg)
        from bert_score import BERTScorer
        from huggingface_hub import snapshot_download

        path = snapshot_download(cfg.bertscore.model, revision=cfg.bertscore.revision)
        if any(marker in path for marker in _SPECIAL_LOADER_MARKERS):
            raise MetricError(f"snapshot path {path} would select a special bert-score loader")
        self._scorer = BERTScorer(
            model_type=path, num_layers=cfg.bertscore.num_layers, device=self._device
        )
        logger.info(
            "loaded BERTScore (%s, layer %d)", cfg.bertscore.model, cfg.bertscore.num_layers
        )

    def _score_examples(
        self,
        predictions: list[str],
        references: list[str],
        evidence: list[str],
        gold: list[str | None],
    ) -> list[dict[str, float]]:
        groups = [
            reference_texts(reference, example_gold)
            for reference, example_gold in zip(references, gold, strict=True)
        ]
        candidates = [p for p, texts in zip(predictions, groups, strict=True) for _ in texts]
        targets = [text for texts in groups for text in texts]
        precision, recall, f1 = self._scorer.score(candidates, targets)
        scores = list(zip(precision.tolist(), recall.tolist(), f1.tolist(), strict=True))
        rows, start = [], 0
        for texts in groups:
            pairs = scores[start : start + len(texts)]
            start += len(texts)
            # max() keeps the first maximum, so scanning in reverse keeps the later answer on a tie.
            p, r, f = max(reversed(pairs), key=lambda pair: pair[2])
            rows.append(
                {
                    "bertscore_precision": float(p),
                    "bertscore_recall": float(r),
                    "bertscore_f1": float(f),
                }
            )
        return rows
