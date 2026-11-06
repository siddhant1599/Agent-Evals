"""FactCC through the HuggingFace conversion of the original checkpoint (D-025, D-029, D-030)."""

from __future__ import annotations

from ..config import MetricsConfig
from ..logging_utils import get_logger
from ..registry import METRICS
from .base import Metric, MetricError

__all__ = ["FactCC"]

logger = get_logger(__name__)

# The checkpoint's label for a consistent summary (config ``id2label``).
_CONSISTENT_LABEL = "CORRECT"


@METRICS.register("factcc")
class FactCC(Metric):
    """FactCC probability that each output is consistent with its article.

    Pairs are encoded as the model card does: padded to ``max_input_tokens``, article side
    truncated, segment ids requested explicitly (D-029).

    Output range: [0, 1].

    Args:
        cfg: Metric configuration; reads ``cfg.factcc`` and ``cfg.device``.
    """

    KEYS = ("factcc",)

    def __init__(self, cfg: MetricsConfig) -> None:
        super().__init__(cfg)
        import torch
        from transformers import BertForSequenceClassification, BertTokenizer

        self._cfg = cfg.factcc
        self._tokenizer = BertTokenizer.from_pretrained(
            self._cfg.model, revision=self._cfg.revision
        )
        self._model = BertForSequenceClassification.from_pretrained(
            self._cfg.model, revision=self._cfg.revision, dtype=torch.float32
        )
        self._model.eval().to(self._device)
        label2id = self._model.config.label2id
        if _CONSISTENT_LABEL not in label2id:
            raise MetricError(f"FactCC labels {sorted(label2id)} lack {_CONSISTENT_LABEL}")
        self._consistent_index = int(label2id[_CONSISTENT_LABEL])
        logger.info("loaded FactCC (%s) on %s", self._cfg.model, self._device)

    def pair_scores(self, documents: list[str], summaries: list[str]) -> list[float]:
        """P(CORRECT) for every (document, summary) pair.

        Args:
            documents: Articles.
            summaries: Summaries, aligned with ``documents``.

        Returns:
            One probability per pair.
        """
        import torch

        scores: list[float] = []
        for start in range(0, len(documents), self._cfg.batch_size):
            stop = start + self._cfg.batch_size
            tokens = self._tokenizer(
                documents[start:stop],
                summaries[start:stop],
                max_length=self._cfg.max_input_tokens,
                padding="max_length",
                truncation="only_first",
                return_tensors="pt",
                return_token_type_ids=True,
            )
            with torch.inference_mode():
                logits = self._model(**{k: v.to(self._device) for k, v in tokens.items()}).logits
            probabilities = torch.softmax(logits, dim=-1)[:, self._consistent_index]
            scores.extend(probabilities.float().cpu().tolist())
        return scores

    def _score_examples(
        self,
        predictions: list[str],
        references: list[str],
        evidence: list[str],
        gold: list[str | None],
    ) -> list[dict[str, float]]:
        return [{"factcc": float(score)} for score in self.pair_scores(evidence, predictions)]
