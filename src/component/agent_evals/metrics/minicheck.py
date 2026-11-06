"""MiniCheck-Flan-T5-Large through the authors' package (D-024, D-028).

The package (``minicheck`` at commit ``b58b9fa69acbd1015ec970fa65dd752413a053d2``; Tang et al. 2024,
*MiniCheck*; https://github.com/Liyan06/MiniCheck) supplies chunking, prompt construction and
scoring unchanged. Only its loader is replaced, so the pinned snapshot is read onto one device
without ``accelerate``.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import MetricsConfig, MiniCheckConfig
from ..logging_utils import get_logger
from ..registry import METRICS
from .base import Metric, MetricError

__all__ = ["MiniCheck", "sentence_mean"]

logger = get_logger(__name__)

# The package's internal selector for its Flan-T5 code paths (chunking by words, the
# ``predict:`` prompt and the two label logits). It names a code path, not a checkpoint.
_PACKAGE_MODEL_NAME = "flan-t5-large"


def sentence_mean(text: str, support: Callable[[str], float]) -> float:
    """D-024's output score: each sentence checked as its own claim, then the mean.

    Args:
        text: Generated output.
        support: Returns the support probability of one claim.

    Returns:
        Mean support over the output's sentences; in [0, 1].

    Raises:
        MetricError: If the text has no sentences.
    """
    import nltk

    sentences = nltk.tokenize.sent_tokenize(text)
    if not sentences:
        raise MetricError("MiniCheck needs an output with at least one sentence")
    return sum(support(sentence) for sentence in sentences) / len(sentences)


def _load_inferencer(cfg: MiniCheckConfig, device: str) -> object:
    import torch
    from minicheck.inference import Inferencer
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    class PinnedInferencer(Inferencer):
        """The package's inferencer, loading the pinned snapshot onto ``device``."""

        def __init__(self) -> None:
            self.model_name = _PACKAGE_MODEL_NAME
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                cfg.model, revision=cfg.revision, dtype=torch.float32
            ).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
            self.max_model_len = cfg.max_input_tokens
            self.model.eval()
            self.batch_size = cfg.batch_size
            self.softmax = torch.nn.Softmax(dim=-1)
            self.chunk_size = cfg.chunk_words

    return PinnedInferencer()


@METRICS.register("minicheck")
class MiniCheck(Metric):
    """MiniCheck support of each output against its article.

    Output range: [0, 1].

    Args:
        cfg: Metric configuration; reads ``cfg.minicheck`` and ``cfg.device``.
    """

    KEYS = ("minicheck",)

    def __init__(self, cfg: MetricsConfig) -> None:
        super().__init__(cfg)
        self._inferencer = _load_inferencer(cfg.minicheck, self._device)
        logger.info("loaded MiniCheck (%s) on %s", cfg.minicheck.model, self._device)

    def claim_support(self, document: str, claim: str) -> float:
        """Support probability of one claim: the maximum over the document's chunks.

        Args:
            document: Grounding document.
            claim: Claim, checked as a single piece of text.

        Returns:
            Probability the claim is supported; in [0, 1].
        """
        import torch

        with torch.inference_mode():
            return float(
                self._inferencer.inference_per_example(document, claim)["max_support_prob"]
            )

    def _score_examples(
        self,
        predictions: list[str],
        references: list[str],
        evidence: list[str],
        gold: list[str | None],
    ) -> list[dict[str, float]]:
        return [
            {
                "minicheck": sentence_mean(
                    output, lambda claim, doc=document: self.claim_support(doc, claim)
                )
            }
            for output, document in zip(predictions, evidence, strict=True)
        ]
