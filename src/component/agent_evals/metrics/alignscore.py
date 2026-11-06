"""AlignScore-large in ``nli_sp`` mode: a port of the official inference code (D-023, D-028).

Ported from AlignScore at commit ``a0936d5afee642a46b22f6c02a163478447aa493``,
``src/alignscore/inference.py`` and ``model.py`` (Zha et al. 2023, *AlignScore*; MIT,
https://github.com/yuh-zha/AlignScore). The port makes only the changes D-023 lists: the Lightning
checkpoint is loaded into plain modules with every needed weight checked present; spaCy is not
loaded; logging replaces progress bars. Only the parts ``nli_sp`` uses are kept: the RoBERTa
backbone's pooled output and the 3-way head.
"""

from __future__ import annotations

from pathlib import Path

from ..config import MetricsConfig
from ..logging_utils import get_logger
from ..registry import METRICS
from .artifacts import sha256_file
from .base import Metric, MetricError

__all__ = ["AlignScore", "aggregate_alignment", "chunk_article"]

logger = get_logger(__name__)

# The 3-way head's class the official ``nli_sp`` mode reads (``[2][:, 0]`` in ``inference``).
_ALIGNED_CLASS = 0
_TRI_CLASSES = 3

# Checkpoint prefixes of the modules ``nli_sp`` uses, and the one buffer the checkpoint carries
# that current RobertaModel no longer registers.
_BACKBONE_PREFIX = "base_model."
_TRI_PREFIX = "tri_layer."
_IGNORED_BACKBONE_KEYS = frozenset({"embeddings.position_ids"})


def chunk_article(text: str, chunk_words: int) -> list[str]:
    """Official ``inference_per_example`` article chunking.

    Args:
        text: Article.
        chunk_words: Target chunk size in whitespace words.

    Returns:
        Chunks of whole sentences joined by spaces; ``[""]`` for text with no sentences.
    """
    import nltk

    sentences = nltk.tokenize.sent_tokenize(text) or [""]
    n_chunks = len(text.strip().split()) // chunk_words + 1
    per_chunk = max(len(sentences) // n_chunks, 1)
    return [" ".join(sentences[i : i + per_chunk]) for i in range(0, len(sentences), per_chunk)]


def aggregate_alignment(probabilities: list[float], n_chunks: int, n_sentences: int) -> float:
    """Official ``nli_sp`` aggregation: max over chunks per summary sentence, then the mean.

    Args:
        probabilities: Aligned-class probability per (chunk, sentence), chunks in the outer loop.
        n_chunks: Number of article chunks.
        n_sentences: Number of summary sentences.

    Returns:
        The example's score; in [0, 1].
    """
    import torch

    matrix = torch.tensor(probabilities).view(n_chunks, n_sentences)
    return matrix.max(dim=0).values.mean().item()


@METRICS.register("alignscore")
class AlignScore(Metric):
    """AlignScore of each output against its article.

    Output range: [0, 1].

    Args:
        cfg: Metric configuration; reads ``cfg.alignscore`` and ``cfg.device``.
    """

    KEYS = ("alignscore",)

    def __init__(self, cfg: MetricsConfig) -> None:
        super().__init__(cfg)
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoTokenizer, RobertaModel

        self._cfg = cfg.alignscore
        path = Path(
            hf_hub_download(
                self._cfg.checkpoint_repo,
                self._cfg.checkpoint_file,
                revision=self._cfg.checkpoint_revision,
            )
        )
        digest = sha256_file(path)
        if digest != self._cfg.checkpoint_sha256:
            raise MetricError(f"{path} has sha256 {digest}, expected {self._cfg.checkpoint_sha256}")
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["state_dict"]

        config = AutoConfig.from_pretrained(
            self._cfg.backbone, revision=self._cfg.backbone_revision
        )
        backbone = RobertaModel(config)
        backbone_state = {
            key.removeprefix(_BACKBONE_PREFIX): value
            for key, value in state.items()
            if key.startswith(_BACKBONE_PREFIX)
        }
        missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
        if missing or set(unexpected) - _IGNORED_BACKBONE_KEYS:
            raise MetricError(
                f"AlignScore backbone weights do not match: missing {missing}, "
                f"unexpected {unexpected}"
            )
        head = torch.nn.Linear(config.hidden_size, _TRI_CLASSES)
        head.load_state_dict(
            {
                key.removeprefix(_TRI_PREFIX): value
                for key, value in state.items()
                if key.startswith(_TRI_PREFIX)
            },
            strict=True,
        )
        self._backbone = backbone.float().eval().to(self._device)
        self._head = head.float().eval().to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._cfg.backbone, revision=self._cfg.backbone_revision
        )
        logger.info("loaded AlignScore (%s) on %s", self._cfg.checkpoint_file, self._device)

    def pair_scores(self, documents: list[str], summaries: list[str]) -> list[float]:
        """AlignScore for every (document, summary) pair.

        Args:
            documents: Articles.
            summaries: Summaries, aligned with ``documents``.

        Returns:
            One score per pair.

        Raises:
            MetricError: If a summary has no sentences.
        """
        return [
            self._pair_score(document, summary)
            for document, summary in zip(documents, summaries, strict=True)
        ]

    def claim_support(self, document: str, claim: str) -> float:
        """AlignScore of one claim against one document, as the citation metric's judge (D-135).

        A claim with no text, such as a sentence that was only citation markers (``[2]``), states
        nothing the document could support and scores 0.0, the bottom of the range (D-137).

        Args:
            document: Grounding document.
            claim: Claim, scored as a summary is: per sentence, then averaged.

        Returns:
            The score; in [0, 1].
        """
        if not claim.strip():
            return 0.0
        return self._pair_score(document, claim)

    def _pair_score(self, document: str, summary: str) -> float:
        import nltk

        chunks = chunk_article(document, self._cfg.chunk_words)
        sentences = nltk.tokenize.sent_tokenize(summary)
        if not sentences:
            raise MetricError("AlignScore needs a summary with at least one sentence")
        premises = [chunk for chunk in chunks for _ in sentences]
        hypotheses = [sentence for _ in chunks for sentence in sentences]
        probabilities: list[float] = []
        for start in range(0, len(premises), self._cfg.batch_size):
            stop = start + self._cfg.batch_size
            probabilities.extend(
                self._aligned_probabilities(premises[start:stop], hypotheses[start:stop])
            )
        return aggregate_alignment(probabilities, len(chunks), len(sentences))

    def _aligned_probabilities(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        import torch

        max_length = self._tokenizer.model_max_length
        try:
            tokens = self._tokenizer(
                premises,
                hypotheses,
                truncation="only_first",
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
        except Exception:  # the official code falls back on any tokenizer failure
            logger.warning("AlignScore: summary side too long for only_first; using longest_first")
            tokens = self._tokenizer(
                premises,
                hypotheses,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
        tokens = {key: value.to(self._device) for key, value in tokens.items()}
        with torch.inference_mode():
            pooled = self._backbone(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                token_type_ids=tokens.get("token_type_ids"),
            ).pooler_output
            probabilities = torch.softmax(self._head(pooled), dim=-1)[:, _ALIGNED_CLASS]
        return probabilities.float().cpu().tolist()

    def _score_examples(
        self,
        predictions: list[str],
        references: list[str],
        evidence: list[str],
        gold: list[str | None],
    ) -> list[dict[str, float]]:
        return [{"alignscore": float(score)} for score in self.pair_scores(evidence, predictions)]
