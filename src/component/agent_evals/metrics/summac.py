"""SummaC-ZS and SummaC-Conv: a port of the official code (D-018, D-028, D-029).

Ported from ``summac`` 0.0.4, ``model_summac.py`` (Laban et al. 2022, *SummaC*; Apache-2.0,
https://github.com/tingofurro/summac). The port makes only the changes D-018 lists: the
pair-encoding call for transformers 5, with segment ids requested explicitly (D-029); pinned model
and checkpoint; no cache folder; logging instead of ``print``. Both variants share one NLI image
per example, built in batches as ``SummaCImager.build_images`` builds them (D-028).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..config import MetricsConfig
from ..logging_utils import get_logger
from ..registry import METRICS
from .artifacts import fetch_pinned
from .base import Metric, MetricError

__all__ = ["SummaC", "conv_histogram", "conv_scores", "split_sentences", "zero_shot_score"]

logger = get_logger(__name__)

# Image channels, in the order the official imager stores them.
_ENTAILMENT, _CONTRADICTION, _NEUTRAL = 0, 1, 2

# SummaC-Conv's aggregated feature is the mean repeated three times (official ``agg="mean"``).
_CONV_FEATURES = 3
_CONV_CLASSES = 2


def split_sentences(text: str, min_chars: int) -> list[str]:
    """Official ``split_sentences``: nltk sentences longer than ``min_chars`` characters.

    Args:
        text: Text to split.
        min_chars: Sentences of this many characters or fewer are dropped.

    Returns:
        The kept sentences, in order.
    """
    import nltk

    return [sentence for sentence in nltk.tokenize.sent_tokenize(text) if len(sentence) > min_chars]


def zero_shot_score(image: np.ndarray) -> float:
    """Official ``image2score`` with ``op1="max"``, ``op2="mean"`` and both labels used.

    Args:
        image: Array of shape (3, article sentences, summary sentences) holding the entailment,
            contradiction and neutral probabilities.

    Returns:
        Mean over summary sentences of (max entailment - max contradiction); in [-1, 1].
    """
    entailment = np.max(image[_ENTAILMENT], axis=0)
    contradiction = np.max(image[_CONTRADICTION], axis=0)
    return float(np.mean(entailment - contradiction))


def conv_histogram(image: np.ndarray, bins: np.ndarray, n_rows: int) -> np.ndarray:
    """Official ``compute_histogram`` for ``nli_labels="e"``: one entailment histogram per sentence.

    Args:
        image: Array of shape (3, article sentences, summary sentences).
        bins: Histogram bin edges.
        n_rows: Summary sentences kept; fewer are zero-padded, more are cut.

    Returns:
        Array of shape (n_rows, len(bins) - 1) of counts.
    """
    n_bins = len(bins) - 1
    rows = [
        np.histogram(image[_ENTAILMENT, :, column], range=(0, 1), bins=bins, density=False)[0]
        for column in range(image.shape[2])
    ]
    rows += [np.zeros(n_bins)] * (n_rows - len(rows))
    return np.array(rows[:n_rows], dtype=float)


def conv_scores(histograms: np.ndarray, weights: Mapping[str, object]) -> list[float]:
    """Official ``SummaCConv.forward`` and ``score`` with ``agg="mean"``.

    Args:
        histograms: Array of shape (examples, rows, bins).
        weights: ``mlp.weight``, ``mlp.bias``, ``layer_final.weight``, ``layer_final.bias`` tensors,
            all on the device to compute on.

    Returns:
        Probability of the consistent class per example; in [0, 1].
    """
    import torch
    import torch.nn.functional as F

    device = weights["mlp.weight"].device
    features = torch.as_tensor(histograms, dtype=torch.float32, device=device)
    n_examples, n_rows, _ = features.shape
    lengths = (features.sum(dim=-1) != 0.0).long().sum(dim=-1).tolist()
    row_scores = F.linear(features, weights["mlp.weight"], weights["mlp.bias"]).reshape(
        n_examples, n_rows
    )
    pooled = torch.stack(
        [
            (
                row_scores[i, :length].mean().repeat(_CONV_FEATURES)
                if length > 0
                else torch.zeros(_CONV_FEATURES, device=device)
            )
            for i, length in enumerate(lengths)
        ]
    )
    logits = F.linear(pooled, weights["layer_final.weight"], weights["layer_final.bias"])
    return torch.softmax(logits, dim=-1)[:, 1].tolist()


def check_conv_weights(state: Mapping[str, object], n_bins: int) -> dict[str, object]:
    """Verify the SummaC-Conv state dict has exactly the official layers and shapes.

    Args:
        state: Loaded state dict.
        n_bins: Number of histogram bins the configuration defines.

    Returns:
        The state as a plain dict.

    Raises:
        MetricError: On missing, extra or misshapen tensors.
    """
    expected = {
        "mlp.weight": (1, n_bins),
        "mlp.bias": (1,),
        "layer_final.weight": (_CONV_CLASSES, _CONV_FEATURES),
        "layer_final.bias": (_CONV_CLASSES,),
    }
    if set(state) != set(expected):
        raise MetricError(
            f"SummaC-Conv weights have keys {sorted(state)}, expected {sorted(expected)}"
        )
    for key, shape in expected.items():
        if tuple(state[key].shape) != shape:
            raise MetricError(
                f"SummaC-Conv {key} has shape {tuple(state[key].shape)}, expected {shape}"
            )
    return dict(state)


@METRICS.register("summac")
class SummaC(Metric):
    """SummaC-ZS and SummaC-Conv of each output against its article.

    Output range: ``summac_zs`` in [-1, 1]; ``summac_conv`` in [0, 1]. An output whose sentences
    are all dropped by the length filter gets the official all-zero image: ``summac_zs`` 0.0, and
    SummaC-Conv's score for a single zero histogram entry.

    Args:
        cfg: Metric configuration; reads ``cfg.summac``, ``cfg.device`` and ``cfg.cache_dir``.
    """

    KEYS = ("summac_zs", "summac_conv")

    def __init__(self, cfg: MetricsConfig) -> None:
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._cfg = cfg.summac
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._cfg.nli_model, revision=self._cfg.nli_revision
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self._cfg.nli_model, revision=self._cfg.nli_revision, dtype=torch.float32
        )
        model.eval().to(self._device)
        if self._cfg.half_precision and torch.device(self._device).type == "cuda":
            model.half()
        self._model = model
        self._label_indices = {
            _ENTAILMENT: self._cfg.entailment_index,
            _CONTRADICTION: self._cfg.contradiction_index,
            _NEUTRAL: 3 - self._cfg.entailment_index - self._cfg.contradiction_index,
        }
        path = fetch_pinned(
            self._cfg.conv_checkpoint_url,
            self._cfg.conv_checkpoint_sha256,
            Path(cfg.cache_dir).expanduser() / "summac",
        )
        state = torch.load(path, map_location=self._device, weights_only=True)
        self._bins = np.asarray(self._cfg.conv_bins, dtype=float)
        self._conv_weights = check_conv_weights(state, len(self._bins) - 1)
        logger.info("loaded SummaC (%s) on %s", self._cfg.nli_model, self._device)

    def images(self, documents: list[str], summaries: list[str]) -> list[np.ndarray]:
        """Build the NLI image of every (document, summary) pair.

        Args:
            documents: Articles.
            summaries: Summaries, aligned with ``documents``.

        Returns:
            One array of shape (3, article sentences, summary sentences) per pair.
        """
        import torch

        images: list[np.ndarray] = []
        pairs: list[tuple[int, int, int, str, str]] = []
        for index, (document, summary) in enumerate(zip(documents, summaries, strict=True)):
            doc_sents = split_sentences(document, self._cfg.min_sentence_chars)
            doc_sents = doc_sents[: self._cfg.max_document_sentences]
            sum_sents = split_sentences(summary, self._cfg.min_sentence_chars)
            if not doc_sents or not sum_sents:
                images.append(np.zeros((3, 1, 1)))
                continue
            images.append(np.zeros((3, len(doc_sents), len(sum_sents))))
            pairs.extend(
                (index, i, j, doc_sents[i], sum_sents[j])
                for i in range(len(doc_sents))
                for j in range(len(sum_sents))
            )
        for start in range(0, len(pairs), self._cfg.batch_size):
            batch = pairs[start : start + self._cfg.batch_size]
            tokens = self._tokenizer(
                [pair[3] for pair in batch],
                [pair[4] for pair in batch],
                padding=True,
                truncation=self._cfg.truncation,
                max_length=self._cfg.max_input_tokens,
                return_tensors="pt",
                return_token_type_ids=True,
            )
            with torch.inference_mode():
                logits = self._model(**{k: v.to(self._device) for k, v in tokens.items()}).logits
            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
            for (index, i, j, _, _), row in zip(batch, probs, strict=True):
                for channel, label in self._label_indices.items():
                    images[index][channel, i, j] = row[label]
        return images

    def pair_scores(self, documents: list[str], summaries: list[str]) -> dict[str, list[float]]:
        """Both SummaC scores for every (document, summary) pair.

        Args:
            documents: Articles.
            summaries: Summaries, aligned with ``documents``.

        Returns:
            ``{"summac_zs": [...], "summac_conv": [...]}``.
        """
        images = self.images(documents, summaries)
        histograms = np.stack(
            [
                conv_histogram(image, self._bins, self._cfg.conv_max_summary_sentences)
                for image in images
            ]
        )
        return {
            "summac_zs": [zero_shot_score(image) for image in images],
            "summac_conv": conv_scores(histograms, self._conv_weights),
        }

    def _score_examples(
        self,
        predictions: list[str],
        references: list[str],
        evidence: list[str],
        gold: list[str | None],
    ) -> list[dict[str, float]]:
        scores = self.pair_scores(evidence, predictions)
        return [
            {"summac_zs": float(zs), "summac_conv": float(conv)}
            for zs, conv in zip(scores["summac_zs"], scores["summac_conv"], strict=True)
        ]
