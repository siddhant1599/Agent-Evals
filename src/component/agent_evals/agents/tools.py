"""Shared NLI fact-check tool the Critic uses to scrutinize a draft against its article (D-043).

Not a registered ``Metric``: this runs during generation, inside the Critic's own prompt-building,
not during the separate scoring phase (see ``BenchmarkRunner`` docstring -- generation and scoring
are kept apart because the backbone and the scorer models do not fit on one 24 GiB GPU together).
The pure ranking/labelling logic is split out as plain functions so it is testable on CPU without
loading the real model, the same way ``metrics/summac.py`` splits ``zero_shot_score`` out from
``SummaC``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..logging_utils import get_logger

__all__ = [
    "NLI_MODEL",
    "NLI_REVISION",
    "MAX_INPUT_TOKENS",
    "WeakClaim",
    "classify_claim",
    "rank_entailment",
    "NLIFactCheckTool",
]

logger = get_logger(__name__)

#: D-043: microsoft/deberta-large-mnli (He et al. 2021, DeBERTa), pinned commit.
NLI_MODEL = "microsoft/deberta-large-mnli"
NLI_REVISION = "7296194b9009373def4f7c5dad292651e4b5cf4e"
MAX_INPUT_TOKENS = 512


@dataclass(frozen=True, slots=True)
class WeakClaim:
    """One draft sentence not clearly entailed by any single article sentence.

    Attributes:
        sentence: The draft sentence.
        label: The top NLI label of its best-matching article sentence: "CONTRADICTION" or
            "NEUTRAL" -- never "ENTAILMENT", since those sentences are not flagged.
        best_entailment_prob: That best-matching article sentence's own entailment probability.
    """

    sentence: str
    label: str
    best_entailment_prob: float


def rank_entailment(probs: np.ndarray, entailment_index: int) -> int:
    """Which article sentence most supports one draft sentence.

    Args:
        probs: Shape (article sentences, 3): NLI class probabilities for one draft sentence
            paired with every article sentence.
        entailment_index: Column holding P(entailment).

    Returns:
        Row index of the article sentence with the highest entailment probability.
    """
    return int(np.argmax(probs[:, entailment_index]))


def classify_claim(
    sentence: str, probs_row: np.ndarray, entailment_index: int, contradiction_index: int
) -> WeakClaim | None:
    """Label one draft sentence from its best-matching article sentence's NLI probabilities.

    Args:
        sentence: The draft sentence being judged.
        probs_row: The 3-class probability row for its best-matching article sentence.
        entailment_index: Column holding P(entailment).
        contradiction_index: Column holding P(contradiction).

    Returns:
        None if that row's own top label is entailment (the claim is supported); otherwise a
        ``WeakClaim`` recording the sentence, that row's top label, and its entailment
        probability.
    """
    top = int(np.argmax(probs_row))
    if top == entailment_index:
        return None
    label = "CONTRADICTION" if top == contradiction_index else "NEUTRAL"
    return WeakClaim(
        sentence=sentence, label=label, best_entailment_prob=float(probs_row[entailment_index])
    )


class NLIFactCheckTool:
    """Scores every draft sentence against every article sentence and flags the weak ones.

    Loaded once, like every ``Metric``'s resident model -- normally by the Critic's own
    ``__init__``, so it is a genuine resident model, not reloaded per example.

    Args:
        device: Torch device to run on. Auto-detected ("cuda" if available, else "cpu") when
            not given: this tool loads alongside the backbone during generation, not during the
            separate scoring phase, so it makes its own device choice rather than reading
            ``MetricsConfig``.
    """

    def __init__(self, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL, revision=NLI_REVISION)
        model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL, revision=NLI_REVISION, dtype=torch.float32
        )
        model.eval().to(self._device)
        self._model = model
        label2id = model.config.label2id
        self._entailment = label2id["ENTAILMENT"]
        self._contradiction = label2id["CONTRADICTION"]
        logger.info("loaded Critic NLI tool (%s) on %s", NLI_MODEL, self._device)

    def find_weak_claims(self, article: str, draft: str) -> list[WeakClaim]:
        """Flag every draft sentence not clearly entailed by some article sentence.

        Args:
            article: The source article.
            draft: The summary to check, sentence by sentence.

        Returns:
            One ``WeakClaim`` per weak draft sentence, in draft order. Empty if every draft
            sentence is entailed by some article sentence, or if either text has no sentences.
        """
        import nltk
        import torch

        article_sentences = nltk.tokenize.sent_tokenize(article)
        draft_sentences = nltk.tokenize.sent_tokenize(draft)
        if not article_sentences or not draft_sentences:
            return []

        weak: list[WeakClaim] = []
        for sentence in draft_sentences:
            tokens = self._tokenizer(
                article_sentences,
                [sentence] * len(article_sentences),
                padding=True,
                truncation="only_first",
                max_length=MAX_INPUT_TOKENS,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = self._model(**{k: v.to(self._device) for k, v in tokens.items()}).logits
            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
            best_row = rank_entailment(probs, self._entailment)
            claim = classify_claim(sentence, probs[best_row], self._entailment, self._contradiction)
            if claim is not None:
                weak.append(claim)
        return weak
