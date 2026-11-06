"""Whether an output copies a prompt's worked examples instead of using its own source (D-110).

T1's feedback and refine prompts carry worked examples (D-103). D-109 found Llama and Mistral
outputs that summarize one of those examples instead of the article under work. This metric flags
such an output automatically, so the leak rate is part of every run's ``summary.json``. The ASQA
variant checks ALCE's four worked examples (D-118, D-128). The ASQA Self-Refine variant checks the
text written for T1's ASQA examples, under its own key (D-150, D-151).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..agents.asqa_examples import ALCE_ASQA_EXAMPLES
from ..agents.self_refine_examples import ASQA_SELF_REFINE_EXAMPLES, XSUM_SELF_REFINE_EXAMPLES
from ..config import MetricsConfig
from ..registry import METRICS
from .base import Metric
from .citation_markers import remove_citations

__all__ = ["ASQASelfRefineExampleOverlap", "ASQAWorkedExampleOverlap", "WorkedExampleOverlap"]

_WORD = re.compile(r"\w+")


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.casefold())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


@METRICS.register("worked_example_overlap")
class WorkedExampleOverlap(Metric):
    """1.0 when an output shares an n-word sequence with a worked example and not with its article.

    Words are lowercased runs of letters and digits. The worked examples' articles, drafts,
    feedback and revised summaries are all checked. A sequence that also occurs in the output's own
    article (the evidence) is ignored, so shared news phrasing is not counted as copying.

    Output range: ``worked_example_overlap`` is 0.0 or 1.0; its run mean is the leak rate.

    Args:
        cfg: Metric configuration; reads ``cfg.worked_example_overlap.ngram``.
    """

    KEYS = ("worked_example_overlap",)

    def __init__(self, cfg: MetricsConfig) -> None:
        super().__init__(cfg)
        self._n = cfg.worked_example_overlap.ngram
        texts = self._example_texts()
        self._example_ngrams = set().union(*(_ngrams(text, self._n) for text in texts))

    def _example_texts(self) -> Iterable[str]:
        """T1's XSum worked examples: each one's article, draft, feedback and revision (D-103)."""
        return (
            f"{e.article} {e.draft} {e.feedback} {e.revised}" for e in XSUM_SELF_REFINE_EXAMPLES
        )

    def _score_examples(
        self,
        predictions: list[str],
        references: list[str],
        evidence: list[str],
        gold: list[str | None],
    ) -> list[dict[str, float]]:
        rows = []
        for prediction, article in zip(predictions, evidence, strict=True):
            copied = (_ngrams(prediction, self._n) & self._example_ngrams) - _ngrams(
                article, self._n
            )
            rows.append({self.KEYS[0]: 1.0 if copied else 0.0})
        return rows


@METRICS.register("asqa_worked_example_overlap")
class ASQAWorkedExampleOverlap(WorkedExampleOverlap):
    """``worked_example_overlap`` against all four of ALCE's ASQA worked examples (D-128).

    Each example's question, each of its passages (title and text) and its answer, without
    citation markers, are checked as separate texts. All four are checked, not only the two a run
    draws: a match with an example the run did not show can only be a coincidence, so the check is
    never weaker than one restricted to the drawn pair. The evidence is the question's own
    passages, so a sequence the model could have read there is not counted.

    Output range: ``worked_example_overlap`` is 0.0 or 1.0; its run mean is the leak rate.
    """

    def _example_texts(self) -> Iterable[str]:
        for example in ALCE_ASQA_EXAMPLES:
            yield example.question
            yield from (f"{title} {text}" for title, text in example.passages)
            yield remove_citations(example.answer)


@METRICS.register("asqa_self_refine_example_overlap")
class ASQASelfRefineExampleOverlap(WorkedExampleOverlap):
    """``worked_example_overlap`` against the text written for T1's ASQA examples (D-150, D-151).

    Checks each example's draft and feedback, and its revision where that is not ALCE's answer,
    without citation markers. ALCE's questions, passages and answers are left to
    ``asqa_worked_example_overlap``, so its key keeps the meaning D-128 gave it, and this metric
    has its own key. The evidence is the question's own passages, as there.

    Output range: ``self_refine_example_overlap`` is 0.0 or 1.0; its run mean is the leak rate.
    """

    KEYS = ("self_refine_example_overlap",)

    def _example_texts(self) -> Iterable[str]:
        for example in ASQA_SELF_REFINE_EXAMPLES:
            yield remove_citations(example.draft)
            yield remove_citations(example.feedback)
            if example.revised != example.qa.answer:
                yield remove_citations(example.revised)
