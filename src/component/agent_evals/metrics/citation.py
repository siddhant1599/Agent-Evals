"""Citation recall and precision as ALCE defines them, with MiniCheck as the judge (D-120).

The algorithm is ALCE's ``compute_autoais`` (Gao et al. 2023, §3.3, Figure 3; ``eval.py``, commit
``246c476``). The output is split into sentences (NLTK, as ALCE). For each sentence:

* **Recall** is 1 when it cites at least one passage, every citation is in range, and the cited
  passages together entail the sentence with its markers removed; otherwise 0.
* **Precision** is scored per citation. A citation counts as precise when its sentence's recall
  is 1 and it is not irrelevant. A citation is irrelevant when it does not entail the sentence on
  its own but the sentence's other citations still do. A sentence whose recall is 0 contributes 0
  for each of its counted citations.

An output's recall is the mean over its sentences, and its precision is the precise citations over
all counted citations (0 when none are counted). At most ``max_citations_per_sentence`` citations
are read per sentence. Passages are given to the judge as ``Title: <title>`` and the text on the
next line, joined by newlines, as ALCE gives them to its judge.

The judge is MiniCheck (D-024) at its 0.5 threshold, in place of ALCE's TRUE model (D-120). T5's
ASQA reward runs the same algorithm with AlignScore as the judge (``metrics.citation.judge``), so
that MiniCheck stays held out (D-135). One deliberate difference: a citation ``[0]`` counts as out
of range; ALCE's code would read it as the last passage.
"""

from __future__ import annotations

from ..config import MetricsConfig
from ..datasets.asqa import parse_passages
from ..registry import METRICS
from .alignscore import AlignScore
from .base import Metric, MetricError
from .citation_markers import CITATION, remove_citations
from .minicheck import MiniCheck

__all__ = ["CitationQuality"]


@METRICS.register("citation")
class CitationQuality(Metric):
    """ALCE citation recall and precision of each output against the passages it was given.

    Reads the output with its citation markers (``READS_CITATIONS``) and the passages from the
    evidence, in ALCE's format (D-117).

    Output range: ``citation_recall`` and ``citation_precision`` in [0, 1].

    Args:
        cfg: Metric configuration; reads ``cfg.citation``, the judge's own group
            (``cfg.minicheck`` or ``cfg.alignscore``) and ``cfg.device``.
    """

    KEYS = ("citation_recall", "citation_precision")
    READS_CITATIONS = True

    def __init__(self, cfg: MetricsConfig) -> None:
        super().__init__(cfg)
        if cfg.citation.judge == "alignscore":
            self._judge: MiniCheck | AlignScore = AlignScore(cfg)
            self._threshold = cfg.alignscore.threshold
        else:
            self._judge = MiniCheck(cfg)
            self._threshold = cfg.minicheck.threshold
        self._max_citations = cfg.citation.max_citations_per_sentence

    def _entails(self, passages: list[dict[str, str]], cited: list[int], claim: str) -> bool:
        premise = "\n".join(f"Title: {passages[i]['title']}\n{passages[i]['text']}" for i in cited)
        return self._judge.claim_support(premise, claim) > self._threshold

    def _score_examples(
        self,
        predictions: list[str],
        references: list[str],
        evidence: list[str],
        gold: list[str | None],
    ) -> list[dict[str, float]]:
        import nltk

        rows = []
        for output, text in zip(predictions, evidence, strict=True):
            try:
                passages = parse_passages(text)
            except ValueError as exc:
                raise MetricError(f"citation metric cannot read the passages: {exc}") from exc
            sentences = nltk.tokenize.sent_tokenize(output)
            if not sentences:
                raise MetricError("citation metric needs an output with at least one sentence")
            supported = precise = counted = 0
            for sentence in sentences:
                claim = remove_citations(sentence).strip()
                cited = [int(number) - 1 for number in CITATION.findall(sentence)]
                if not cited or any(not 0 <= i < len(passages) for i in cited):
                    continue
                cited = cited[: self._max_citations]
                counted += len(cited)
                if not self._entails(passages, cited, claim):
                    continue
                supported += 1
                if len(cited) == 1:
                    precise += 1
                    continue
                for position, passage in enumerate(cited):
                    others = cited[:position] + cited[position + 1 :]
                    if self._entails(passages, [passage], claim) or not self._entails(
                        passages, others, claim
                    ):
                        precise += 1
            rows.append(
                {
                    "citation_recall": supported / len(sentences),
                    "citation_precision": precise / counted if counted else 0.0,
                }
            )
        return rows
