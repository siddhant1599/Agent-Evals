"""The T5 reward on ASQA: Huang et al. 2024's three terms, each its own GRPO reward (D-133).

Correctness is exact-match recall of the gold short answers (``str_em``, D-128). Citation recall
and precision are ALCE's (D-128), judged by AlignScore in training so that MiniCheck, which
judges every reported ASQA score, stays held out (D-135); the training config sets
``metrics.citation.judge: alignscore``. TRL receives the three as separate reward functions and
combines them with GDPO and equal weights (D-134, D-135).

Each term is one score per answer in [0, 1], the answer's ALCE value (D-135). A blank answer
scores ``reward.blank_reward`` on every term (D-050). The correctness term reads the answer with
its citation markers removed, as evaluation does (D-126); the citation terms read the markers.
"""

from __future__ import annotations

from ..registry import REWARDS
from .base import MetricReward

__all__ = ["CitationPrecisionReward", "CitationRecallReward", "ExactMatchRecallReward"]


@REWARDS.register("str_em")
class ExactMatchRecallReward(MetricReward):
    """Share of the question's sub-questions with a gold short answer in the answer (D-128).

    Output range: [0, 1]. Reads the ``gold`` column (D-126).

    Args:
        cfg: Reward configuration.
        metric: A constructed ``str_em`` metric.
        strip_citations: Remove citation markers first (D-126).
    """

    METRIC = "str_em"
    KEY = "str_em"
    OUTPUT_RANGE = (0.0, 1.0)
    COLUMNS = ("evidence", "reference", "gold")


@REWARDS.register("citation_recall")
class CitationRecallReward(MetricReward):
    """Share of the answer's sentences entailed by the passages they cite (ALCE, D-128).

    Output range: [0, 1]. The judge is ``metrics.citation.judge`` (AlignScore in training, D-135).

    Args:
        cfg: Reward configuration.
        metric: A constructed ``citation`` metric, or a ``SharedMetric`` over it.
        strip_citations: Ignored: the citation metric reads the markers.
    """

    METRIC = "citation"
    KEY = "citation_recall"
    OUTPUT_RANGE = (0.0, 1.0)


@REWARDS.register("citation_precision")
class CitationPrecisionReward(MetricReward):
    """Share of the answer's citations that help support their sentence (ALCE, D-128).

    Output range: [0, 1]; 0 for an answer with no counted citation. The judge is
    ``metrics.citation.judge`` (AlignScore in training, D-135).

    Args:
        cfg: Reward configuration.
        metric: A constructed ``citation`` metric, or a ``SharedMetric`` over it.
        strip_citations: Ignored: the citation metric reads the markers.
    """

    METRIC = "citation"
    KEY = "citation_precision"
    OUTPUT_RANGE = (0.0, 1.0)
