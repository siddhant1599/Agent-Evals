"""The T5 reward on XSum: AlignScore of each completion against its article (D-048)."""

from __future__ import annotations

from ..registry import REWARDS
from .base import MetricReward

__all__ = ["AlignScoreReward"]


@REWARDS.register("alignscore")
class AlignScoreReward(MetricReward):
    """AlignScore of each completion against its source article.

    Wraps the registered ``alignscore`` metric (AlignScore-large, ``nli_sp``, D-023) unchanged.
    T5 is judged by MiniCheck, not by this score (D-048).

    Output range: [0, 1]. A blank completion receives ``cfg.blank_reward`` (D-050).

    Args:
        cfg: Reward configuration.
        metric: A constructed ``alignscore`` metric.
    """

    METRIC = "alignscore"
    KEY = "alignscore"
    OUTPUT_RANGE = (0.0, 1.0)
