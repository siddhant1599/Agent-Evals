"""RL for T5: GRPO rewards (AGENTS.md §4.6). Training and KL monitoring arrive with the trainer.

Importing this package registers every reward it holds. A reward receives a metric its caller
built, so importing loads no models and downloads nothing.
"""

from .alignscore import AlignScoreReward
from .asqa import CitationPrecisionReward, CitationRecallReward, ExactMatchRecallReward
from .base import (
    MetricReward,
    Reward,
    RewardError,
    SharedMetric,
    as_trl_reward,
    completion_text,
    training_template,
    worked_examples_seed,
)

__all__ = [
    "AlignScoreReward",
    "CitationPrecisionReward",
    "CitationRecallReward",
    "ExactMatchRecallReward",
    "MetricReward",
    "Reward",
    "RewardError",
    "SharedMetric",
    "as_trl_reward",
    "completion_text",
    "training_template",
    "worked_examples_seed",
]
