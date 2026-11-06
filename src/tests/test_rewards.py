"""GRPO rewards: the Reward contract, AlignScoreReward over a stub metric, and the TRL adapter.

CPU only. The stub metric stands in for AlignScore, so no model loads and nothing downloads.
"""

from __future__ import annotations

import math

import pytest
from agent_evals.config import MetricsConfig, RewardConfig
from agent_evals.metrics import Metric
from agent_evals.registry import METRICS, REWARDS
from agent_evals.rl import AlignScoreReward, RewardError, as_trl_reward


class StubAlignScore(Metric):
    """Emits the ``alignscore`` key: len(text) / 100, or scripted scores. Records every call."""

    KEYS = ("alignscore",)

    def __init__(self, scores=None):
        super().__init__(MetricsConfig(device="cpu"))
        self.scores = scores
        self.calls = []

    def _score_examples(self, predictions, references, evidence, gold):
        self.calls.append((predictions, references, evidence))
        if self.scores is not None:
            return [{"alignscore": score} for score in self.scores]
        return [{"alignscore": len(p) / 100} for p in predictions]


class StubOtherMetric(StubAlignScore):
    KEYS = ("other",)


def _reward(metric=None, **cfg):
    return REWARDS.build("alignscore", RewardConfig(**cfg), metric or StubAlignScore())


def _columns(n):
    return {
        "evidence": [f"FAKE_EXAMPLE article {i}" for i in range(n)],
        "reference": [f"FAKE_EXAMPLE reference {i}" for i in range(n)],
    }


def _assistant(text):
    return [{"role": "assistant", "content": text}]


class TestRegistration:
    def test_alignscore_reward_is_registered(self):
        assert "alignscore" in REWARDS.names()
        assert REWARDS.get("alignscore") is AlignScoreReward

    def test_declared_range_is_alignscores(self):
        assert AlignScoreReward.OUTPUT_RANGE == (0.0, 1.0)

    def test_wraps_the_registered_alignscore_metric(self):
        assert AlignScoreReward.METRIC == "alignscore"
        assert METRICS.get(AlignScoreReward.METRIC).KEYS == (AlignScoreReward.KEY,)


class TestScore:
    def test_one_float_per_completion_in_order(self):
        metric = StubAlignScore()
        rewards = _reward(metric).score(["a", "bbb", "cc"], **_columns(3))
        assert rewards == [0.01, 0.03, 0.02]
        assert all(isinstance(r, float) for r in rewards)
        predictions, references, evidence = metric.calls[0]
        assert predictions == ["a", "bbb", "cc"]
        assert evidence == _columns(3)["evidence"]
        assert references == _columns(3)["reference"]

    def test_conversational_completions_score_like_strings(self):
        reward = _reward()
        texts = ["a", "bbb"]
        assert reward.score([_assistant(t) for t in texts], **_columns(2)) == reward.score(
            texts, **_columns(2)
        )

    def test_extra_trl_arguments_are_ignored(self):
        rewards = _reward().score(
            ["a", "bb"],
            prompts=["p0", "p1"],
            completion_ids=[[1], [2, 3]],
            trainer_state=object(),
            log_extra=lambda *a, **k: None,
            log_metric=lambda *a, **k: None,
            id=["x", "y"],
            **_columns(2),
        )
        assert rewards == [0.01, 0.02]

    def test_empty_batch_raises(self):
        with pytest.raises(RewardError, match="no completions"):
            _reward().score([], **_columns(0))


class TestBlankCompletions:
    def test_blank_gets_blank_reward_and_never_reaches_the_metric(self):
        metric = StubAlignScore()
        rewards = _reward(metric).score(["a", "", " \n\t", "cc"], **_columns(4))
        assert rewards == [0.01, 0.0, 0.0, 0.02]
        predictions, _, evidence = metric.calls[0]
        assert predictions == ["a", "cc"]
        assert evidence == ["FAKE_EXAMPLE article 0", "FAKE_EXAMPLE article 3"]

    def test_all_blank_batch_does_not_call_the_metric(self):
        metric = StubAlignScore()
        assert _reward(metric).score(["", _assistant("  ")], **_columns(2)) == [0.0, 0.0]
        assert metric.calls == []

    def test_configured_blank_reward_is_used(self):
        assert _reward(blank_reward=0.25).score(["", "a"], **_columns(2)) == [0.25, 0.01]

    @pytest.mark.parametrize("value", [-0.1, 1.5])
    def test_blank_reward_outside_the_range_is_refused(self, value):
        with pytest.raises(RewardError, match="output range"):
            _reward(blank_reward=value)


class TestColumns:
    def test_missing_evidence_raises(self):
        columns = _columns(2)
        del columns["evidence"]
        with pytest.raises(RewardError, match="'evidence'"):
            _reward().score(["a", "b"], **columns)

    def test_misaligned_column_raises(self):
        with pytest.raises(RewardError, match="3 values for 2 completions"):
            _reward().score(["a", "b"], **_columns(3))

    def test_string_column_raises(self):
        columns = _columns(2)
        columns["evidence"] = "FAKE_EXAMPLE one article for every completion"
        with pytest.raises(RewardError, match="must be a list"):
            _reward().score(["a", "b"], **columns)


class TestValues:
    def test_out_of_range_score_raises(self):
        with pytest.raises(RewardError, match="outside"):
            _reward(StubAlignScore(scores=[1.5])).score(["a"], **_columns(1))

    def test_nan_from_the_metric_raises(self):
        with pytest.raises(RewardError, match="could not score"):
            _reward(StubAlignScore(scores=[math.nan])).score(["a"], **_columns(1))


class TestConstruction:
    def test_metric_without_the_key_is_refused(self):
        with pytest.raises(RewardError, match="'alignscore'"):
            _reward(StubOtherMetric())

    def test_name_registered_to_another_class_is_refused(self):
        class Other(AlignScoreReward):
            pass

        with REWARDS.temporary("other", Other), pytest.raises(RewardError, match="registered"):
            AlignScoreReward(RewardConfig(name="other"), StubAlignScore())

    def test_unregistered_name_is_refused(self):
        with pytest.raises(RewardError, match="unknown reward"):
            AlignScoreReward(RewardConfig(name="nope"), StubAlignScore())


@pytest.mark.parametrize(
    "completion",
    [
        [],
        [{"role": "user", "content": "a"}],
        [{"role": "assistant", "content": ["a"]}],
        _assistant("a") + _assistant("b"),
        3,
    ],
)
def test_malformed_completion_raises(completion):
    with pytest.raises(RewardError, match="single assistant message"):
        _reward().score([completion], **_columns(1))


class TestTrlAdapter:
    def test_forwards_to_score_as_trl_calls_it(self):
        reward = _reward()
        trl_reward = as_trl_reward(reward)
        out = trl_reward(
            prompts=["p0", "p1"],
            completions=["a", ""],
            completion_ids=[[1], []],
            trainer_state=object(),
            **_columns(2),
        )
        assert out == reward.score(["a", ""], **_columns(2)) == [0.01, 0.0]

    def test_is_named_after_the_reward(self):
        assert as_trl_reward(_reward()).__name__ == "alignscore"
