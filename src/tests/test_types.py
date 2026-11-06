"""Contracts in types.py: cost arithmetic, immutability, and construction-time validation."""

from __future__ import annotations

import dataclasses

import pytest
from agent_evals.types import Example, Message, TokenCost, TopologyResult


class TestTokenCost:
    def test_total_sums_prompt_and_completion(self):
        assert TokenCost(prompt_tokens=800, completion_tokens=60).total == 860

    def test_addition_aggregates_both_fields(self):
        total = TokenCost(800, 60) + TokenCost(200, 40)
        assert (total.prompt_tokens, total.completion_tokens) == (1000, 100)

    def test_zero_is_a_usable_sum_start(self):
        costs = [TokenCost(10, 1), TokenCost(20, 2), TokenCost(30, 3)]
        assert sum(costs, TokenCost.zero()).total == 66

    def test_negative_counts_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            TokenCost(prompt_tokens=-1, completion_tokens=0)

    def test_as_dict_keys_are_the_stable_metric_keys(self):
        assert TokenCost(800, 60).as_dict() == {
            "prompt_tokens": 800,
            "completion_tokens": 60,
            "total_tokens": 860,
        }

    def test_frozen(self):
        cost = TokenCost(1, 1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cost.prompt_tokens = 5


class TestExample:
    def test_valid_tiers_accepted(self):
        for tier in (1, 2, 3):
            assert Example("id", tier, "summarization", "s", "e", "r").tier == tier

    def test_tier_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="tier must be 1, 2 or 3"):
            Example("id", 4, "summarization", "s", "e", "r")

    def test_empty_id_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            Example("", 1, "summarization", "s", "e", "r")

    def test_question_defaults_to_none_for_summarization(self):
        assert Example("id", 1, "summarization", "s", "e", "r").question is None

    def test_gold_defaults_to_none(self):
        assert Example("id", 1, "summarization", "s", "e", "r").gold is None

    def test_gold_accepts_a_json_object(self):
        gold = '{"short_answers": [["Paris"]], "long_answers": ["It is Paris."]}'
        assert Example("id", 2, "grounded_qa", "s", "e", "r", gold=gold).gold == gold

    @pytest.mark.parametrize(
        ("gold", "error", "message"),
        [
            ("not json", ValueError, "not valid JSON"),
            ('["a list"]', ValueError, "JSON object"),
            ({"short_answers": []}, TypeError, "JSON string"),
        ],
    )
    def test_gold_must_be_a_json_object_string(self, gold, error, message):
        with pytest.raises(error, match=message):
            Example("id", 2, "grounded_qa", "s", "e", "r", gold=gold)

    def test_source_words_use_whitespace_and_support_empty_unicode(self):
        assert Example("id", 1, "summarization", "", "", "r").source_words == 0
        example = Example("id", 1, "summarization", "café\t東京\nhello", "e", "r")
        assert example.source_words == 3
        assert example.truncated is False

    def test_serving_can_replace_evidence_without_losing_original_length(self):
        original = Example("id", 1, "summarization", "one two three", "one two three", "r")
        shortened = dataclasses.replace(original, evidence="one", truncated=True)
        assert shortened.source == original.source
        assert shortened.source_words == original.source_words
        assert shortened.evidence == "one" and shortened.truncated
        with pytest.raises(dataclasses.FrozenInstanceError):
            shortened.source_words = 1

    def test_truncated_flag_is_not_a_truthy_string(self):
        with pytest.raises(TypeError, match="boolean"):
            Example("id", 1, "summarization", "s", "e", "r", truncated="false")


class TestTopologyResult:
    def test_single_pass_result_is_valid(self, result):
        assert result.n_rounds == 1 and result.converged is True

    def test_negative_latency_rejected(self):
        with pytest.raises(ValueError, match="wall_clock_s"):
            TopologyResult("out", TokenCost(1, 1), -0.5, 1, True, (), "stop")

    def test_zero_rounds_rejected(self):
        with pytest.raises(ValueError, match="n_rounds"):
            TopologyResult("out", TokenCost(1, 1), 1.0, 0, True, (), "stop")

    def test_list_trace_rejected_so_evidence_cannot_be_mutated(self):
        message = Message("generator", "x", 0, TokenCost(1, 1), 0.1)
        with pytest.raises(TypeError, match="tuple"):
            TopologyResult("out", TokenCost(1, 1), 1.0, 1, True, [message], "stop")

    def test_finish_reason_is_required(self):
        message = Message("generator", "x", 0, TokenCost(1, 1), 0.1)
        with pytest.raises(TypeError, match="finish_reason"):
            TopologyResult("x", TokenCost(1, 1), 1.0, 1, True, (message,))

    def test_trace_cannot_be_appended_to(self, result):
        assert not hasattr(result.trace, "append")


class TestMessage:
    def test_negative_round_index_rejected(self):
        with pytest.raises(ValueError, match="round_index"):
            Message("generator", "x", -1, TokenCost(1, 1), 0.1)

    def test_negative_latency_rejected(self):
        with pytest.raises(ValueError, match="latency_s"):
            Message("generator", "x", 0, TokenCost(1, 1), -0.1)
