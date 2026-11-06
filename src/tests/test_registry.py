"""Registry behaviour, including the silent-empty-registry failure mode."""

from __future__ import annotations

import pytest
from agent_evals.datasets import (
    ASQATrainAdapter,
    CnnDailyMailAdapter,
    SAMSumAdapter,
    XSumAdapter,
)
from agent_evals.registry import (
    ALL_REGISTRIES,
    DATASETS,
    REWARDS,
    SERVING,
    Registry,
    RegistryError,
)
from agent_evals.rl import (
    AlignScoreReward,
    CitationPrecisionReward,
    CitationRecallReward,
    ExactMatchRecallReward,
)
from agent_evals.serving import VLLMChatClient


class _Thing:
    def __init__(self, value: int = 0) -> None:
        self.value = value


@pytest.fixture
def registry() -> Registry:
    return Registry("widget")


class TestRegistration:
    def test_register_then_get_round_trips(self, registry):
        registry.register("thing")(_Thing)
        assert registry.get("thing") is _Thing

    def test_decorator_returns_the_class_unchanged(self, registry):
        returned = registry.register("thing")(_Thing)
        assert returned is _Thing

    def test_build_constructs_an_instance_with_arguments(self, registry):
        registry.register("thing")(_Thing)
        assert registry.build("thing", value=7).value == 7

    def test_duplicate_name_raises_rather_than_depending_on_import_order(self, registry):
        registry.register("thing")(_Thing)

        class Other:
            pass

        with pytest.raises(RegistryError, match="already registered"):
            registry.register("thing")(Other)


class TestLookupFailures:
    def test_unknown_name_raises(self, registry):
        with pytest.raises(RegistryError, match="unknown widget"):
            registry.get("nope")

    def test_error_lists_available_names(self, registry):
        registry.register("alpha")(_Thing)
        registry.register("beta")(_Thing)
        with pytest.raises(RegistryError, match="alpha, beta"):
            registry.get("nope")

    def test_empty_registry_error_hints_at_a_missing_import(self, registry):
        with pytest.raises(RegistryError, match="never imported"):
            registry.get("nope")


class TestIntrospection:
    def test_names_are_sorted(self, registry):
        registry.register("zulu")(_Thing)
        registry.register("alpha")(_Thing)
        assert registry.names() == ("alpha", "zulu")

    def test_contains_and_len(self, registry):
        registry.register("thing")(_Thing)
        assert "thing" in registry and "other" not in registry
        assert len(registry) == 1


class TestTemporary:
    def test_entry_is_visible_inside_and_gone_after(self, registry):
        with registry.temporary("thing", _Thing):
            assert registry.get("thing") is _Thing
        assert "thing" not in registry

    def test_entry_is_removed_even_when_the_block_raises(self, registry):
        with pytest.raises(RuntimeError), registry.temporary("thing", _Thing):
            raise RuntimeError("boom")
        assert "thing" not in registry

    def test_cannot_shadow_a_registered_name(self, registry):
        registry.register("thing")(_Thing)
        with (
            pytest.raises(RegistryError, match="already registered"),
            registry.temporary("thing", _Thing),
        ):
            pass

    def test_a_permanent_entry_registered_inside_survives(self, registry):
        with registry.temporary("temp", _Thing):
            registry.register("permanent")(_Thing)
        assert registry.names() == ("permanent",)


class TestTheSixRegistries:
    def test_there_are_exactly_six(self):
        assert len(ALL_REGISTRIES) == 6

    def test_each_has_a_distinct_kind(self):
        kinds = [r.kind for r in ALL_REGISTRIES]
        assert len(set(kinds)) == 6

    def test_shipped_implementations_are_registered(self):
        assert DATASETS.names() == (
            "tier1_cnndm",
            "tier1_samsum",
            "tier1_xsum",
            "tier2_asqa",
            "tier2_asqa_train",
        )
        assert DATASETS.get("tier1_cnndm") is CnnDailyMailAdapter
        assert DATASETS.get("tier1_samsum") is SAMSumAdapter
        assert DATASETS.get("tier1_xsum") is XSumAdapter
        assert SERVING.names() == ("vllm_openai",)
        assert SERVING.get("vllm_openai") is VLLMChatClient
        assert REWARDS.names() == ("alignscore", "citation_precision", "citation_recall", "str_em")
        assert REWARDS.get("alignscore") is AlignScoreReward
        assert REWARDS.get("str_em") is ExactMatchRecallReward
        assert REWARDS.get("citation_recall") is CitationRecallReward
        assert REWARDS.get("citation_precision") is CitationPrecisionReward
        assert DATASETS.get("tier2_asqa_train") is ASQATrainAdapter
