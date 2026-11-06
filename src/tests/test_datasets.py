"""Offline dataset contracts and sampling, using only an eight-row synthetic raw split."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset
from agent_evals.config import ConfigError, DataConfig, LoggingConfig
from agent_evals.datasets import (
    CnnDailyMailAdapter,
    DatasetAdapter,
    DatasetError,
    SAMSumAdapter,
    XSumAdapter,
)
from agent_evals.datasets import base as base_module
from agent_evals.logging_utils import configure_logging
from agent_evals.registry import DATASETS
from agent_evals.seeding import set_seed

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "xsum.json"
CNNDM_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cnndm.json"
SAMSUM_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "samsum.json"


class FixtureXSum(XSumAdapter):
    """Inject a raw split without patching HuggingFace or accessing the network."""

    def __init__(self, cfg: DataConfig, raw: Dataset) -> None:
        super().__init__(cfg)
        self._raw = raw

    def _load_raw(self) -> Dataset:
        return self._raw


class FixtureTier3(FixtureXSum):
    """Intentionally inherit trainable declarations to test the unconditional Tier 3 veto."""

    TIER = 3


class FixtureEvaluationOnly(FixtureXSum):
    TRAINABLE_SPLITS = frozenset()


class FixtureCnnDm(CnnDailyMailAdapter):
    """CNN/DailyMail's mapping over an offline raw split."""

    def __init__(self, cfg: DataConfig, raw: Dataset) -> None:
        super().__init__(cfg)
        self._raw = raw

    def _load_raw(self) -> Dataset:
        return self._raw


class FixtureSAMSum(SAMSumAdapter):
    """SAMSum's mapping over an offline raw split."""

    def __init__(self, cfg: DataConfig, raw: Dataset) -> None:
        super().__init__(cfg)
        self._raw = raw

    def _load_raw(self) -> Dataset:
        return self._raw


@pytest.fixture
def raw():
    return Dataset.from_list(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


@pytest.fixture
def data_cfg():
    return DataConfig(n_samples=5, min_source_words=1)


@pytest.fixture
def frozen_cfg(data_cfg, raw):
    ids = FixtureXSum(data_cfg, raw).draw_sample_ids()
    return replace(data_cfg, sample_ids=ids)


class TestSchema:
    def test_base_is_abstract(self, data_cfg):
        with pytest.raises(TypeError, match="abstract"):
            DatasetAdapter(data_cfg)

    def test_xsum_registered_on_package_import(self):
        assert DATASETS.get("tier1_xsum") is XSumAdapter

    def test_mapping_exact_schema_and_example_round_trip(self, frozen_cfg, raw):
        adapter = FixtureXSum(frozen_cfg, raw)
        result = adapter.build()
        assert set(result.column_names) == {
            "id",
            "tier",
            "task",
            "source",
            "evidence",
            "question",
            "gold",
            "reference",
            "source_words",
            "truncated",
        }
        original = {row["id"]: row for row in raw}
        for row in result:
            assert row["tier"] == 1 and row["task"] == "summarization"
            assert row["source"] == row["evidence"] == original[row["id"]]["document"]
            assert row["reference"] == original[row["id"]]["summary"]
            assert row["question"] is None and row["truncated"] is False
            assert row["gold"] is None
            assert row == asdict(adapter.as_example(row))
        assert result.features["question"].dtype == "string"
        assert result.features["gold"].dtype == "string"

    def test_empty_and_unicode_sources_and_no_truncation(self, data_cfg, raw):
        ids = ("FAKE_EXAMPLE_empty", "FAKE_EXAMPLE_b", "FAKE_EXAMPLE_g")
        cfg = replace(data_cfg, min_source_words=0, n_samples=len(ids), sample_ids=ids)
        rows = FixtureXSum(cfg, raw).build()
        assert rows[0]["source_words"] == 0
        assert rows[1]["source_words"] == 2 and rows[1]["source"] == "café\t東京"
        assert rows[2]["source"] == raw[7]["document"]
        assert not any(rows["truncated"])

    @pytest.mark.parametrize(
        "field,value",
        [
            ("id", ""),
            ("document", None),
            ("summary", None),
        ],
    )
    def test_invalid_raw_data_rejected(self, data_cfg, raw, field, value):
        rows = raw.to_list()
        rows[1][field] = value
        with pytest.raises(DatasetError):
            FixtureXSum(data_cfg, Dataset.from_list(rows)).draw_sample_ids()

    def test_duplicate_raw_ids_rejected(self, data_cfg, raw):
        rows = raw.to_list()
        rows[1]["id"] = rows[2]["id"]
        with pytest.raises(DatasetError, match="duplicate raw"):
            FixtureXSum(data_cfg, Dataset.from_list(rows)).draw_sample_ids()

    @pytest.mark.parametrize(
        "change",
        [
            {"source_words": -1},
            {"tier": 2},
            {"task": "wrong"},
            {"extra": "not allowed"},
        ],
    )
    def test_as_example_rejects_corrupted_contract(self, frozen_cfg, raw, change):
        adapter = FixtureXSum(frozen_cfg, raw)
        row = adapter.build()[0]
        row.update(change)
        with pytest.raises(DatasetError):
            adapter.as_example(row)


class TestFiltering:
    def test_inclusive_bounds_and_before_after_log(self, data_cfg, raw, capsys):
        ids = tuple(raw[index]["id"] for index in range(2, 7))
        cfg = replace(data_cfg, min_source_words=2, max_source_words=6, sample_ids=ids)
        configure_logging(LoggingConfig())
        result = FixtureXSum(cfg, raw).build()
        assert sorted(result["source_words"]) == [2, 3, 4, 5, 6]
        assert "8 → 5" in capsys.readouterr().err

    def test_filter_precedes_quantiles(self, data_cfg, raw, capsys):
        cfg = replace(data_cfg, min_source_words=2, max_source_words=6, n_samples=3)
        configure_logging(LoggingConfig())
        FixtureXSum(cfg, raw).draw_sample_ids()
        log = capsys.readouterr().err
        assert "stratum sizes=[2, 1, 2]" in log

    def test_empty_filtered_pool_fails(self, data_cfg, raw):
        with pytest.raises(DatasetError, match="no examples remain"):
            FixtureXSum(replace(data_cfg, min_source_words=8), raw).draw_sample_ids()


class TestSampling:
    def test_ties_enter_shorter_bucket_and_remainder_enters_earlier_buckets(self, data_cfg, raw):
        # Filtered lengths 1..7 have exact boundaries at 3 and 5; quotas must be 2, 2, 1.
        ids = FixtureXSum(data_cfg, raw).draw_sample_ids()
        by_id = {row["id"]: len(row["document"].split()) for row in raw}
        groups = Counter(0 if by_id[x] <= 3 else 1 if by_id[x] <= 5 else 2 for x in ids)
        assert groups == {0: 2, 1: 2, 2: 1}
        assert "FAKE_EXAMPLE_e" in ids  # Boundary length 5 belongs to the middle bucket.

    def test_tied_lengths_are_never_split_across_buckets(self, data_cfg, raw):
        rows = raw.to_list()
        rows[3]["document"] = rows[2]["document"]
        cfg = replace(data_cfg, min_source_words=0, n_samples=3)
        adapter = FixtureXSum(cfg, Dataset.from_list(rows))
        ids = adapter.draw_sample_ids()
        by_id = {row["id"]: len(row["document"].split()) for row in rows}
        lengths = [by_id[x] for x in ids]
        assert lengths[0] <= 2 and lengths[1] == 4 and lengths[2] > 4

    def test_same_seed_same_ids_across_calls_instances_and_raw_order(self, data_cfg, raw):
        adapter = FixtureXSum(data_cfg, raw)
        expected = adapter.draw_sample_ids()
        assert adapter.draw_sample_ids() == expected
        reversed_raw = raw.select(list(reversed(range(len(raw)))))
        assert FixtureXSum(data_cfg, reversed_raw).draw_sample_ids() == expected

    def test_private_sampling_does_not_change_generation_state(self, data_cfg, raw):
        expected = None
        for generation_seed in (0, 1, 2):
            set_seed(generation_seed)
            python_state, numpy_state = random.getstate(), np.random.get_state()
            torch_state = torch.get_rng_state().clone()
            actual = FixtureXSum(data_cfg, raw).draw_sample_ids()
            if expected is None:
                expected = actual
            assert actual == expected
            assert random.getstate() == python_state
            after = np.random.get_state()
            assert after[0] == numpy_state[0] and after[2:] == numpy_state[2:]
            np.testing.assert_array_equal(after[1], numpy_state[1])
            assert torch.equal(torch.get_rng_state(), torch_state)

    def test_unfrozen_build_does_not_load_or_sample(self, data_cfg, raw):
        class NoLoad(FixtureXSum):
            def _load_raw(self):
                pytest.fail("unfrozen build must fail before loading")

        with pytest.raises(DatasetError, match="has not been frozen"):
            NoLoad(data_cfg, raw).build()

    def test_build_replays_authoritative_id_order_without_sampling(self, frozen_cfg, raw):
        class NoSampling(FixtureXSum):
            def draw_sample_ids(self):
                pytest.fail("build must never sample")

        cfg = replace(frozen_cfg, sample_ids=tuple(reversed(frozen_cfg.sample_ids)))
        adapter = NoSampling(cfg, raw)
        assert tuple(adapter.build()["id"]) == cfg.sample_ids
        assert tuple(adapter.build()["id"]) == cfg.sample_ids

    def test_frozen_preparation_refused(self, frozen_cfg, raw):
        with pytest.raises(DatasetError, match="already frozen"):
            FixtureXSum(frozen_cfg, raw).draw_sample_ids()

    def test_missing_frozen_id_fails_instead_of_replacement(self, frozen_cfg, raw):
        ids = ("FAKE_EXAMPLE_missing", *frozen_cfg.sample_ids[1:])
        with pytest.raises(DatasetError, match="missing or excluded"):
            FixtureXSum(replace(frozen_cfg, sample_ids=ids), raw).build()

    def test_filtered_frozen_id_fails(self, data_cfg, raw):
        ids = ("FAKE_EXAMPLE_a", "FAKE_EXAMPLE_b", "FAKE_EXAMPLE_c")
        cfg = replace(data_cfg, n_samples=3, sample_ids=ids, min_source_words=2)
        with pytest.raises(DatasetError, match="missing or excluded"):
            FixtureXSum(cfg, raw).build()

    def test_insufficient_stratum_refuses_to_adjust_sample(self, data_cfg, raw):
        with pytest.raises(DatasetError, match="stratum 2.*requires 3"):
            FixtureXSum(replace(data_cfg, n_samples=8), raw).draw_sample_ids()

    def test_constant_lengths_refuse_empty_strata(self, data_cfg, raw):
        rows = raw.to_list()
        for row in rows:
            row["document"] = "same length"
        with pytest.raises(DatasetError, match="stratum 2"):
            FixtureXSum(data_cfg, Dataset.from_list(rows)).draw_sample_ids()

    def test_number_of_strata_is_configured(self, data_cfg, raw):
        cfg = replace(data_cfg, n_strata=1, n_samples=7)
        assert len(FixtureXSum(cfg, raw).draw_sample_ids()) == 7


class FixtureXSumWithTest(FixtureXSum):
    """Also serves a test split, so sources shared with it can be excluded offline."""

    def __init__(self, cfg: DataConfig, raw: Dataset, test_raw: Dataset) -> None:
        super().__init__(cfg, raw)
        self._test_raw = test_raw

    def _load_split(self, split: str) -> Dataset:
        assert split == "test"
        return self._test_raw


class TestExcludeSourcesFromSplit:
    def test_a_source_also_in_the_test_split_is_dropped(self, data_cfg, raw):
        cfg = replace(data_cfg, split="train", exclude_sources_from_split="test")
        shared = next(row for row in raw.to_list() if row["id"] == "FAKE_EXAMPLE_c")
        test_raw = Dataset.from_list([dict(shared, id="FAKE_EXAMPLE_test_copy")])
        pool = FixtureXSumWithTest(cfg, raw, test_raw).eligible_pool()
        unfiltered = FixtureXSum(replace(cfg, exclude_sources_from_split=None), raw)
        assert "FAKE_EXAMPLE_c" not in pool["id"]
        assert len(pool) == len(unfiltered.eligible_pool()) - 1

    def test_no_shared_source_drops_nothing(self, data_cfg, raw):
        cfg = replace(data_cfg, split="train", exclude_sources_from_split="test")
        test_raw = Dataset.from_list([dict(raw[1], id="FAKE_EXAMPLE_x", document="Unseen text.")])
        pool = FixtureXSumWithTest(cfg, raw, test_raw).eligible_pool()
        assert len(pool) == len(
            FixtureXSum(replace(cfg, exclude_sources_from_split=None), raw).eligible_pool()
        )

    @pytest.mark.parametrize("value", ["", "train"])
    def test_invalid_split_is_refused(self, value):
        with pytest.raises(ConfigError, match="exclude_sources_from_split"):
            DataConfig(split="train", exclude_sources_from_split=value)


class TestTrainability:
    def test_xsum_training_accepted(self, data_cfg, raw):
        FixtureXSum(replace(data_cfg, split="train"), raw).assert_trainable("train")

    @pytest.mark.parametrize("split", ["test", "validation"])
    def test_evaluation_splits_refused(self, data_cfg, raw, split):
        with pytest.raises(DatasetError, match="not trainable"):
            FixtureXSum(replace(data_cfg, split=split), raw).assert_trainable(split)

    def test_cannot_claim_a_different_split(self, data_cfg, raw):
        with pytest.raises(DatasetError, match="differs from loaded split"):
            FixtureXSum(data_cfg, raw).assert_trainable("train")

    def test_tier3_refused_even_with_trainable_declarations(self, data_cfg, raw):
        adapter = FixtureTier3(replace(data_cfg, split="train"), raw)
        with pytest.raises(DatasetError, match="Tier 3 is never trainable"):
            adapter.assert_trainable("train")

    def test_evaluation_only_dataset_refuses_training(self, data_cfg, raw):
        adapter = FixtureEvaluationOnly(replace(data_cfg, split="train"), raw)
        with pytest.raises(DatasetError, match="not trainable"):
            adapter.assert_trainable("train")


@pytest.fixture
def cnndm_raw():
    return Dataset.from_list(json.loads(CNNDM_FIXTURE_PATH.read_text(encoding="utf-8")))


@pytest.fixture
def cnndm_cfg():
    return DataConfig(name="tier1_cnndm", n_samples=7, n_strata=1, min_source_words=1)


class TestCnnDailyMail:
    def test_registered_on_package_import(self):
        assert DATASETS.get("tier1_cnndm") is CnnDailyMailAdapter

    def test_mapping_keeps_highlights_verbatim(self, cnndm_cfg, cnndm_raw):
        cfg = replace(cnndm_cfg, n_samples=8)
        ids = FixtureCnnDm(cfg, cnndm_raw).draw_sample_ids()
        adapter = FixtureCnnDm(replace(cfg, sample_ids=ids), cnndm_raw)
        original = {row["id"]: row for row in cnndm_raw}
        rows = adapter.build()
        assert len(rows) == 8
        for row in rows:
            assert row["tier"] == 1 and row["task"] == "summarization"
            assert row["source"] == row["evidence"] == original[row["id"]]["article"]
            assert row["reference"] == original[row["id"]]["highlights"]
            assert row["question"] is None and row["truncated"] is False
            assert row == asdict(adapter.as_example(row))
        by_id = {row["id"]: row for row in rows}
        assert by_id["FAKE_EXAMPLE_cnn_c"]["reference"] == "Unicode words.\nCafé in 東京."

    @pytest.mark.parametrize("split", ["train", "validation", "test"])
    def test_no_split_is_trainable(self, cnndm_cfg, cnndm_raw, split):
        adapter = FixtureCnnDm(replace(cnndm_cfg, split=split), cnndm_raw)
        with pytest.raises(DatasetError, match="not trainable"):
            adapter.assert_trainable(split)


@pytest.fixture
def samsum_raw():
    return Dataset.from_list(json.loads(SAMSUM_FIXTURE_PATH.read_text(encoding="utf-8")))


@pytest.fixture
def samsum_cfg():
    return DataConfig(name="tier1_samsum", n_samples=8, n_strata=1, min_source_words=1)


class TestSAMSum:
    def test_registered_on_package_import(self):
        assert DATASETS.get("tier1_samsum") is SAMSumAdapter

    def test_mapping_keeps_dialogue_and_summary_verbatim(self, samsum_cfg, samsum_raw):
        ids = FixtureSAMSum(samsum_cfg, samsum_raw).draw_sample_ids()
        adapter = FixtureSAMSum(replace(samsum_cfg, sample_ids=ids), samsum_raw)
        original = {row["id"]: row for row in samsum_raw}
        rows = adapter.build()
        assert len(rows) == 8
        for row in rows:
            assert row["tier"] == 1 and row["task"] == "summarization"
            assert row["source"] == row["evidence"] == original[row["id"]]["dialogue"]
            assert row["reference"] == original[row["id"]]["summary"]
            assert row["question"] is None and row["truncated"] is False
            assert row == asdict(adapter.as_example(row))
        by_id = {row["id"]: row for row in rows}
        assert (
            by_id["FAKE_EXAMPLE_sam_c"]["source"]
            == "Gus: Café at noon?\nHal: Ja, 東京 time or ours? 🙂"
        )
        assert by_id["FAKE_EXAMPLE_sam_d"]["source"].count("<file_gif>") == 1

    def test_suffixed_ids_stay_strings(self, samsum_cfg, samsum_raw):
        ids = FixtureSAMSum(samsum_cfg, samsum_raw).draw_sample_ids()
        assert "FAKE_EXAMPLE_sam_b-1" in ids
        assert all(isinstance(identifier, str) for identifier in ids)

    @pytest.mark.parametrize("split", ["train", "validation", "test"])
    def test_no_split_is_trainable(self, samsum_cfg, samsum_raw, split):
        adapter = FixtureSAMSum(replace(samsum_cfg, split=split), samsum_raw)
        with pytest.raises(DatasetError, match="not trainable"):
            adapter.assert_trainable(split)


class TestDuplicateSources:
    def test_lowest_id_kept_whatever_the_raw_order(self, cnndm_cfg, cnndm_raw):
        cfg = replace(cnndm_cfg, dedupe_sources=True)
        reversed_raw = cnndm_raw.select(list(reversed(range(len(cnndm_raw)))))
        for raw_split in (cnndm_raw, reversed_raw):
            ids = FixtureCnnDm(cfg, raw_split).draw_sample_ids()
            assert "FAKE_EXAMPLE_cnn_b" in ids and "FAKE_EXAMPLE_cnn_d" not in ids
            assert len(ids) == 7

    def test_removal_is_logged_and_precedes_length_filter(self, cnndm_cfg, cnndm_raw, capsys):
        cfg = replace(cnndm_cfg, dedupe_sources=True, min_source_words=2, n_samples=6)
        configure_logging(LoggingConfig())
        FixtureCnnDm(cfg, cnndm_raw).draw_sample_ids()
        log = capsys.readouterr().err
        removal = log.index("duplicate-source removal: 8 → 7")
        assert removal < log.index("source-word filtering: 7 → 6")

    def test_duplicates_kept_when_disabled(self, cnndm_cfg, cnndm_raw, capsys):
        cfg = replace(cnndm_cfg, n_samples=8)
        configure_logging(LoggingConfig())
        ids = FixtureCnnDm(cfg, cnndm_raw).draw_sample_ids()
        assert {"FAKE_EXAMPLE_cnn_b", "FAKE_EXAMPLE_cnn_d"} <= set(ids)
        assert "duplicate-source removal" not in capsys.readouterr().err

    def test_frozen_duplicate_id_fails_once_removal_is_enabled(self, cnndm_cfg, cnndm_raw):
        ids = ("FAKE_EXAMPLE_cnn_d",)
        cfg = replace(cnndm_cfg, dedupe_sources=True, n_samples=1, sample_ids=ids)
        with pytest.raises(DatasetError, match="missing or excluded"):
            FixtureCnnDm(cfg, cnndm_raw).build()


class TestHubLoading:
    """The shared Hub loader, with ``load_dataset`` replaced so no test reaches the network."""

    @pytest.fixture
    def calls(self, monkeypatch, cnndm_raw):
        recorded = []

        def fake_load_dataset(path, **kwargs):
            recorded.append((path, kwargs))
            return cnndm_raw

        monkeypatch.setattr(base_module, "load_dataset", fake_load_dataset)
        return recorded

    def test_pinned_arguments_and_config_name_are_passed(self, cnndm_cfg, calls):
        cfg = replace(cnndm_cfg, hf_path="FAKE_EXAMPLE/repo", hf_config="FAKE_config")
        CnnDailyMailAdapter(cfg).draw_sample_ids()
        assert calls == [
            (
                "FAKE_EXAMPLE/repo",
                {
                    "name": "FAKE_config",
                    "revision": cfg.hf_revision,
                    "split": "test",
                    "token": False,
                },
            )
        ]

    def test_single_config_repositories_pass_no_config_name(self, data_cfg, calls):
        cfg = replace(data_cfg, n_samples=3, n_strata=1)
        with pytest.raises(KeyError):  # The CNN/DM fixture lacks XSum's columns.
            XSumAdapter(cfg).draw_sample_ids()
        assert calls[0][1]["name"] is None

    def test_non_dataset_result_rejected(self, cnndm_cfg, monkeypatch):
        monkeypatch.setattr(base_module, "load_dataset", lambda path, **kwargs: {"test": None})
        with pytest.raises(DatasetError, match="non-streaming"):
            CnnDailyMailAdapter(cnndm_cfg).draw_sample_ids()
