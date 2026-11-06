"""Metric contract, ROUGE-L, artifact fetching, and the validation statistics. CPU, no network."""

from __future__ import annotations

import hashlib
import math

import pytest
from agent_evals.config import ConfigError, MetricsConfig, SummaCConfig, load_group
from agent_evals.metrics import Metric, MetricError, RougeL, remove_citations
from agent_evals.metrics.artifacts import fetch_pinned, sha256_file
from agent_evals.metrics.validation import (
    balanced_accuracy,
    build_xsumfaith,
    choose_best_threshold,
    roc_auc,
)
from agent_evals.registry import METRICS


class StubMetric(Metric):
    """Returns scripted rows so the base-class contract can be tested without a model."""

    KEYS = ("stub",)

    def __init__(self, rows=None):
        super().__init__(MetricsConfig(device="cpu"))
        self.rows = rows

    def _score_examples(self, predictions, references, evidence, gold):
        if self.rows is not None:
            return self.rows
        return [{"stub": float(len(p))} for p in predictions]


class TestMetricContract:
    def test_scores_every_example_and_averages(self):
        metric = StubMetric()
        inputs = (["ab", "abcd"], ["r", "r"], ["e", "e"])
        assert metric.score(*inputs) == [{"stub": 2.0}, {"stub": 4.0}]
        assert metric.compute(*inputs) == {"stub": 3.0}

    @pytest.mark.parametrize(
        ("predictions", "references", "evidence", "message"),
        [
            ([], [], [], "no examples"),
            (["a"], ["r", "r"], ["e"], "misaligned"),
            (["a"], ["r"], [None], "must all be strings"),
            (["fine", "   "], ["r", "r"], ["e", "e"], "blank prediction"),
            (["fine", ""], ["r", "r"], ["e", "e"], "blank prediction"),
        ],
    )
    def test_invalid_inputs_are_rejected(self, predictions, references, evidence, message):
        with pytest.raises(MetricError, match=message):
            StubMetric().score(predictions, references, evidence)

    @pytest.mark.parametrize(
        "rows",
        [
            [{"stub": math.nan}],
            [{"stub": math.inf}],
            [{"stub": 1}],
            [{"other": 1.0}],
            [{"stub": 1.0}, {"stub": 2.0}],
        ],
    )
    def test_invalid_outputs_are_rejected(self, rows):
        with pytest.raises(MetricError):
            StubMetric(rows).score(["a"], ["r"], ["e"])


class GoldEcho(StubMetric):
    """Records the gold it receives; optionally requires it."""

    def _score_examples(self, predictions, references, evidence, gold):
        self.seen = gold
        return [{"stub": 1.0} for _ in predictions]


class GoldRequired(GoldEcho):
    NEEDS_GOLD = True


class TestMetricGold:
    def test_gold_reaches_the_metric_aligned_with_predictions(self):
        metric = GoldEcho()
        metric.score(["a", "b"], ["r", "r"], ["e", "e"], ['{"x": 1}', None])
        assert metric.seen == ['{"x": 1}', None]

    def test_without_gold_the_metric_sees_none_per_example(self):
        metric = GoldEcho()
        metric.score(["a", "b"], ["r", "r"], ["e", "e"])
        assert metric.seen == [None, None]

    def test_compute_passes_gold_through(self):
        metric = GoldEcho()
        assert metric.compute(["a"], ["r"], ["e"], ['{"x": 1}']) == {"stub": 1.0}
        assert metric.seen == ['{"x": 1}']

    @pytest.mark.parametrize(
        ("gold", "message"),
        [
            (['{"x": 1}'], "misaligned"),
            ([1, None], "JSON strings or None"),
        ],
    )
    def test_invalid_gold_is_rejected(self, gold, message):
        with pytest.raises(MetricError, match=message):
            GoldEcho().score(["a", "b"], ["r", "r"], ["e", "e"], gold)

    def test_a_metric_that_needs_gold_refuses_an_example_without_it(self):
        with pytest.raises(MetricError, match="needs gold"):
            GoldRequired().score(["a", "b"], ["r", "r"], ["e", "e"], ['{"x": 1}', None])
        with pytest.raises(MetricError, match="needs gold"):
            GoldRequired().score(["a"], ["r"], ["e"])

    def test_defaults_read_no_gold_and_no_citations(self):
        assert Metric.NEEDS_GOLD is False and Metric.READS_CITATIONS is False


class TestRemoveCitations:
    """ALCE's ``remove_citations`` behaviour (utils.py, commit 246c476), kept unchanged."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Paris is the capital [1][2].", "Paris is the capital."),
            ("One [1]. Two [3][4].", "One. Two."),
            ("[2]Starts with a marker.", "Starts with a marker."),
            ("No markers here.", "No markers here."),
            ("Mid [12] sentence.", "Mid sentence."),
            ("Pipes a | b and [x] brackets.", "Pipes a b and [x brackets."),
        ],
    )
    def test_matches_alce(self, text, expected):
        assert remove_citations(text) == expected


class TestRougeL:
    def test_hand_computed_scores(self):
        # LCS of the two sentences is "the cat on the mat": 5 of 6 tokens on each side.
        row = RougeL(MetricsConfig(device="cpu")).score(
            ["the cat sat on the mat"], ["the cat is on the mat"], ["unused"]
        )[0]
        for key in RougeL.KEYS:
            assert row[key] == pytest.approx(5 / 6)

    def test_identical_and_disjoint_texts(self):
        rows = RougeL(MetricsConfig(device="cpu")).score(
            ["alpha beta", "alpha beta"], ["alpha beta", "gamma delta"], ["e", "e"]
        )
        assert rows[0]["rouge_l_f1"] == pytest.approx(1.0)
        assert rows[1]["rouge_l_f1"] == 0.0

    def test_unicode_output_is_scored(self):
        row = RougeL(MetricsConfig(device="cpu")).score(["café Zürich"], ["café Zürich"], ["e"])[0]
        assert 0.0 <= row["rouge_l_f1"] <= 1.0

    def test_registered_under_its_name(self):
        assert METRICS.get("rouge_l") is RougeL


class TestFetchPinned:
    """Uses ``file://`` URLs, which ``urllib`` serves locally, so no network is touched."""

    def _source(self, tmp_path, payload=b"weights"):
        source = tmp_path / "src" / "weights.bin"
        source.parent.mkdir()
        source.write_bytes(payload)
        return source.as_uri(), hashlib.sha256(payload).hexdigest()

    def test_downloads_verifies_and_reuses_cache(self, tmp_path):
        url, digest = self._source(tmp_path)
        cache = tmp_path / "cache"
        path = fetch_pinned(url, digest, cache)
        assert path.read_bytes() == b"weights"
        assert sha256_file(path) == digest
        (tmp_path / "src" / "weights.bin").unlink()
        assert fetch_pinned(url, digest, cache) == path

    def test_digest_mismatch_leaves_nothing_behind(self, tmp_path):
        url, _ = self._source(tmp_path)
        cache = tmp_path / "cache"
        with pytest.raises(MetricError, match="sha256"):
            fetch_pinned(url, "0" * 64, cache)
        assert list(cache.iterdir()) == []

    def test_corrupt_cache_is_replaced(self, tmp_path):
        url, digest = self._source(tmp_path)
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "weights.bin").write_bytes(b"corrupt")
        assert fetch_pinned(url, digest, cache).read_bytes() == b"weights"

    def test_missing_source_is_a_metric_error(self, tmp_path):
        with pytest.raises(MetricError, match="could not download"):
            fetch_pinned((tmp_path / "absent.bin").as_uri(), "0" * 64, tmp_path / "cache")


class TestBalancedAccuracy:
    def test_matches_mean_recall(self):
        # class 1 recall 2/3, class 0 recall 1/1
        assert balanced_accuracy([1, 1, 1, 0], [1, 1, 0, 0]) == pytest.approx((2 / 3 + 1) / 2)

    def test_misaligned_inputs_rejected(self):
        with pytest.raises(MetricError):
            balanced_accuracy([1, 0], [1])


class TestRocAuc:
    def test_perfect_and_inverted_rankings(self):
        assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
        assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0

    def test_ties_count_half(self):
        # one positive and one negative tie at 0.5, one clean separation: (1 + 0.5) / 2
        assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.1, 0.9]) == pytest.approx(0.875)

    def test_single_class_rejected(self):
        with pytest.raises(MetricError, match="both classes"):
            roc_auc([1, 1], [0.2, 0.3])


class TestThresholdSearch:
    def test_separable_scores_reach_perfect_balanced_accuracy(self):
        threshold, accuracy = choose_best_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.2)
        assert accuracy == 1.0
        assert 0.2 <= threshold < 0.8

    def test_prediction_is_strictly_above_threshold(self):
        # All scores equal: every threshold equals the score, so nothing is strictly above it.
        _, accuracy = choose_best_threshold([0, 1], [0.5, 0.5], 0.2)
        assert accuracy == 0.5


class TestXSumFaithBuilder:
    ROWS = [
        {"bbcid": "1", "system": "A", "summary": "s1", "hallucination_type": "NULL"},
        {"bbcid": "1", "system": "A", "summary": "s1", "hallucination_type": "intrinsic"},
        {"bbcid": "1", "system": "A", "summary": "s1", "hallucination_type": "NULL"},
        {"bbcid": "1", "system": "B", "summary": "s2", "hallucination_type": "extrinsic"},
        {"bbcid": "2", "system": "A", "summary": "s3", "hallucination_type": "NULL"},
    ]
    DOCUMENTS = {"1": "doc one", "2": "doc two"}

    def test_groups_alternate_between_validation_and_test(self):
        split = build_xsumfaith(self.ROWS, self.DOCUMENTS)
        assert [item["summary"] for item in split["val"]] == ["s1", "s3"]
        assert [item["summary"] for item in split["test"]] == ["s2"]

    def test_majority_null_is_consistent(self):
        split = build_xsumfaith(self.ROWS, self.DOCUMENTS)
        assert split["val"][0]["label"] == 1
        assert split["test"][0]["label"] == 0
        assert split["val"][0]["document"] == "doc one"

    def test_missing_article_is_an_error(self):
        with pytest.raises(MetricError, match="missing"):
            build_xsumfaith(self.ROWS, {"1": "doc one"})


class TestMetricsConfig:
    def test_base_yaml_matches_dataclass_defaults(self, base_config_path):
        assert load_group(base_config_path, "metrics", MetricsConfig) == MetricsConfig()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"nli_revision": "not-a-sha"},
            {"conv_checkpoint_sha256": "abc"},
            {"conv_bins": (0.0, 0.5, 0.4, 1.0)},
            {"entailment_index": 1, "contradiction_index": 1},
            {"truncation": "sideways"},
        ],
    )
    def test_invalid_summac_values_rejected(self, kwargs):
        with pytest.raises(ConfigError):
            SummaCConfig(**kwargs)


class TestLongFormQAValidation:
    """D-124: MiniCheck and AlignScore on LLM-AggreFact's long-form QA subsets, stubbed."""

    ITEMS = {
        "Lfqa": [("doc", "supported", 1), ("doc", "unsupported", 0), ("doc", "supported", 0)],
        "ClaimVerify": [("doc", "supported", 1), ("doc", "unsupported", 0)],
    }

    @pytest.fixture
    def stubbed(self, monkeypatch):
        from agent_evals.metrics import alignscore, minicheck, reproduce

        loaded = []

        def fake_load(cfg, subset=None):
            loaded.append(subset)
            return [{"document": d, "claim": c, "label": y} for d, c, y in self.ITEMS[subset]]

        class FakeMiniCheck:
            def __init__(self, cfg):
                pass

            def claim_support(self, document, claim):
                return 0.9 if claim == "supported" else 0.1

        class FakeAlignScore:
            def __init__(self, cfg):
                pass

            def pair_scores(self, documents, summaries):
                return [0.7 if s == "supported" else 0.2 for s in summaries]

        monkeypatch.setattr(reproduce, "load_aggrefact", fake_load)
        monkeypatch.setattr(minicheck, "MiniCheck", FakeMiniCheck)
        monkeypatch.setattr(alignscore, "AlignScore", FakeAlignScore)
        return reproduce, loaded

    @pytest.mark.parametrize(
        ("name", "targets"), [("minicheck_qa", (85.2, 74.6)), ("alignscore_qa", (84.5, 69.6))]
    )
    def test_each_subset_against_its_published_target(self, stubbed, name, targets):
        reproduce, loaded = stubbed
        results, details = reproduce.REPRODUCERS[name](MetricsConfig(device="cpu"), {})
        assert loaded == ["Lfqa", "ClaimVerify"]
        assert [r.dataset for r in results] == [
            "LLM-AggreFact Lfqa, test split",
            "LLM-AggreFact ClaimVerify, test split",
        ]
        assert tuple(r.target for r in results) == targets
        # Lfqa: one true positive, one false positive, one true negative -> BAcc (1 + 0.5) / 2.
        assert results[0].achieved == pytest.approx(75.0)
        assert results[1].achieved == pytest.approx(100.0)
        assert all("fixed threshold 0.5" in r.statistic for r in results)
        assert set(details) == {"Lfqa", "ClaimVerify"}

    @pytest.mark.parametrize(
        "values",
        [
            {"aggrefact_qa_subsets": ()},
            {"aggrefact_qa_subsets": ("Lfqa", "")},
            {"target_minicheck_qa_bacc": (85.2,)},
            {"target_alignscore_qa_bacc": (84.5, 69.6, 1.0)},
            {"alignscore_threshold": 1.0},
            {"alignscore_threshold": 0.0},
        ],
    )
    def test_invalid_validation_settings_raise(self, values):
        from agent_evals.config import MetricValidationConfig

        with pytest.raises(ConfigError):
            MetricValidationConfig(**values)
