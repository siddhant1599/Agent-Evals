"""Model-based metrics: hand-computed checks of the ported logic on CPU, and GPU smoke tests.

The GPU tests load the real checkpoints; run them with ``pytest -m gpu``. Agreement with the
published numbers is checked by ``src/validate_metrics.py``, not here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from agent_evals.config import MetricsConfig
from agent_evals.metrics import AlignScore, BertScore, FactCC, MetricError, MiniCheck, SummaC
from agent_evals.metrics.alignscore import aggregate_alignment, chunk_article
from agent_evals.metrics.minicheck import sentence_mean
from agent_evals.metrics.summac import (
    check_conv_weights,
    conv_histogram,
    conv_scores,
    split_sentences,
    zero_shot_score,
)
from agent_evals.registry import METRICS

BINS = np.asarray(MetricsConfig().summac.conv_bins)


class TestSummaCLogic:
    def test_short_sentences_are_dropped(self):
        text = "Too short. This sentence is long enough to keep. Tiny one."
        assert split_sentences(text, 10) == ["This sentence is long enough to keep."]

    def test_unicode_sentences_are_kept(self):
        assert split_sentences("Zürich hat 400.000 Einwohner.", 10) == [
            "Zürich hat 400.000 Einwohner."
        ]

    def test_zero_shot_is_mean_of_max_entailment_minus_max_contradiction(self):
        image = np.zeros((3, 2, 2))
        image[0] = [[0.9, 0.2], [0.1, 0.8]]
        image[1] = [[0.05, 0.1], [0.6, 0.1]]
        # columns: max entailment (0.9, 0.8), max contradiction (0.6, 0.1) -> (0.3, 0.7) -> 0.5
        assert zero_shot_score(image) == pytest.approx(0.5)

    def test_all_zero_image_scores_zero(self):
        assert zero_shot_score(np.zeros((3, 1, 1))) == 0.0

    def test_histogram_bins_pads_and_cuts_rows(self):
        image = np.zeros((3, 3, 1))
        image[0, :, 0] = [0.005, 0.5, 1.0]
        histogram = conv_histogram(image, BINS, n_rows=2)
        assert histogram.shape == (2, 23)
        assert histogram[0, 0] == 1 and histogram[0, 7] == 1 and histogram[0, 22] == 1
        assert histogram[0].sum() == 3 and histogram[1].sum() == 0
        wide = np.zeros((3, 1, 5))
        assert conv_histogram(wide, BINS, n_rows=2).shape == (2, 23)

    def test_conv_mean_over_non_empty_rows(self):
        weights = {
            "mlp.weight": torch.ones(1, 23),
            "mlp.bias": torch.zeros(1),
            "layer_final.weight": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "layer_final.bias": torch.zeros(2),
        }
        histograms = np.zeros((2, 3, 23))
        histograms[0, 0, 0] = 3  # row sums 3 and 1 -> mean 2 -> softmax([0, 2])[1]
        histograms[0, 1, 5] = 1
        scores = conv_scores(histograms, weights)
        assert scores[0] == pytest.approx(1 / (1 + math.exp(-2)))
        assert scores[1] == pytest.approx(0.5)  # no non-empty rows -> zero features

    def test_conv_weights_shape_check(self):
        good = {
            "mlp.weight": torch.zeros(1, 23),
            "mlp.bias": torch.zeros(1),
            "layer_final.weight": torch.zeros(2, 3),
            "layer_final.bias": torch.zeros(2),
        }
        assert set(check_conv_weights(good, 23)) == set(good)
        with pytest.raises(MetricError, match="shape"):
            check_conv_weights({**good, "mlp.weight": torch.zeros(1, 50)}, 23)
        with pytest.raises(MetricError, match="keys"):
            check_conv_weights({**good, "extra": torch.zeros(1)}, 23)


class TestAlignScoreLogic:
    TEXT = "One two. Three four. Five six. Seven eight."

    def test_short_article_is_one_chunk(self):
        assert chunk_article(self.TEXT, 350) == [self.TEXT]

    def test_chunking_follows_official_arithmetic(self):
        # 8 words // 3 + 1 = 3 chunks wanted; 4 sentences // 3 = 1 sentence per chunk.
        assert chunk_article(self.TEXT, 3) == [
            "One two.",
            "Three four.",
            "Five six.",
            "Seven eight.",
        ]
        # 8 words // 5 + 1 = 2 chunks wanted; 4 sentences // 2 = 2 sentences per chunk.
        assert chunk_article(self.TEXT, 5) == ["One two. Three four.", "Five six. Seven eight."]

    def test_empty_article_gives_one_empty_chunk(self):
        assert chunk_article("", 350) == [""]

    def test_max_over_chunks_then_mean_over_sentences(self):
        # chunks x sentences = [[0.2, 0.9], [0.6, 0.1]] -> max per sentence (0.6, 0.9) -> 0.75
        assert aggregate_alignment([0.2, 0.9, 0.6, 0.1], 2, 2) == pytest.approx(0.75)


class TestMiniCheckLogic:
    def test_mean_over_sentences(self):
        scores = {"The sky is blue.": 0.9, "Grass is red.": 0.1}
        assert sentence_mean("The sky is blue. Grass is red.", scores.__getitem__) == pytest.approx(
            0.5
        )

    def test_single_unicode_sentence(self):
        assert sentence_mean("Café au lait coûte 3 €.", lambda claim: 0.25) == 0.25


def test_every_metric_is_registered():
    for name, cls in [
        ("summac", SummaC),
        ("alignscore", AlignScore),
        ("minicheck", MiniCheck),
        ("factcc", FactCC),
        ("bertscore", BertScore),
    ]:
        assert METRICS.get(name) is cls


ARTICLE = (
    "The city council of Leeds voted on Tuesday to close the Kirkgate library in March. "
    "Council leader Jane Smith said the building needed repairs costing 2 million pounds. "
    "Residents have started a petition to keep the library open."
)
CONSISTENT = "Leeds council voted to close the Kirkgate library in March."
INCONSISTENT = "Leeds council voted to keep the Kirkgate library open for ten more years."
UNICODE = "Der Stadtrat von Leeds stimmte für die Schließung der Bibliothek – im März."


@pytest.mark.gpu
@pytest.mark.parametrize("cls", [SummaC, AlignScore, MiniCheck, FactCC])
def test_consistency_metrics_rank_a_contradiction_below_a_faithful_summary(cls):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    metric = cls(MetricsConfig())
    rows = metric.score([CONSISTENT, INCONSISTENT, UNICODE], ["unused"] * 3, [ARTICLE] * 3)
    for key in cls.KEYS:
        if key == "summac_zs":
            assert all(-1.0 <= row[key] <= 1.0 for row in rows)
        else:
            assert all(0.0 <= row[key] <= 1.0 for row in rows)
    # SummaC-Conv is the weakest ranker on short inputs; the other scores must separate clearly.
    main_key = "summac_zs" if cls is SummaC else cls.KEYS[0]
    assert rows[0][main_key] > rows[1][main_key]
    with pytest.raises(MetricError, match="blank"):
        metric.score([""], ["unused"], [ARTICLE])


@pytest.mark.gpu
def test_bertscore_identical_text_scores_one():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    metric = BertScore(MetricsConfig())
    rows = metric.score([CONSISTENT, UNICODE], [CONSISTENT, CONSISTENT], ["unused", "unused"])
    assert rows[0]["bertscore_f1"] == pytest.approx(1.0, abs=1e-4)
    assert rows[1]["bertscore_f1"] < rows[0]["bertscore_f1"]
