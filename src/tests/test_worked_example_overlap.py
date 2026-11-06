"""WorkedExampleOverlap: flags outputs that copy T1's worked examples (D-110)."""

from __future__ import annotations

import re

import pytest
from agent_evals.agents.self_refine_examples import XSUM_SELF_REFINE_EXAMPLES
from agent_evals.config import ConfigError, MetricsConfig, WorkedExampleOverlapConfig
from agent_evals.metrics import WorkedExampleOverlap
from agent_evals.registry import METRICS

ARTICLE = "Tanzania beat Kenya 2-1 in the regional final on Tuesday evening."
FIRST_REVISED = XSUM_SELF_REFINE_EXAMPLES[0].revised


def _metric(ngram: int = 8) -> WorkedExampleOverlap:
    return WorkedExampleOverlap(
        MetricsConfig(device="cpu", worked_example_overlap=WorkedExampleOverlapConfig(ngram))
    )


def _flags(outputs, article=ARTICLE, ngram=8):
    rows = _metric(ngram).score(outputs, ["ref"] * len(outputs), [article] * len(outputs))
    return [row["worked_example_overlap"] for row in rows]


def test_a_copied_worked_example_summary_is_flagged():
    assert _flags([FIRST_REVISED]) == [1.0]


def test_a_summary_of_its_own_article_is_not_flagged():
    assert _flags(["Tanzania beat Kenya 2-1 in the regional final."]) == [0.0]


def test_matching_ignores_case_and_punctuation():
    assert _flags([FIRST_REVISED.upper().replace(",", " ; ")]) == [1.0]


def test_a_shared_run_shorter_than_ngram_is_not_flagged():
    seven = " ".join(re.findall(r"\w+", FIRST_REVISED)[:7])  # words as the metric counts them
    assert _flags([f"{seven} zebra quartz"]) == [0.0]
    assert _flags([f"{seven} zebra quartz"], ngram=7) == [1.0]


def test_a_run_also_in_the_outputs_own_article_is_not_flagged():
    assert _flags([FIRST_REVISED], article=f"Background. {FIRST_REVISED} More text.") == [0.0]


def test_every_worked_example_part_is_checked():
    for e in XSUM_SELF_REFINE_EXAMPLES:
        for text in (e.article, e.draft, e.revised):
            assert _flags([text]) == [1.0]


def test_registered_and_in_the_default_suite():
    assert METRICS.get("worked_example_overlap") is WorkedExampleOverlap
    assert "worked_example_overlap" in MetricsConfig().suite
    assert WorkedExampleOverlap.KEYS == ("worked_example_overlap",)


@pytest.mark.parametrize("ngram", [0, -1])
def test_ngram_must_be_positive(ngram):
    with pytest.raises(ConfigError):
        WorkedExampleOverlapConfig(ngram)
