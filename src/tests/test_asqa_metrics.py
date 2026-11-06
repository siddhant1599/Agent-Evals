"""ASQA metrics (D-120, D-123, D-128): exact-match recall, citation quality, two-answer ROUGE-L and
BERTScore, and the ALCE leak check. CPU; the judge and BERTScore model are stubbed."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
from agent_evals.agents.asqa_examples import ALCE_ASQA_EXAMPLES
from agent_evals.agents.self_refine_examples import ASQA_SELF_REFINE_EXAMPLES
from agent_evals.config import CitationConfig, ConfigError, MetricsConfig, load_config
from agent_evals.datasets.asqa import format_passages, parse_passages
from agent_evals.metrics import (
    ASQASelfRefineExampleOverlap,
    ASQAWorkedExampleOverlap,
    BertScore,
    CitationQuality,
    MetricError,
    RougeL,
    StrEM,
    reference_texts,
    remove_citations,
)
from agent_evals.metrics import citation as citation_module
from agent_evals.metrics.str_em import normalize_answer
from agent_evals.registry import METRICS

CPU = MetricsConfig(device="cpu")
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _gold(short_answers=(("x",),), long_answers=("First answer.", "Second answer.")):
    return json.dumps(
        {"short_answers": [list(a) for a in short_answers], "long_answers": list(long_answers)}
    )


class TestStrEM:
    def test_alce_figure_2_example_scores_one_in_three(self):
        # Gao et al. 2023, Figure 2: one of three short answers appears in the output.
        output = (
            "The US declared independence on July 3, 1776. The Treaty of Paris was signed on "
            "September 3, 1783."
        )
        gold = _gold(short_answers=(("July 2, 1776",), ("July 4, 1776",), ("September 3, 1783",)))
        row = StrEM(CPU).score([output], ["ref"], ["evidence"], [gold])[0]
        assert row == {"str_em": pytest.approx(1 / 3)}

    def test_any_accepted_answer_counts_and_matching_is_normalized(self):
        gold = _gold(short_answers=(("Josef Bican", "Bican"), ("The Ali Daei",)))
        row = StrEM(CPU).score(["bican scored most; ALI-DAEI too."], ["r"], ["e"], [gold])[0]
        assert row["str_em"] == 0.5
        row = StrEM(CPU).score(["Bican and Ali Daei."], ["r"], ["e"], [gold])[0]
        assert row["str_em"] == 1.0

    def test_normalization_is_alces(self):
        assert normalize_answer("The  Treaty, of  Paris!") == "treaty of paris"

    def test_needs_gold_on_every_example(self):
        with pytest.raises(MetricError, match="needs gold"):
            StrEM(CPU).score(["a"], ["r"], ["e"])

    def test_registered_with_its_key(self):
        assert METRICS.get("str_em") is StrEM and StrEM.KEYS == ("str_em",)


class FakeJudge:
    """Support decided by which passages the premise holds, so Figure 3 can be replayed."""

    RULES = {
        "Statement one.": lambda titles: bool({"P1", "P2"} & titles),
        "Statement two.": lambda titles: False,
        "Statement three.": lambda titles: {"P4", "P5"} <= titles,
    }

    def __init__(self, cfg):
        self.calls = []

    def claim_support(self, document, claim):
        titles = {
            line[len("Title: ") :] for line in document.split("\n") if line.startswith("Title: ")
        }
        self.calls.append((frozenset(titles), claim))
        return 0.9 if self.RULES.get(claim, lambda t: False)(titles) else 0.1


@pytest.fixture
def citation_metric(monkeypatch):
    monkeypatch.setattr(citation_module, "MiniCheck", FakeJudge)
    return CitationQuality(CPU)


PASSAGES = format_passages([{"title": f"P{i}", "text": f"Passage {i}."} for i in range(1, 6)])


class TestCitationQuality:
    def test_alce_figure_3_example(self, citation_metric):
        # Gao et al. 2023, Figure 3: recall 2/3; precision 4/6, with [2] irrelevant in sentence 3.
        output = "Statement one [1][2]. Statement two [3]. Statement three [2][4][5]."
        row = citation_metric.score([output], ["r"], [PASSAGES])[0]
        assert row["citation_recall"] == pytest.approx(2 / 3)
        assert row["citation_precision"] == pytest.approx(4 / 6)

    def test_the_judge_sees_alces_premise_and_the_claim_without_markers(self, citation_metric):
        citation_metric.score(["Statement one [1]."], ["r"], [PASSAGES])
        assert citation_metric._judge.calls == [(frozenset({"P1"}), "Statement one.")]

    @pytest.mark.parametrize(
        ("output", "recall", "precision"),
        [
            ("Statement one.", 0.0, 0.0),  # no citation: unsupported, nothing counted
            ("Statement one [9].", 0.0, 0.0),  # out of range
            ("Statement one [0].", 0.0, 0.0),  # [0] counts as out of range
            ("Statement two [1].", 0.0, 0.0),  # cited but not entailed: one citation, not precise
        ],
    )
    def test_unsupported_sentences(self, citation_metric, output, recall, precision):
        row = citation_metric.score([output], ["r"], [PASSAGES])[0]
        assert (row["citation_recall"], row["citation_precision"]) == (recall, precision)

    def test_reads_at_most_three_citations_per_sentence(self, citation_metric):
        # [1][2][3] are read and [4] ignored; [1] and [2] each support it, [3] alone does not but
        # the others do, so [3] is irrelevant: precision 2/3.
        row = citation_metric.score(["Statement one [1][2][3][4]."], ["r"], [PASSAGES])[0]
        assert row["citation_recall"] == 1.0
        assert row["citation_precision"] == pytest.approx(2 / 3)

    def test_reads_citations_and_refuses_unreadable_passages(self, citation_metric):
        assert CitationQuality.READS_CITATIONS is True
        with pytest.raises(MetricError, match="cannot read the passages"):
            citation_metric.score(["Statement one [1]."], ["r"], ["not ALCE's format"])

    def test_parse_passages_inverts_format_passages(self):
        assert format_passages(parse_passages(PASSAGES)) == PASSAGES
        with pytest.raises(ValueError, match="passage \\[2\\]"):
            parse_passages("Document [1](Title: A): a\nDocument [3](Title: B): b")


class TestReferenceTexts:
    def test_human_answers_from_gold_else_the_reference(self):
        assert reference_texts("ref", None) == ["ref"]
        assert reference_texts("ref", '{"short_answers": [["x"]]}') == ["ref"]
        assert reference_texts("ref", _gold()) == ["First answer.", "Second answer."]

    def test_malformed_long_answers_are_refused(self):
        with pytest.raises(MetricError, match="long_answers"):
            reference_texts("ref", '{"long_answers": []}')


class TestRougeTwoAnswers:
    def test_keeps_the_answer_with_the_higher_f1(self):
        gold = _gold(long_answers=("Completely unrelated words here.", "The cat sat on the mat."))
        row = RougeL(CPU).score(["The cat sat on the mat."], ["unused"], ["e"], [gold])[0]
        assert row["rouge_l_f1"] == 1.0

    def test_without_gold_the_reference_is_used(self):
        row = RougeL(CPU).score(["The cat sat on the mat."], ["The cat sat on the mat."], ["e"])[0]
        assert row["rouge_l_f1"] == 1.0


class FakeBertScorer:
    """F1 is 1.0 when candidate and reference are equal, else 0.5; P and R mark the pair."""

    def __init__(self):
        self.calls = []

    def score(self, candidates, references):
        self.calls.append((list(candidates), list(references)))
        f1 = [1.0 if c == r else 0.5 for c, r in zip(candidates, references, strict=True)]
        index = [float(i) for i in range(len(candidates))]
        return torch.tensor(index), torch.tensor(index), torch.tensor(f1)


def _bertscore():
    metric = object.__new__(BertScore)
    metric._scorer = FakeBertScorer()
    return metric


class TestBertScoreTwoAnswers:
    def test_scores_every_answer_and_keeps_precision_recall_f1_of_the_best(self):
        metric = _bertscore()
        rows = metric._score_examples(
            ["a", "b"], ["ref", "ref"], ["e", "e"], [_gold(long_answers=("x", "a")), None]
        )
        assert metric._scorer.calls == [(["a", "a", "b"], ["x", "a", "ref"])]
        assert rows[0] == {"bertscore_precision": 1.0, "bertscore_recall": 1.0, "bertscore_f1": 1.0}
        assert rows[1]["bertscore_precision"] == 2.0 and rows[1]["bertscore_f1"] == 0.5

    def test_on_a_tie_the_later_answer_is_kept(self):
        metric = _bertscore()
        row = metric._score_examples(["a"], ["ref"], ["e"], [_gold(long_answers=("x", "y"))])[0]
        assert row["bertscore_precision"] == 1.0

    def test_tier1_calls_the_scorer_exactly_as_before(self):
        metric = _bertscore()
        metric._score_examples(["a", "b"], ["r1", "r2"], ["e", "e"], [None, None])
        assert metric._scorer.calls == [(["a", "b"], ["r1", "r2"])]


class TestASQALeakCheck:
    def test_copying_an_alce_example_answer_is_flagged(self):
        copied = " ".join(remove_citations(ALCE_ASQA_EXAMPLES[2].answer).split()[:10])
        row = ASQAWorkedExampleOverlap(CPU).score([copied], ["r"], [PASSAGES])[0]
        assert row == {"worked_example_overlap": 1.0}

    def test_copying_an_example_passage_is_flagged(self):
        title, text = ALCE_ASQA_EXAMPLES[0].passages[2]
        copied = " ".join(text.split()[:12])
        row = ASQAWorkedExampleOverlap(CPU).score([copied], ["r"], [PASSAGES])[0]
        assert row["worked_example_overlap"] == 1.0

    def test_text_also_in_the_questions_own_passages_is_not_flagged(self):
        copied = " ".join(remove_citations(ALCE_ASQA_EXAMPLES[2].answer).split()[:10])
        own = format_passages([{"title": "T", "text": copied}])
        row = ASQAWorkedExampleOverlap(CPU).score([copied], ["r"], [own])[0]
        assert row["worked_example_overlap"] == 0.0

    def test_an_original_answer_is_not_flagged(self):
        row = ASQAWorkedExampleOverlap(CPU).score(["Passage 1 says so [1]."], ["r"], [PASSAGES])[0]
        assert row["worked_example_overlap"] == 0.0

    def test_registered_under_its_own_name_with_the_shared_key(self):
        assert METRICS.get("asqa_worked_example_overlap") is ASQAWorkedExampleOverlap
        assert ASQAWorkedExampleOverlap.KEYS == ("worked_example_overlap",)


class TestASQASelfRefineLeakCheck:
    def _score(self, output, evidence=PASSAGES):
        return ASQASelfRefineExampleOverlap(CPU).score([output], ["r"], [evidence])[0]

    def test_copying_a_draft_is_flagged_under_its_own_key(self):
        draft = remove_citations(ASQA_SELF_REFINE_EXAMPLES[0].draft)
        assert self._score(" ".join(draft.split()[:10])) == {"self_refine_example_overlap": 1.0}

    def test_copying_feedback_is_flagged(self):
        feedback = remove_citations(ASQA_SELF_REFINE_EXAMPLES[1].feedback)
        copied = " ".join(feedback.split("\n")[1].split()[-12:])
        assert self._score(copied)["self_refine_example_overlap"] == 1.0

    def test_copying_the_written_revision_is_flagged(self):
        revised = remove_citations(ASQA_SELF_REFINE_EXAMPLES[2].revised)
        assert self._score(" ".join(revised.split()[-12:]))["self_refine_example_overlap"] == 1.0

    def test_alces_own_answers_are_left_to_the_alce_check(self):
        # A clause of ALCE's answer that no draft or feedback repeats; drafts built from an ALCE
        # answer share some of its phrases, and copying those is flagged by both checks.
        answer = ASQA_SELF_REFINE_EXAMPLES[1].qa.answer
        copied = answer[answer.index("although") :].split("Congress")[0]
        assert self._score(copied)["self_refine_example_overlap"] == 0.0
        alce = ASQAWorkedExampleOverlap(CPU).score([copied], ["r"], [PASSAGES])[0]
        assert alce["worked_example_overlap"] == 1.0

    def test_text_also_in_the_questions_own_passages_is_not_flagged(self):
        draft = remove_citations(ASQA_SELF_REFINE_EXAMPLES[0].draft)
        copied = " ".join(draft.split()[:10])
        own = format_passages([{"title": "T", "text": copied}])
        assert self._score(copied, own)["self_refine_example_overlap"] == 0.0

    def test_an_original_answer_is_not_flagged(self):
        assert self._score("Passage 1 says so [1].")["self_refine_example_overlap"] == 0.0

    def test_registered_under_its_own_name_and_key_and_in_the_asqa_suite(self):
        assert METRICS.get("asqa_self_refine_example_overlap") is ASQASelfRefineExampleOverlap
        assert ASQASelfRefineExampleOverlap.KEYS == ("self_refine_example_overlap",)
        cfg = load_config(CONFIGS / "data" / "tier2_asqa.yaml", ["run.owner=sameer"])
        suite = cfg.metrics.suite
        assert suite[-2:] == ("asqa_worked_example_overlap", "asqa_self_refine_example_overlap")


class TestCitationConfig:
    def test_default_is_alces_three(self):
        assert CitationConfig().max_citations_per_sentence == 3

    @pytest.mark.parametrize("value", [0, -1, True, 2.5])
    def test_invalid_values_raise(self, value):
        with pytest.raises(ConfigError):
            CitationConfig(max_citations_per_sentence=value)


def test_the_stub_judge_has_minichecks_claim_support_signature():
    from agent_evals.metrics.minicheck import MiniCheck

    real = inspect.signature(MiniCheck.claim_support).parameters
    stub = inspect.signature(FakeJudge.claim_support).parameters
    assert list(real) == list(stub) == ["self", "document", "claim"]
