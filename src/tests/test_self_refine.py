"""T1 Self-Refine: the self-feedback role, the Generator's refine step, the worked examples, and
the T1 loop end to end against a scripted client. CPU, offline."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest
import yaml
from agent_evals.agents import AgentError, Generator, SelfFeedback
from agent_evals.agents import generator as generator_module
from agent_evals.agents import self_feedback as self_feedback_module
from agent_evals.agents.generator import XSUM_SELF_REFINE
from agent_evals.agents.self_feedback import XSUM_SELF_FEEDBACK
from agent_evals.agents.self_refine_examples import XSUM_SELF_REFINE_EXAMPLES, FewShotExample
from agent_evals.config import AgentConfig, SamplingConfig, ServingConfig, TopologyConfig
from agent_evals.registry import AGENTS, TOPOLOGIES
from agent_evals.serving import BackboneSpec, ChatClient, Completion
from agent_evals.topologies import SelfRefine, TopologyError
from agent_evals.types import AgentState, Example, Message, TokenCost

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SOURCE = "The council voted to close the {old} library in Zürich."
SEP = "\n\n###\n\n"

BACKBONE = BackboneSpec(
    name="fake-backbone",
    hf_path="org/fake",
    hf_revision="a" * 40,
    gated=False,
    quantization="bitsandbytes",
    max_model_len=2809,
)

FAKE_EXAMPLES = tuple(
    FewShotExample(**row)
    for row in json.loads((FIXTURES / "self_refine_fake_examples.json").read_text("utf-8"))
)


class ScriptedClient(ChatClient):
    """Returns the scripted replies in order and records every prompt, seed and set of turns.

    Call ``n`` (from 1) costs ``TokenCost(100 * n, n)``, so totals show every call was counted.
    """

    def __init__(self, replies, finish_reasons=None):
        super().__init__(ServingConfig(), SamplingConfig(), BACKBONE)
        self.replies = list(replies)
        self.finish_reasons = list(finish_reasons or ["stop"] * len(self.replies))
        self.calls: list[tuple[str, int | None]] = []
        self.turns: list[tuple[tuple[str, str], ...]] = []

    def complete(self, prompt, seed=None, turns=()):
        self.calls.append((prompt, seed))
        self.turns.append(tuple(turns))
        n = len(self.calls)
        return Completion(
            self.replies[n - 1], TokenCost(100 * n, n), 0.1, self.finish_reasons[n - 1]
        )


@pytest.fixture
def fake_templates(monkeypatch):
    """Swap the real worked examples for the FAKE_EXAMPLE fixtures in both XSum templates."""
    monkeypatch.setitem(
        self_feedback_module.FEEDBACK_TEMPLATES,
        "tier1_xsum",
        dataclasses.replace(XSUM_SELF_FEEDBACK, examples=FAKE_EXAMPLES),
    )
    monkeypatch.setitem(
        generator_module.REFINE_TEMPLATES,
        "tier1_xsum",
        dataclasses.replace(XSUM_SELF_REFINE, examples=FAKE_EXAMPLES),
    )


def _example():
    return Example("x1", 1, "summarization", SOURCE, SOURCE, "The library will close.")


def _cfg(seed=2, dataset="tier1_xsum"):
    return AgentConfig(dataset=dataset, seed=seed)


def _summary(text, round_index=0):
    return Message("generator", text, round_index, TokenCost(1, 1), 0.0, "stop")


def _feedback(text, round_index=0):
    return Message("self_feedback", text, round_index, TokenCost(1, 1), 0.0, "stop")


def _state(history=(), round_index=0):
    # As SelfRefine sends every call: no role preface (D-105).
    return AgentState(
        example=_example(),
        history=tuple(history),
        round_index=round_index,
        include_role_preface=False,
    )


class TestSelfFeedback:
    def test_instruction_is_the_approved_wording(self):
        assert XSUM_SELF_FEEDBACK.instruction == (
            "Check whether every part of the summary is supported by the article. For each part "
            'that the article does not support, write one line starting with "- " that quotes '
            "the unsupported words from the summary and says what the article states instead, "
            "or that the article does not mention it. If every part of the summary is supported "
            "by the article, reply with only the word NONE."
        )
        assert XSUM_SELF_FEEDBACK.label == "xsum_self_feedback@v3"
        assert XSUM_SELF_FEEDBACK.target_label == "Now check the summary of the article below."

    def test_examples_are_chat_turns_in_draft_none_draft_order(self, fake_templates):
        client = ScriptedClient(["NONE"])
        SelfFeedback(_cfg(), client).act(_state([_summary("The library will close.")]))
        ((prompt, _),) = client.calls
        one, two, three = FAKE_EXAMPLES
        instruction = XSUM_SELF_FEEDBACK.instruction
        assert client.turns[0] == (
            (f"{instruction}\n\nArticle:\n{one.article}\n\nSummary:\n{one.draft}", one.feedback),
            (f"{instruction}\n\nArticle:\n{two.article}\n\nSummary:\n{two.revised}", "NONE"),
            (
                f"{instruction}\n\nArticle:\n{three.article}\n\nSummary:\n{three.draft}",
                three.feedback,
            ),
        )
        assert prompt == (
            f"{instruction}\n\nNow check the summary of the article below.\n\n"
            f"Article:\n{SOURCE}\n\nSummary:\nThe library will close."
        )
        assert SEP not in prompt and all(one.article not in p for p, _ in client.calls)

    def test_only_the_latest_summary_is_shown(self, fake_templates):
        client = ScriptedClient(["NONE"])
        history = [_summary("first summary"), _feedback("- first feedback"), _summary("second", 1)]
        SelfFeedback(_cfg(), client).act(_state(history, round_index=1))
        prompt = client.calls[0][0]
        assert prompt.endswith("Summary:\nsecond")
        assert "first summary" not in prompt and "first feedback" not in prompt

    def test_message_keeps_the_text_unmodified_with_round_cost_and_finish_reason(self):
        client = ScriptedClient(["  none.\n"], ["length"])
        message = SelfFeedback(_cfg(), client).act(_state([_summary("s", 3)], round_index=3))
        assert message.role == "self_feedback" and message.content == "  none.\n"
        assert message.round_index == 3
        assert message.token_cost == TokenCost(100, 1) and message.finish_reason == "length"

    def test_one_user_message_per_call_with_the_run_seed(self):
        client = ScriptedClient(["NONE", "NONE"])
        agent = SelfFeedback(_cfg(seed=7), client)
        agent.act(_state([_summary("s")]))
        agent.act(_state([_summary("s")]))
        assert [seed for _, seed in client.calls] == [7, 7]
        assert all(p.startswith(XSUM_SELF_FEEDBACK.instruction) for p, _ in client.calls)

    @pytest.mark.parametrize("history", [[], [_summary("s"), _feedback("- x")]])
    def test_history_must_end_in_a_generator_summary(self, history):
        with pytest.raises(AgentError, match="Generator"):
            SelfFeedback(_cfg(), ScriptedClient([])).act(_state(history))

    def test_dataset_without_a_template_is_refused_at_construction(self):
        with pytest.raises(AgentError, match="no SelfFeedback prompt template"):
            SelfFeedback(_cfg(dataset="tier1_cnndm"), ScriptedClient([]))

    def test_template_version_is_reported(self):
        assert dict(SelfFeedback(_cfg(), ScriptedClient([])).prompt_templates) == {
            "xsum_self_feedback": "xsum_self_feedback@v3"
        }

    @pytest.mark.parametrize("text", ["NONE", " none.\n", "None", "NONE.", "nOnE . "])
    def test_stop_word_alone_stops(self, text):
        assert SelfFeedback(_cfg(), ScriptedClient([])).is_stop(_feedback(text))

    @pytest.mark.parametrize(
        "text",
        [
            "NONE. The summary is accurate.",
            "",
            "   ",
            '- "library": the article does not mention it.',
            "NONE..",
            "**NONE**",
            '"NONE"',
            "NONE\n- extra",
            "None of the summary is supported.",
        ],
    )
    def test_anything_else_is_feedback(self, text):
        assert not SelfFeedback(_cfg(), ScriptedClient([])).is_stop(_feedback(text))


class TestGeneratorRefine:
    def test_instruction_is_the_approved_wording(self):
        assert XSUM_SELF_REFINE.instruction == (
            "Rewrite the latest summary so that it fixes every problem listed in its feedback. "
            "Use only information stated in the article. Keep it to one sentence. Reply with "
            "only the rewritten summary."
        )
        assert XSUM_SELF_REFINE.label == "xsum_self_refine@v2"
        assert XSUM_SELF_REFINE.target_label == (
            "Now rewrite the latest summary of the article below."
        )

    def test_empty_history_is_the_d016_prompt_whatever_the_round(self, fake_templates):
        client = ScriptedClient(["s"])
        Generator(_cfg(), client).act(_state(round_index=2))
        assert client.calls[0][0] == f"Summarize the following article in one sentence:\n\n{SOURCE}"

    def test_refine_layout_examples_as_chat_turns_then_the_labelled_history(self, fake_templates):
        client = ScriptedClient(["s3"])
        history = [_summary("y0"), _feedback("fb0"), _summary("y1", 1), _feedback("fb1", 1)]
        message = Generator(_cfg(), client).act(_state(history, round_index=2))
        instruction = XSUM_SELF_REFINE.instruction
        assert client.turns[0] == tuple(
            (
                f"{instruction}\n\nArticle:\n{e.article}\n\n"
                f"Summary 1:\n{e.draft}\n\nFeedback 1:\n{e.feedback}",
                e.revised,
            )
            for e in FAKE_EXAMPLES
        )
        assert client.calls[0][0] == (
            f"{instruction}\n\nNow rewrite the latest summary of the article below.\n\n"
            f"Article:\n{SOURCE}\n\nSummary 1:\ny0\n\nFeedback 1:\nfb0\n\n"
            "Summary 2:\ny1\n\nFeedback 2:\nfb1"
        )
        assert SEP not in client.calls[0][0]
        assert message.role == "generator" and message.round_index == 2

    def test_draft_sends_no_turns(self, fake_templates):
        client = ScriptedClient(["s"])
        Generator(_cfg(), client).act(_state())
        assert client.turns == [()]

    @pytest.mark.parametrize(
        "history",
        [
            [_summary("y0")],
            [_summary("y0"), _feedback("fb0"), _summary("y1")],
            [_feedback("fb0")],
            [_summary("y0"), _feedback("fb0"), _feedback("fb1")],
            [_summary("y0"), _summary("y1"), _feedback("fb1")],
            [Message("critic", "c", 0, TokenCost(1, 1), 0.0, "stop")],
        ],
    )
    def test_history_that_is_not_answerable_is_refused(self, history):
        with pytest.raises(AgentError):
            Generator(_cfg(), ScriptedClient(["s"])).act(_state(history))

    def test_refine_on_a_dataset_without_a_refine_template_is_refused(self):
        generator = Generator(_cfg(dataset="tier1_cnndm"), ScriptedClient(["s"]))
        with pytest.raises(AgentError, match="no refine prompt template"):
            generator.act(_state([_summary("y0"), _feedback("fb0")]))


class TestRealExamples:
    def test_exactly_three_shared_by_both_templates(self):
        assert len(XSUM_SELF_REFINE_EXAMPLES) == 3
        assert XSUM_SELF_FEEDBACK.examples is XSUM_SELF_REFINE_EXAMPLES
        assert XSUM_SELF_REFINE.examples is XSUM_SELF_REFINE_EXAMPLES

    def test_none_is_from_the_frozen_evaluation_sample(self):
        sample = yaml.safe_load((CONFIGS / "data" / "tier1_xsum_sample.yaml").read_text())
        ids = {str(i) for i in sample["data"]["sample_ids"]}
        assert not {e.source_id for e in XSUM_SELF_REFINE_EXAMPLES} & ids

    def test_articles_have_the_evaluation_median_length_from_d103(self):
        assert [len(e.article.split()) for e in XSUM_SELF_REFINE_EXAMPLES] == [300, 300, 300]

    def test_feedback_lines_quote_words_that_are_in_the_draft_and_not_the_revision(self):
        for example in XSUM_SELF_REFINE_EXAMPLES:
            lines = example.feedback.split("\n")
            assert all(line.startswith("- ") for line in lines)
            quotes = [re.match(r'- "([^"]+)"', line).group(1) for line in lines]
            assert all(q in example.draft and q not in example.revised for q in quotes)


def _agents(client, seed=2):
    return {
        "generator": Generator(_cfg(seed), client),
        "self_feedback": SelfFeedback(_cfg(seed), client),
    }


def _t1(max_rounds=4):
    return SelfRefine(TopologyConfig(name="t1_self_refine", max_rounds=max_rounds))


def _never_none(max_rounds):
    """Replies for a run whose feedback never stops: summary, feedback, ... ending in feedback."""
    replies = []
    for k in range(max_rounds + 1):
        replies += [f"summary {k}", f'- "summary {k}": the article does not say this.']
    return replies


@pytest.mark.usefixtures("fake_templates")
class TestSelfRefineTopology:
    def test_first_feedback_none_keeps_the_draft(self):
        client = ScriptedClient(["the draft", "NONE"])
        result = _t1().run(_example(), _agents(client))
        assert len(client.calls) == 2
        assert result.output == "the draft"
        assert result.n_rounds == 1 and result.converged is True
        assert [m.role for m in result.trace] == ["generator", "self_feedback"]

    def test_none_after_two_refinements(self):
        client = ScriptedClient(["y0", "- a", "y1", "- b", "y2", "None."])
        result = _t1().run(_example(), _agents(client))
        assert len(client.calls) == 6
        assert result.output == "y2"
        assert result.n_rounds == 3 and result.converged is True
        assert [m.role for m in result.trace] == ["generator", "self_feedback"] * 3

    def test_budget_of_four_without_none_is_ten_calls_ending_in_a_final_check(self):
        replies = _never_none(4)
        finish = ["stop"] * 8 + ["length", "stop"]
        client = ScriptedClient(replies, finish)
        result = _t1(max_rounds=4).run(_example(), _agents(client))
        assert len(client.calls) == 10
        assert result.n_rounds == 5 and result.converged is False
        assert result.trace[-1].role == "self_feedback"
        assert result.output == "summary 4"
        assert result.finish_reason == "length"
        assert [m.round_index for m in result.trace] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        assert result.token_cost == TokenCost(sum(100 * n for n in range(1, 11)), 55)
        assert result.wall_clock_s >= 0

    def test_budget_of_two_without_none_is_six_calls(self):
        client = ScriptedClient(_never_none(2))
        result = _t1(max_rounds=2).run(_example(), _agents(client))
        assert len(client.calls) == 6
        assert result.n_rounds == 3 and result.converged is False
        assert result.output == "summary 2"

    def test_feedback_sees_the_latest_summary_and_refine_sees_the_whole_history(self):
        client = ScriptedClient(_never_none(2))
        _t1(max_rounds=2).run(_example(), _agents(client))
        feedback_on_y1 = client.calls[3][0]
        assert "summary 1" in feedback_on_y1 and "summary 0" not in feedback_on_y1
        refine_to_y2 = client.calls[4][0]
        assert all(f"summary {k}" in refine_to_y2 for k in (0, 1))
        assert refine_to_y2.endswith('Feedback 2:\n- "summary 1": the article does not say this.')

    def test_every_call_carries_the_run_seed(self):
        client = ScriptedClient(["y0", "- a", "y1", "NONE"])
        _t1().run(_example(), _agents(client, seed=9))
        assert {seed for _, seed in client.calls} == {9}

    def test_missing_max_rounds_is_refused_at_construction(self):
        with pytest.raises(TopologyError, match="max_rounds"):
            SelfRefine(TopologyConfig(name="t1_self_refine"))

    def test_missing_role_is_refused(self):
        agents = _agents(ScriptedClient([]))
        del agents["self_feedback"]
        with pytest.raises(TopologyError, match="self_feedback"):
            _t1().run(_example(), agents)

    def test_feedback_role_without_a_stop_signal_is_refused(self):
        agents = _agents(ScriptedClient([]))
        agents["self_feedback"] = agents["generator"]
        with pytest.raises(TopologyError, match="is_stop"):
            _t1().run(_example(), agents)


def test_roles_and_topology_are_registered():
    assert AGENTS.get("self_feedback") is SelfFeedback
    assert TOPOLOGIES.get("t1_self_refine") is SelfRefine
