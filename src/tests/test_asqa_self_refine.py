"""T1 Self-Refine on ASQA (D-150): the feedback and refine prompts, the worked examples, and the T1
loop end to end against a scripted client. CPU, offline."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from agent_evals.agents import AgentError, Generator, SelfFeedback
from agent_evals.agents.asqa_examples import ALCE_ASQA_EXAMPLES
from agent_evals.agents.generator import (
    ASQA_ALCE_VANILLA,
    ASQA_SELF_REFINE,
    REFINE_TEMPLATES,
    TEMPLATES,
)
from agent_evals.agents.self_feedback import ASQA_SELF_FEEDBACK, FEEDBACK_TEMPLATES
from agent_evals.agents.self_refine_examples import ASQA_SELF_REFINE_EXAMPLES
from agent_evals.config import AgentConfig, SamplingConfig, ServingConfig, TopologyConfig
from agent_evals.datasets.asqa import format_passages
from agent_evals.serving import BackboneSpec, ChatClient, Completion
from agent_evals.topologies import SelfRefine
from agent_evals.types import AgentState, Example, Message, TokenCost

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PASSAGES = "Document [1](Title: T1): One.\nDocument [2](Title: T2): Two."
QUESTION = "Who won the fixture contest?"

BACKBONE = BackboneSpec(
    name="fake-backbone",
    hf_path="org/fake",
    hf_revision="a" * 40,
    gated=False,
    quantization="bitsandbytes",
    max_model_len=32768,
)


class ScriptedClient(ChatClient):
    """Returns the scripted replies in order and records every prompt, seed and set of turns."""

    def __init__(self, replies):
        super().__init__(ServingConfig(), SamplingConfig(), BACKBONE)
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, prompt, seed=None, turns=()):
        self.calls.append({"prompt": prompt, "seed": seed, "turns": tuple(turns)})
        n = len(self.calls)
        return Completion(self.replies[n - 1], TokenCost(100 * n, n), 0.1, "stop")


def _example(question=QUESTION):
    return Example(
        id="FAKE_EXAMPLE_q1",
        tier=2,
        task="grounded_qa",
        source=PASSAGES,
        evidence=PASSAGES,
        reference="A human answer.",
        question=question,
        gold='{"short_answers": [["X"]], "long_answers": ["A.", "B."]}',
    )


def _cfg(seed=1):
    return AgentConfig(dataset="tier2_asqa", seed=seed)


def _message(role, text, round_index=0):
    return Message(role, text, round_index, TokenCost(1, 1), 0.0, "stop")


def _state(history, example=None):
    return AgentState(
        example=example or _example(),
        history=tuple(history),
        round_index=0,
        include_role_preface=False,
    )


def _source(qa):
    return f"Question: {qa.question}\n\n{format_passages(qa.passage_records())}"


class TestWording:
    def test_feedback_instruction_is_the_d150_wording(self):
        assert ASQA_SELF_FEEDBACK.instruction == (
            "Check whether every sentence of the answer is supported by the search results it "
            "cites. For each sentence that cites no search result, or whose cited search results "
            'do not support it, write one line starting with "- " that quotes the unsupported '
            "words and says which search result supports them, or that none does. If every "
            "sentence is supported by the search results it cites, reply with only the word NONE."
        )
        assert ASQA_SELF_FEEDBACK.label == "asqa_self_feedback@v1"
        assert ASQA_SELF_FEEDBACK.target_label == "Now check the answer to the question below."

    def test_refine_instruction_is_the_d150_wording(self):
        assert ASQA_SELF_REFINE.instruction == (
            "Rewrite the latest answer so that it fixes every problem listed in its feedback. Use "
            "only information stated in the search results. Write an accurate, engaging, and "
            "concise answer in an unbiased and journalistic tone. Always cite for any factual "
            "claim. When citing several search results, use [1][2][3]. Cite at least one document "
            "and at most three documents in each sentence. If multiple documents support the "
            "sentence, only cite a minimum sufficient subset of the documents. Reply with only "
            "the rewritten answer."
        )
        assert ASQA_SELF_REFINE.label == "asqa_self_refine@v1"
        assert ASQA_SELF_REFINE.target_label == (
            "Now rewrite the latest answer to the question below."
        )

    def test_refine_restates_t0s_citation_rules_word_for_word(self):
        rules = ASQA_ALCE_VANILLA.instruction[ASQA_ALCE_VANILLA.instruction.index("Always cite") :]
        assert rules in ASQA_SELF_REFINE.instruction

    def test_both_registered_for_asqa_and_t0s_template_untouched(self):
        assert FEEDBACK_TEMPLATES["tier2_asqa"] is ASQA_SELF_FEEDBACK
        assert REFINE_TEMPLATES["tier2_asqa"] is ASQA_SELF_REFINE
        assert TEMPLATES["tier2_asqa"] is ASQA_ALCE_VANILLA


class TestFeedbackPrompt:
    def test_examples_are_chat_turns_in_draft_none_draft_order(self):
        client = ScriptedClient(["NONE"])
        SelfFeedback(_cfg(), client).act(_state([_message("generator", "An answer [1].")]))
        one, two, three = ASQA_SELF_REFINE_EXAMPLES
        instruction = ASQA_SELF_FEEDBACK.instruction
        assert client.calls[0]["turns"] == (
            (f"{instruction}\n\n{_source(one.qa)}\n\nAnswer:\n{one.draft}", one.feedback),
            (f"{instruction}\n\n{_source(two.qa)}\n\nAnswer:\n{two.revised}", "NONE"),
            (f"{instruction}\n\n{_source(three.qa)}\n\nAnswer:\n{three.draft}", three.feedback),
        )

    def test_final_message_is_the_labelled_question_passages_and_latest_answer_only(self):
        client = ScriptedClient(["NONE"])
        history = [
            _message("generator", "first [1]."),
            _message("self_feedback", "- first feedback"),
            _message("generator", "second [2].", 1),
        ]
        SelfFeedback(_cfg(), client).act(_state(history))
        assert client.calls[0]["prompt"] == (
            f"{ASQA_SELF_FEEDBACK.instruction}\n\nNow check the answer to the question below."
            f"\n\nQuestion: {QUESTION}\n\n{PASSAGES}\n\nAnswer:\nsecond [2]."
        )

    def test_an_example_without_a_question_is_refused(self):
        agent = SelfFeedback(_cfg(), ScriptedClient(["NONE"]))
        state = _state([_message("generator", "a [1].")], _example(question=None))
        with pytest.raises(AgentError, match="asqa_self_feedback@v1 needs a question"):
            agent.act(state)

    def test_manifest_records_the_template(self):
        assert dict(SelfFeedback(_cfg(), ScriptedClient([])).prompt_templates) == {
            "asqa_self_feedback": "asqa_self_feedback@v1"
        }


class TestRefinePrompt:
    def test_examples_are_chat_turns_with_numbered_answer_and_feedback(self):
        client = ScriptedClient(["rewrite [1]."])
        history = [_message("generator", "a [1]."), _message("self_feedback", "- fb")]
        Generator(_cfg(), client).act(_state(history))
        instruction = ASQA_SELF_REFINE.instruction
        assert client.calls[0]["turns"] == tuple(
            (
                f"{instruction}\n\n{_source(e.qa)}\n\nAnswer 1:\n{e.draft}\n\n"
                f"Feedback 1:\n{e.feedback}",
                e.revised,
            )
            for e in ASQA_SELF_REFINE_EXAMPLES
        )

    def test_final_message_numbers_the_whole_history(self):
        client = ScriptedClient(["rewrite [1]."])
        history = [
            _message("generator", "a0 [1]."),
            _message("self_feedback", "- fb0"),
            _message("generator", "a1 [2].", 1),
            _message("self_feedback", "- fb1", 1),
        ]
        Generator(_cfg(), client).act(_state(history))
        assert client.calls[0]["prompt"] == (
            f"{ASQA_SELF_REFINE.instruction}\n\nNow rewrite the latest answer to the question "
            f"below.\n\nQuestion: {QUESTION}\n\n{PASSAGES}\n\nAnswer 1:\na0 [1].\n\n"
            "Feedback 1:\n- fb0\n\nAnswer 2:\na1 [2].\n\nFeedback 2:\n- fb1"
        )

    def test_an_example_without_a_question_is_refused(self):
        generator = Generator(_cfg(), ScriptedClient(["s"]))
        history = [_message("generator", "a [1]."), _message("self_feedback", "- fb")]
        with pytest.raises(AgentError, match="asqa_self_refine@v1 needs a question"):
            generator.act(_state(history, _example(question=None)))


class TestExamples:
    def test_three_shared_by_both_templates_in_d150_order(self):
        assert ASQA_SELF_FEEDBACK.examples is ASQA_SELF_REFINE_EXAMPLES
        assert ASQA_SELF_REFINE.examples is ASQA_SELF_REFINE_EXAMPLES
        assert [e.source_id for e in ASQA_SELF_REFINE_EXAMPLES] == [
            "-8254578338731748300",
            "-3207884824372475669",
            "3219470493874132834",
        ]

    def test_questions_and_passages_are_alces_own_examples(self):
        alce = {id(e) for e in ALCE_ASQA_EXAMPLES}
        assert all(id(e.qa) in alce for e in ASQA_SELF_REFINE_EXAMPLES)

    def test_first_two_revisions_are_alces_answers_and_the_third_is_not(self):
        one, two, three = ASQA_SELF_REFINE_EXAMPLES
        assert one.revised == one.qa.answer and two.revised == two.qa.answer
        assert three.revised != three.qa.answer

    def test_none_is_in_the_frozen_evaluation_sample(self):
        sample = yaml.safe_load((CONFIGS / "data" / "tier2_asqa_sample.yaml").read_text())
        ids = {str(i) for i in sample["data"]["sample_ids"]}
        assert not {e.source_id for e in ASQA_SELF_REFINE_EXAMPLES} & ids

    def test_two_feedback_lines_each_quoting_the_draft_and_not_the_revision(self):
        for example in ASQA_SELF_REFINE_EXAMPLES:
            lines = example.feedback.split("\n")
            assert len(lines) == 2 and all(line.startswith("- ") for line in lines)
            quotes = [re.match(r'- "([^"]+)"', line).group(1) for line in lines]
            assert all(q in example.draft and q not in example.revised for q in quotes)


class TestLoop:
    def test_draft_is_sent_exactly_as_t0_sends_it(self):
        t0 = ScriptedClient(["draft [1]."])
        Generator(_cfg(seed=1), t0).act(_state([]))
        t1 = ScriptedClient(["draft [1].", "NONE"])
        agents = {
            "generator": Generator(_cfg(seed=1), t1),
            "self_feedback": SelfFeedback(_cfg(seed=1), t1),
        }
        SelfRefine(TopologyConfig(name="t1_self_refine", max_rounds=4)).run(_example(), agents)
        assert t1.calls[0] == t0.calls[0]

    def test_draft_feedback_rewrite_stop(self):
        replies = ["draft [1].", '- "draft": no search result says this.', "rewrite [2].", "NONE"]
        client = ScriptedClient(replies)
        agents = {
            "generator": Generator(_cfg(), client),
            "self_feedback": SelfFeedback(_cfg(), client),
        }
        result = SelfRefine(TopologyConfig(name="t1_self_refine", max_rounds=4)).run(
            _example(), agents
        )
        assert result.output == "rewrite [2]."
        assert result.converged is True and result.n_rounds == 2
        assert [m.role for m in result.trace] == [
            "generator",
            "self_feedback",
            "generator",
            "self_feedback",
        ]
        assert result.token_cost == TokenCost(1000, 10)
        assert result.wall_clock_s is not None
        assert client.calls[3]["prompt"].endswith("Answer:\nrewrite [2].")
