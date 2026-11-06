"""The Generator's ASQA prompt: ALCE's instruction and worked examples, drawn per seed (D-118)."""

from __future__ import annotations

import hashlib
import json

import pytest
from agent_evals.agents import TEMPLATES, AgentError, Generator
from agent_evals.agents.asqa_examples import ALCE_ASQA_EXAMPLES, ALCE_COMMIT
from agent_evals.agents.generator import ASQA_ALCE_VANILLA
from agent_evals.config import AgentConfig, SamplingConfig, ServingConfig
from agent_evals.serving import BackboneSpec, ChatClient, Completion
from agent_evals.types import AgentState, Example, TokenCost

#: SHA-256 of the instruction and the four examples as copied from ALCE's prompt file at
#: ALCE_COMMIT, checked against the file on 2026-10-23. Any edit to that text must bump the
#: template version and this value together.
ALCE_CONTENT_SHA256 = "de1319d8d99f3e347e60bf270ad19ddee70dea07c1f9edf298f014a0856ff0e7"

#: The examples each seed draws, by ASQA train ID, in the order they are sent.
DRAWS = {
    0: ("-8254578338731748300", "-3207884824372475669"),
    1: ("-3207884824372475669", "3219470493874132834"),
    2: ("-5117526355389369643", "-8254578338731748300"),
}

PASSAGES = "Document [1](Title: T1): One.\nDocument [2](Title: T2): Two."

BACKBONE = BackboneSpec(
    name="fake-backbone",
    hf_path="org/fake",
    hf_revision="a" * 40,
    gated=False,
    quantization="bitsandbytes",
    max_model_len=8192,
)


class CannedClient(ChatClient):
    """Records every prompt, seed, system message and earlier turns it was sent."""

    def __init__(self):
        super().__init__(ServingConfig(), SamplingConfig(), BACKBONE)
        self.calls = []

    def complete(self, prompt, seed=None, system=None, turns=()):
        self.calls.append({"prompt": prompt, "seed": seed, "system": system, "turns": turns})
        return Completion("An answer [1].", TokenCost(900, 40), 0.5, "stop")


def _example(question="Who won the fixture contest?"):
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


def _act(seed=0, example=None):
    client = CannedClient()
    generator = Generator(AgentConfig(dataset="tier2_asqa", seed=seed), client)
    # T0 sends no role preface (D-048), as SingleAgent does.
    state = AgentState(
        example=example or _example(), history=(), round_index=0, include_role_preface=False
    )
    generator.act(state)
    return generator, client.calls[0]


class TestTemplate:
    def test_registered_for_asqa(self):
        assert TEMPLATES["tier2_asqa"] is ASQA_ALCE_VANILLA
        assert ASQA_ALCE_VANILLA.label == "asqa_alce_vanilla@v1"
        assert ASQA_ALCE_VANILLA.n_examples == 2

    def test_text_is_unchanged_from_alce(self):
        payload = json.dumps(
            [
                ASQA_ALCE_VANILLA.instruction,
                [
                    [e.source_id, e.question, [list(p) for p in e.passages], e.answer]
                    for e in ASQA_ALCE_VANILLA.examples
                ],
            ],
            ensure_ascii=False,
        )
        assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == ALCE_CONTENT_SHA256
        assert ALCE_COMMIT == "246c476a4edfc564266b7346b6e29ef4861ae937"

    def test_pool_is_alces_four_examples_with_five_passages_each(self):
        assert ASQA_ALCE_VANILLA.examples is ALCE_ASQA_EXAMPLES
        assert len({e.source_id for e in ALCE_ASQA_EXAMPLES}) == 4
        assert all(len(e.passages) == 5 for e in ALCE_ASQA_EXAMPLES)


class TestDraw:
    @pytest.mark.parametrize("seed", sorted(DRAWS))
    def test_each_seed_draws_its_recorded_pair(self, seed):
        drawn = ASQA_ALCE_VANILLA.drawn_examples(seed)
        assert tuple(e.source_id for e in drawn) == DRAWS[seed]

    def test_the_draw_repeats_for_a_seed(self):
        assert ASQA_ALCE_VANILLA.drawn_examples(7) == ASQA_ALCE_VANILLA.drawn_examples(7)


class TestMessages:
    def test_examples_are_sent_as_user_and_assistant_turns(self):
        _, call = _act(seed=0)
        assert len(call["turns"]) == 2
        for (user, assistant), example in zip(
            call["turns"], ASQA_ALCE_VANILLA.drawn_examples(0), strict=True
        ):
            lines = user.split("\n")
            assert user.startswith(ASQA_ALCE_VANILLA.instruction + "\n\nQuestion: ")
            assert lines[2] == f"Question: {example.question}"
            assert lines[4].startswith(f"Document [1](Title: {example.passages[0][0]}): ")
            assert lines[8].startswith(f"Document [5](Title: {example.passages[4][0]}): ")
            assert len(lines) == 9
            assert assistant == example.answer

    def test_final_message_is_the_question_under_work_in_the_same_form(self):
        _, call = _act(seed=1)
        assert call["prompt"] == (
            f"{ASQA_ALCE_VANILLA.instruction}\n\nQuestion: Who won the fixture contest?\n\n"
            f"{PASSAGES}"
        )

    def test_run_seed_on_the_request_and_no_system_message(self):
        _, call = _act(seed=2)
        assert call["seed"] == 2
        assert call["system"] is None

    def test_manifest_records_the_template_and_the_drawn_examples(self):
        generator, _ = _act(seed=1)
        assert dict(generator.prompt_templates) == {
            "asqa_alce_vanilla": "asqa_alce_vanilla@v1",
            "asqa_alce_vanilla_examples": ",".join(DRAWS[1]),
            "asqa_self_refine": "asqa_self_refine@v1",
        }

    def test_an_example_without_a_question_is_refused(self):
        with pytest.raises(AgentError, match="needs a question"):
            _act(example=_example(question=None))

    def test_zero_shot_templates_record_no_examples(self):
        generator = Generator(AgentConfig(dataset="tier1_xsum", seed=0), CannedClient())
        assert not any(name.endswith("_examples") for name in generator.prompt_templates)
