"""The Generator's prompt and message contract, and T0 end to end against a canned client.

T1's self-feedback, the Generator's refine step and the T1 loop are in ``test_self_refine.py``.
"""

from __future__ import annotations

import pytest
from agent_evals.agents import TEMPLATES, AgentError, Generator
from agent_evals.config import (
    AgentConfig,
    ConfigError,
    SamplingConfig,
    ServingConfig,
    TopologyConfig,
)
from agent_evals.registry import AGENTS, TOPOLOGIES
from agent_evals.serving import BackboneSpec, ChatClient, Completion
from agent_evals.topologies import SingleAgent, TopologyError
from agent_evals.types import AgentState, Example, Message, TokenCost, TopologyResult

BACKBONE = BackboneSpec(
    name="fake-backbone",
    hf_path="org/fake",
    hf_revision="a" * 40,
    gated=False,
    quantization="bitsandbytes",
    max_model_len=2809,
)


class CannedClient(ChatClient):
    """Returns one scripted completion and records every prompt, seed and system it was sent."""

    def __init__(self, text="A one-sentence summary.", finish_reason="stop"):
        super().__init__(ServingConfig(), SamplingConfig(), BACKBONE)
        self.text = text
        self.finish_reason = finish_reason
        self.calls: list[tuple[str, int | None, str | None]] = []
        self.turns: list[tuple[tuple[str, str], ...]] = []

    def complete(self, prompt, seed=None, system=None, turns=()):
        self.calls.append((prompt, seed, system))
        self.turns.append(tuple(turns))
        return Completion(
            text=self.text,
            token_cost=TokenCost(prompt_tokens=120, completion_tokens=18),
            latency_s=0.5,
            finish_reason=self.finish_reason,
        )


def _example(source="The council voted to close the {old} library in Zürich."):
    return Example(
        id="x1",
        tier=1,
        task="summarization",
        source=source,
        evidence=source,
        reference="The library will close.",
    )


def _generator(client, seed=2):
    return Generator(AgentConfig(dataset="tier1_xsum", seed=seed), client)


class TestGenerator:
    def test_prompt_is_exactly_the_d016_wording_then_the_article(self):
        client = CannedClient()
        _generator(client).act(
            AgentState(
                example=_example(), history=(), round_index=0, include_role_preface=False
            )
        )
        prompt, _, _= client.calls[0]
        assert prompt == (
            "Summarize the following article in one sentence:\n\n"
            "The council voted to close the {old} library in Zürich."
        )

    def test_cnndm_prompt_is_exactly_the_d037_wording_then_the_article(self):
        client = CannedClient()
        generator = Generator(AgentConfig(dataset="tier1_cnndm", seed=0), client)
        generator.act(AgentState(example=_example(), history=(), round_index=0, include_role_preface=False))
        prompt, _, _ = client.calls[0]
        assert prompt == (
            "Summarize the following article in three sentences:\n\n"
            "The council voted to close the {old} library in Zürich."
        )
        assert dict(generator.prompt_templates) == {
            "cnndm_three_sentences": "cnndm_three_sentences@v1",
            "cnndm_critique_revise": "cnndm_critique_revise@v1",
        }

    def test_samsum_prompt_is_exactly_the_d043_wording_then_the_dialogue(self):
        client = CannedClient()
        generator = Generator(AgentConfig(dataset="tier1_samsum", seed=0), client)
        generator.act(AgentState(example=_example(), history=(), round_index=0, include_role_preface=False))
        prompt, _, _ = client.calls[0]
        assert prompt == (
            "Generate a summary of the following dialogue snippet:\n\n"
            "The council voted to close the {old} library in Zürich."
        )
        assert dict(generator.prompt_templates) == {
            "samsum_dialogue_snippet": "samsum_dialogue_snippet@v1"
        }

    def test_every_call_carries_the_run_seed(self):
        client = CannedClient()
        generator = _generator(client, seed=2)
        for _ in range(2):
            generator.act(AgentState(example=_example(), history=(), round_index=0))
        assert [seed for _, seed, _ in client.calls] == [2, 2]

    def test_role_preface_is_folded_into_the_prompt_when_the_topology_asks_for_one(self):
        client = CannedClient()
        _generator(client).act(
            AgentState(
                example=_example(), history=(), round_index=0, include_role_preface=True
            )
        )
        prompt, _, system = client.calls[0]
        assert system is None
        assert prompt.startswith(Generator.SYSTEM_PROMPT + "\n\n")
        assert Generator.SYSTEM_PROMPT == (
            "You are a summarization assistant. Follow the instruction exactly and reply with "
            "only the requested summary."
        )

    def test_role_preface_defaults_to_included_when_the_topology_does_not_say(self):
        """T2 builds AgentState without setting include_role_preface (D-048); the default must
        be True so it gets one without needing to touch every call site."""
        client = CannedClient()
        _generator(client).act(AgentState(example=_example(), history=(), round_index=0))
        prompt, _, system = client.calls[0]
        assert system is None
        assert prompt.startswith(Generator.SYSTEM_PROMPT + "\n\n")

    def test_no_role_preface_when_the_topology_says_so(self):
        """T0 sets include_role_preface=False to keep D-016's original zero-shot baseline
        exact (D-048). No call, from any topology, ever sends a separate system message -- that
        is what D-048 moved away from D-045's approach."""
        client = CannedClient()
        _generator(client).act(
            AgentState(
                example=_example(), history=(), round_index=0, include_role_preface=False
            )
        )
        prompt, _, system = client.calls[0]
        assert system is None
        assert not prompt.startswith(Generator.SYSTEM_PROMPT)

    def test_message_carries_text_cost_latency_and_finish_reason(self):
        message = _generator(CannedClient(finish_reason="length")).act(
            AgentState(example=_example(), history=(), round_index=0)
        )
        assert message.role == "generator"
        assert message.content == "A one-sentence summary."
        assert message.token_cost == TokenCost(120, 18)
        assert message.latency_s == 0.5
        assert message.finish_reason == "length"

    def test_blank_model_output_is_passed_through_not_hidden(self):
        message = _generator(CannedClient(text="")).act(
            AgentState(example=_example(), history=(), round_index=0)
        )
        assert message.content == ""

    def test_template_version_is_reported(self):
        assert TEMPLATES["tier1_xsum"].version == "v1"

    def test_dataset_without_a_template_is_refused(self):
        with pytest.raises(AgentError, match="no Generator prompt template"):
            Generator(AgentConfig(dataset="cnn_dailymail", seed=0), CannedClient())

    @pytest.mark.parametrize("kwargs", [{"dataset": "", "seed": 0}, {"dataset": "x", "seed": -1}])
    def test_agent_config_is_validated(self, kwargs):
        with pytest.raises(ConfigError):
            AgentConfig(**kwargs)

    def test_round_zero_with_an_empty_history_uses_the_initial_template(self):
        client = CannedClient()
        _generator(client).act(
            AgentState(example=_example(), history=(), round_index=0, include_role_preface=False)
        )
        prompt, _, _ = client.calls[0]
        assert prompt == (
            "Summarize the following article in one sentence:\n\n"
            "The council voted to close the {old} library in Zürich."
        )

    def test_history_not_the_round_decides_the_prompt_d101(self):
        # A history ending in the Generator's own draft has nothing to answer, even at round 0:
        # the prompt is chosen from who spoke last, never from round_index (D-101).
        client = CannedClient()
        draft = Message(
            role="generator",
            content="stale draft",
            round_index=0,
            token_cost=TokenCost(1, 1),
            latency_s=0.0,
        )
        with pytest.raises(AgentError, match="does not answer a history ending in 'generator'"):
            _generator(client).act(
                AgentState(
                    example=_example(),
                    history=(draft,),
                    round_index=0,
                    include_role_preface=False,
                )
            )
        assert client.calls == []

    def test_critique_revise_template_version_is_reported_alongside_the_initial_one(self):
        assert dict(_generator(CannedClient()).prompt_templates) == {
            "xsum_one_sentence": "xsum_one_sentence@v1",
            "xsum_self_refine": "xsum_self_refine@v2",
            "xsum_critique_revise": "xsum_critique_revise@v1",
        }

    def test_critique_revise_prompt_is_the_d043_wording_with_draft_and_critique(self):
        client = CannedClient()
        draft = Message(
            role="generator",
            content="A flawed one-sentence summary.",
            round_index=0,
            token_cost=TokenCost(120, 18),
            latency_s=0.5,
        )
        critique = Message(
            role="critic",
            content="The library location is wrong.",
            round_index=1,
            token_cost=TokenCost(150, 20),
            latency_s=0.5,
        )
        _generator(client).act(
            AgentState(example=_example(), history=(draft, critique), round_index=2)
        )
        prompt, _, system = client.calls[0]
        assert system is None
        assert prompt == (
            f"{Generator.SYSTEM_PROMPT}\n\n"
            "Rewrite the summary below to fix the problems raised in the critique, using only "
            "the article as evidence. If the critique found no problems, repeat the summary "
            "unchanged. Reply with only the corrected one-sentence summary.\n\n"
            "Article:\n"
            "The council voted to close the {old} library in Zürich.\n\n"
            "Summary:\n"
            "A flawed one-sentence summary.\n\n"
            "Critique:\n"
            "The library location is wrong."
        )

    def test_critique_revise_reads_the_generators_prior_message_not_the_original_draft(self):
        """With a longer trace, "previous_output" is the message right before the critique."""
        client = CannedClient()
        original_draft = Message(
            role="generator",
            content="original draft",
            round_index=0,
            token_cost=TokenCost(1, 1),
            latency_s=0.0,
        )
        first_critique = Message(
            role="critic",
            content="first critique",
            round_index=1,
            token_cost=TokenCost(1, 1),
            latency_s=0.0,
        )
        second_draft = Message(
            role="generator",
            content="second draft",
            round_index=2,
            token_cost=TokenCost(1, 1),
            latency_s=0.0,
        )
        second_critique = Message(
            role="critic",
            content="second critique",
            round_index=3,
            token_cost=TokenCost(1, 1),
            latency_s=0.0,
        )
        _generator(client).act(
            AgentState(
                example=_example(),
                history=(original_draft, first_critique, second_draft, second_critique),
                round_index=4,
            )
        )
        prompt, _, _ = client.calls[0]
        assert "second draft" in prompt
        assert "second critique" in prompt
        assert "original draft" not in prompt
        assert "first critique" not in prompt

    def test_critique_revise_round_with_only_one_prior_message_is_an_error(self):
        critique_only = Message(
            role="critic", content="x", round_index=1, token_cost=TokenCost(1, 1), latency_s=0.0
        )
        with pytest.raises(AgentError, match="prior summary"):
            _generator(CannedClient()).act(
                AgentState(example=_example(), history=(critique_only,), round_index=2)
            )

    def test_critique_revise_round_without_a_template_for_the_dataset_is_an_error(self):
        generator = Generator(AgentConfig(dataset="tier1_xsum", seed=0), CannedClient())
        generator._dataset = "tier1_xsum_no_critique_revise"
        generator._critique_revise_template = None
        draft = Message(
            role="generator", content="x", round_index=0, token_cost=TokenCost(1, 1), latency_s=0.0
        )
        critique = Message(
            role="critic", content="y", round_index=1, token_cost=TokenCost(1, 1), latency_s=0.0
        )
        with pytest.raises(AgentError, match="no Generator critique-revise template"):
            generator.act(AgentState(example=_example(), history=(draft, critique), round_index=2))


class TestSingleAgent:
    def test_one_call_one_round_and_a_complete_result(self):
        client = CannedClient()
        topology = SingleAgent(TopologyConfig(name="t0_single_agent"))
        result = topology.run(_example(), {"generator": _generator(client)})
        assert isinstance(result, TopologyResult)
        assert len(client.calls) == 1
        assert result.output == "A one-sentence summary."
        assert result.token_cost == TokenCost(120, 18)
        assert result.n_rounds == 1 and result.converged is True
        assert len(result.trace) == 1 and result.trace[0].role == "generator"
        assert result.wall_clock_s >= 0
        assert result.finish_reason == "stop"

    def test_sends_no_role_preface_d016_baseline_restored_by_d048(self):
        client = CannedClient()
        SingleAgent(TopologyConfig(name="t0_single_agent")).run(
            _example(), {"generator": _generator(client)}
        )
        prompt, _, system = client.calls[0]
        assert system is None
        assert not prompt.startswith(Generator.SYSTEM_PROMPT)

    def test_missing_generator_is_an_error(self):
        with pytest.raises(TopologyError, match="generator"):
            SingleAgent(TopologyConfig(name="t0_single_agent")).run(_example(), {})

    def test_assemble_rejects_an_output_message_not_in_the_trace(self):
        topology = SingleAgent(TopologyConfig(name="t0_single_agent"))
        kept, other = _message("generator", "kept"), _message("generator", "not kept")
        with pytest.raises(TopologyError, match="not in the trace"):
            topology._assemble(other, (kept,), n_rounds=1, converged=True, started=0.0)


def _message(role, content, round_index=0):
    return Message(role, content, round_index, TokenCost(1, 1), 0.0, "stop")


def test_roles_and_topologies_are_registered():
    assert AGENTS.get("generator") is Generator
    assert TOPOLOGIES.get("t0_single_agent") is SingleAgent
