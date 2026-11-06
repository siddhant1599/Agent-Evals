"""The Critic's prompt and tool-use contract, T2 end to end, and the NLI tool's pure logic.

Uses a fake NLI tool throughout (injected via Critic's ``tool`` parameter) so nothing here loads
the real DeBERTa checkpoint -- that needs ``transformers``, which real-model tests across this
suite gate behind ``@pytest.mark.gpu`` rather than requiring in the default CPU run.
"""

from __future__ import annotations

import numpy as np
import pytest
from agent_evals.agents import CRITIQUE_TEMPLATES, Critic, CriticNoNLI, Generator
from agent_evals.agents.generator import CRITIQUE_REVISE_TEMPLATES
from agent_evals.agents.tools import WeakClaim, classify_claim, rank_entailment
from agent_evals.config import AgentConfig, TopologyConfig
from agent_evals.registry import AGENTS, TOPOLOGIES
from agent_evals.serving import BackboneSpec
from agent_evals.topologies import CritiqueRevise, CritiqueReviseNoNLI, TopologyError
from agent_evals.types import AgentState, TokenCost, TopologyResult
from test_agents_topologies import CannedClient, _example, _generator

BACKBONE = BackboneSpec(
    name="fake-backbone",
    hf_path="org/fake",
    hf_revision="a" * 40,
    gated=False,
    quantization="bitsandbytes",
    max_model_len=2809,
)


class FakeTool:
    """Returns a scripted list of weak claims and records every (article, draft) it was asked."""

    def __init__(self, weak_claims=()):
        self.weak_claims = list(weak_claims)
        self.calls: list[tuple[str, str]] = []

    def find_weak_claims(self, article, draft):
        self.calls.append((article, draft))
        return self.weak_claims


def _critic(client, tool=None, seed=2, dataset="tier1_xsum"):
    return Critic(AgentConfig(dataset=dataset, seed=seed), client, tool=tool or FakeTool())


def _critic_no_nli(client, seed=2, dataset="tier1_xsum"):
    return CriticNoNLI(AgentConfig(dataset=dataset, seed=seed), client)


def _draft_message(content="A flawed one-sentence summary."):
    from agent_evals.types import Message

    return Message(
        role="generator",
        content=content,
        round_index=0,
        token_cost=TokenCost(120, 18),
        latency_s=0.5,
    )


class TestCritic:
    def test_critique_prompt_is_the_d043_wording_with_article_and_draft(self):
        client = CannedClient()
        _critic(client).act(
            AgentState(example=_example(), history=(_draft_message(),), round_index=1)
        )
        prompt, _, system = client.calls[0]
        assert system is None
        assert prompt == (
            f"{Critic.SYSTEM_PROMPT}\n\n"
            "List the factual problems in the summary below, using only the article as "
            "evidence. Be brief and specific. If there are none, reply with exactly: No "
            "factual problems found.\n\n"
            "Article:\n"
            "The council voted to close the {old} library in Zürich.\n\n"
            "Summary:\n"
            "A flawed one-sentence summary."
        )

    def test_tool_findings_are_spliced_into_the_prompt_when_present(self):
        client = CannedClient()
        tool = FakeTool(
            weak_claims=[
                WeakClaim(
                    sentence="The library closes in June.",
                    label="CONTRADICTION",
                    best_entailment_prob=0.05,
                )
            ]
        )
        _critic(client, tool=tool).act(
            AgentState(example=_example(), history=(_draft_message(),), round_index=1)
        )
        prompt, _, _ = client.calls[0]
        assert "automatic fact-checker flagged" in prompt
        assert '"The library closes in June." (contradiction)' in prompt

    def test_no_tool_findings_means_no_flagged_section(self):
        client = CannedClient()
        _critic(client, tool=FakeTool(weak_claims=[])).act(
            AgentState(example=_example(), history=(_draft_message(),), round_index=1)
        )
        prompt, _, _ = client.calls[0]
        assert "automatic fact-checker" not in prompt

    def test_tool_is_called_with_the_article_and_the_drafts_content(self):
        tool = FakeTool()
        _critic(CannedClient(), tool=tool).act(
            AgentState(
                example=_example(), history=(_draft_message("the draft text"),), round_index=1
            )
        )
        assert tool.calls == [
            ("The council voted to close the {old} library in Zürich.", "the draft text")
        ]

    def test_call_carries_the_d044_persona_in_the_prompt_not_as_a_system_message(self):
        client = CannedClient()
        _critic(client).act(
            AgentState(example=_example(), history=(_draft_message(),), round_index=1)
        )
        prompt, _, system = client.calls[0]
        assert system is None
        assert prompt.startswith(Critic.SYSTEM_PROMPT + "\n\n")
        assert Critic.SYSTEM_PROMPT != Generator.SYSTEM_PROMPT

    def test_message_carries_role_cost_latency_and_finish_reason(self):
        message = _critic(
            CannedClient(text="No factual problems found.", finish_reason="stop")
        ).act(AgentState(example=_example(), history=(_draft_message(),), round_index=1))
        assert message.role == "critic"
        assert message.content == "No factual problems found."
        assert message.token_cost == TokenCost(120, 18)
        assert message.finish_reason == "stop"

    def test_empty_history_is_an_error(self):
        with pytest.raises(Exception, match="prior Generator message"):
            _critic(CannedClient()).act(AgentState(example=_example(), history=(), round_index=1))

    def test_dataset_without_a_template_is_refused(self):
        with pytest.raises(Exception, match="no Critic prompt template"):
            Critic(AgentConfig(dataset="no_such_dataset", seed=0), CannedClient(), tool=FakeTool())

    def test_template_is_reported_and_registered(self):
        assert dict(_critic(CannedClient()).prompt_templates) == {
            "xsum_critique": "xsum_critique@v1"
        }
        assert CRITIQUE_TEMPLATES["tier1_xsum"].version == "v1"
        assert AGENTS.get("critic") is Critic

    def test_does_not_construct_a_real_tool_when_one_is_injected(self):
        """No transformers import happens when a fake tool is given -- confirms the injection
        seam actually bypasses NLIFactCheckTool's constructor."""
        Critic(AgentConfig(dataset="tier1_xsum", seed=0), CannedClient(), tool=FakeTool())


class TestCriticNoNLI:
    """D-049: the no-tool Critic variant. Same wording and persona as Critic; the only
    difference is that no tool ever runs, so ``flagged`` is always empty."""

    def test_builds_without_any_tool_no_transformers_import(self):
        """Confirms _build_tool is actually overridden, not just never consulted -- this would
        hang or fail without network/transformers if NLIFactCheckTool got constructed."""
        assert _critic_no_nli(CannedClient())._tool is None

    def test_critique_prompt_never_carries_tool_findings(self):
        client = CannedClient()
        _critic_no_nli(client).act(
            AgentState(example=_example(), history=(_draft_message(),), round_index=1)
        )
        prompt, _, system = client.calls[0]
        assert system is None
        assert "automatic fact-checker" not in prompt
        assert prompt == (
            f"{CriticNoNLI.SYSTEM_PROMPT}\n\n"
            "List the factual problems in the summary below, using only the article as "
            "evidence. Be brief and specific. If there are none, reply with exactly: No "
            "factual problems found.\n\n"
            "Article:\n"
            "The council voted to close the {old} library in Zürich.\n\n"
            "Summary:\n"
            "A flawed one-sentence summary."
        )

    def test_persona_and_template_are_the_same_as_critic(self):
        assert CriticNoNLI.SYSTEM_PROMPT == Critic.SYSTEM_PROMPT
        assert dict(_critic_no_nli(CannedClient()).prompt_templates) == {
            "xsum_critique": "xsum_critique@v1"
        }

    def test_message_role_is_still_critic_not_critic_no_nli(self):
        """The Generator's critique-revise branch keys off Message.role, not the AGENTS
        registry name -- this has to stay "critic" or CritiqueReviseNoNLI breaks."""
        message = _critic_no_nli(CannedClient()).act(
            AgentState(example=_example(), history=(_draft_message(),), round_index=1)
        )
        assert message.role == "critic"

    def test_is_registered_separately_from_critic(self):
        assert AGENTS.get("critic_no_nli") is CriticNoNLI
        assert AGENTS.get("critic") is Critic
        assert CriticNoNLI is not Critic


class TestCritiqueRevise:
    def test_three_calls_draft_critique_revision(self):
        client = CannedClient()
        tool = FakeTool()
        generator = _generator(client)
        critic = _critic(client, tool=tool)
        topology = CritiqueRevise(TopologyConfig(name="t2_critique_revise"))
        result = topology.run(_example(), {"generator": generator, "critic": critic})
        assert isinstance(result, TopologyResult)
        assert len(client.calls) == 3
        assert [m.role for m in result.trace] == ["generator", "critic", "generator"]
        assert [m.round_index for m in result.trace] == [0, 1, 2]
        assert result.token_cost == TokenCost(360, 54)
        assert result.n_rounds == 3 and result.converged is True
        assert result.output == "A one-sentence summary."

    def test_critic_receives_the_draft_and_generator_receives_the_critique(self):
        client = CannedClient(text="the model's fixed reply")
        tool = FakeTool()
        generator = _generator(client)
        critic = _critic(client, tool=tool)
        CritiqueRevise(TopologyConfig(name="t2_critique_revise")).run(
            _example(), {"generator": generator, "critic": critic}
        )
        # Round 1 (critic) is asked about round 0's draft.
        assert tool.calls == [
            ("The council voted to close the {old} library in Zürich.", "the model's fixed reply")
        ]
        # Round 2 (generator revise) is told about the critique.
        revise_prompt, _, _ = client.calls[2]
        assert "Critique:\nthe model's fixed reply" in revise_prompt

    def test_missing_generator_is_an_error(self):
        with pytest.raises(TopologyError, match="generator"):
            CritiqueRevise(TopologyConfig(name="t2_critique_revise")).run(
                _example(), {"critic": _critic(CannedClient())}
            )

    def test_missing_critic_is_an_error(self):
        with pytest.raises(TopologyError, match="critic"):
            CritiqueRevise(TopologyConfig(name="t2_critique_revise")).run(
                _example(), {"generator": _generator(CannedClient())}
            )


class TestCritiqueReviseNoNLI:
    """D-049: CritiqueRevise with CriticNoNLI in the critic slot. Everything but which Critic
    gets built is identical to TestCritiqueRevise -- this class exists to confirm the swap
    actually works end to end, not to re-test CritiqueRevise's own logic."""

    def test_three_calls_draft_critique_revision(self):
        client = CannedClient()
        generator = _generator(client)
        critic = _critic_no_nli(client)
        topology = CritiqueReviseNoNLI(TopologyConfig(name="t2_critique_revise_no_nli"))
        result = topology.run(_example(), {"generator": generator, "critic_no_nli": critic})
        assert isinstance(result, TopologyResult)
        assert len(client.calls) == 3
        # Message.role stays "critic" even though the registry/dict key is "critic_no_nli".
        assert [m.role for m in result.trace] == ["generator", "critic", "generator"]
        assert [m.round_index for m in result.trace] == [0, 1, 2]
        assert result.token_cost == TokenCost(360, 54)
        assert result.n_rounds == 3 and result.converged is True

    def test_critique_call_never_carries_tool_findings(self):
        client = CannedClient()
        generator = _generator(client)
        critic = _critic_no_nli(client)
        CritiqueReviseNoNLI(TopologyConfig(name="t2_critique_revise_no_nli")).run(
            _example(), {"generator": generator, "critic_no_nli": critic}
        )
        critique_prompt, _, _ = client.calls[1]
        assert "automatic fact-checker" not in critique_prompt

    def test_missing_generator_is_an_error(self):
        with pytest.raises(TopologyError, match="generator"):
            CritiqueReviseNoNLI(TopologyConfig(name="t2_critique_revise_no_nli")).run(
                _example(), {"critic_no_nli": _critic_no_nli(CannedClient())}
            )

    def test_missing_critic_no_nli_is_an_error(self):
        with pytest.raises(TopologyError, match="critic_no_nli"):
            CritiqueReviseNoNLI(TopologyConfig(name="t2_critique_revise_no_nli")).run(
                _example(), {"generator": _generator(CannedClient())}
            )


def test_critic_and_t2_are_registered():
    assert AGENTS.get("critic") is Critic
    assert TOPOLOGIES.get("t2_critique_revise") is CritiqueRevise
    assert AGENTS.get("critic_no_nli") is CriticNoNLI
    assert TOPOLOGIES.get("t2_critique_revise_no_nli") is CritiqueReviseNoNLI


def test_critique_revise_templates_cover_xsum_and_cnndm():
    assert set(CRITIQUE_TEMPLATES) == {"tier1_xsum", "tier1_cnndm"}
    assert set(CRITIQUE_REVISE_TEMPLATES) == {"tier1_xsum", "tier1_cnndm"}
    assert "three-sentence" in CRITIQUE_REVISE_TEMPLATES["tier1_cnndm"].instruction
    assert "one-sentence" in CRITIQUE_REVISE_TEMPLATES["tier1_xsum"].instruction


class TestNLIToolLogic:
    """Pure functions only -- no model, no transformers, CPU and instant."""

    def test_rank_entailment_picks_the_highest_entailment_row(self):
        probs = np.array(
            [
                [0.9, 0.05, 0.05],  # contradiction, entailment index 2 low
                [0.1, 0.1, 0.8],  # highest entailment
                [0.3, 0.6, 0.1],
            ]
        )
        assert rank_entailment(probs, entailment_index=2) == 1

    def test_classify_claim_returns_none_when_entailed(self):
        row = np.array([0.05, 0.05, 0.9])
        assert classify_claim("s", row, entailment_index=2, contradiction_index=0) is None

    def test_classify_claim_flags_contradiction(self):
        row = np.array([0.8, 0.1, 0.1])
        claim = classify_claim("the sky is green", row, entailment_index=2, contradiction_index=0)
        assert claim == WeakClaim(
            sentence="the sky is green", label="CONTRADICTION", best_entailment_prob=0.1
        )

    def test_classify_claim_flags_neutral(self):
        row = np.array([0.1, 0.8, 0.1])
        claim = classify_claim("unrelated claim", row, entailment_index=2, contradiction_index=0)
        assert claim == WeakClaim(
            sentence="unrelated claim", label="NEUTRAL", best_entailment_prob=0.1
        )
