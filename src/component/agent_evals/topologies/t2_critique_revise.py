"""T2: Critique-Revise. Generator drafts; Critic scrutinizes it with the shared NLI tool;
Generator revises using the critique. Fixed at two summary iterations for now (D-043)."""

from __future__ import annotations

import time
from typing import ClassVar

from ..registry import TOPOLOGIES
from ..types import AgentSet, AgentState, Example, Message, TopologyResult
from .base import Topology

__all__ = ["CritiqueRevise", "CritiqueReviseNoNLI"]


@TOPOLOGIES.register("t2_critique_revise")
class CritiqueRevise(Topology):
    """Generator drafts, Critic critiques it, Generator revises using the critique.

    ``MAX_ITERATIONS`` counts Generator summaries: the draft is iteration 1, and a
    critique-then-revise pair is inserted before every iteration after that. Fixed at 2 for now
    (D-043) -- a one-line change to raise later -- and there is no convergence check.

    ``ROLES`` names the two registry entries this topology builds, in (generator, critic)
    order; ``CritiqueReviseNoNLI`` overrides it to swap in the no-tool Critic and inherits this
    ``run`` unchanged (D-049).
    """

    ROLES = ("generator", "critic")
    MAX_ITERATIONS: ClassVar[int] = 2

    def run(self, example: Example, agents: AgentSet) -> TopologyResult:
        """Draft, then alternate critique and revision until ``MAX_ITERATIONS`` is reached.

        Args:
            example: The item under work.
            agents: Must hold every role in ``self.ROLES``.

        Returns:
            The final revision's text, total token cost and wall-clock time across every call,
            and the full draft/critique/revision trace.

        Raises:
            TopologyError: If a needed role is missing.
            ServingError: If a model call fails.
        """
        self._require_roles(agents)
        generator_role, critic_role = self.ROLES
        generator, critic = agents[generator_role], agents[critic_role]
        started = time.perf_counter()
        trace: list[Message] = [
            generator.act(AgentState(example=example, history=(), round_index=0))
        ]
        round_index = 1
        for _ in range(self.MAX_ITERATIONS - 1):
            trace.append(
                critic.act(
                    AgentState(example=example, history=tuple(trace), round_index=round_index)
                )
            )
            round_index += 1
            trace.append(
                generator.act(
                    AgentState(example=example, history=tuple(trace), round_index=round_index)
                )
            )
            round_index += 1
        return self._assemble(
            trace[-1], tuple(trace), n_rounds=len(trace), converged=True, started=started
        )


@TOPOLOGIES.register("t2_critique_revise_no_nli")
class CritiqueReviseNoNLI(CritiqueRevise):
    """``CritiqueRevise`` with the Critic's NLI tool access removed (D-049).

    The Critic reads the article and the draft itself and critiques from context, with no
    deterministic checker's findings spliced into its prompt. Draft, critique, revise, fixed at
    two iterations -- everything but which Critic is built is identical to ``CritiqueRevise``,
    whose ``run`` this inherits unchanged.
    """

    ROLES = ("generator", "critic_no_nli")
