"""T1: Self-Refine (Madaan et al. 2023). One model drafts a summary, writes feedback on it, and
rewrites it from that feedback, until the feedback finds nothing to fix or the round budget is
spent (D-101, D-102)."""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from ..config import TopologyConfig
from ..registry import TOPOLOGIES
from ..types import AgentSet, AgentState, Example, Message, SupportsAct, TopologyResult
from .base import Topology, TopologyError

__all__ = ["SelfRefine"]


@runtime_checkable
class _SupportsStop(SupportsAct, Protocol):
    """An agent whose messages can end the loop: T1's self-feedback role."""

    def is_stop(self, message: Message) -> bool:
        """Whether ``message`` is the stop signal."""
        ...


@TOPOLOGIES.register("t1_self_refine")
class SelfRefine(Topology):
    """Self-Refine's Algorithm 1 (Madaan et al. 2023, §2), with a final check on the last summary.

    The Generator drafts with the unchanged T0 prompt (D-016), so the first draft is the T0
    baseline. Every call sets ``include_role_preface=False``: T1 adds no persona line (D-105).
    Self-feedback then checks the latest summary against the article, seeing that summary only
    (Eq. 2). Unless it replies with the stop word, the Generator rewrites the summary
    from the whole history so far (Eq. 4), and feedback checks the new summary. After
    ``max_rounds`` rewrites the last summary still gets its feedback check, so a run makes at most
    ``2 * max_rounds + 2`` model calls. The output is the last summary, never feedback (D-101).

    A summary and the feedback on it share one ``round_index``, starting at 0 for the draft.
    ``n_rounds`` is the number of summaries written, from 1 to ``max_rounds + 1``. ``converged``
    is True when some feedback was the stop word, False when the budget ran out without one.

    Args:
        cfg: Must set ``max_rounds`` (``src/configs/topology/t1_self_refine.yaml``, D-102).

    Raises:
        TopologyError: If ``max_rounds`` is unset.
    """

    ROLES = ("generator", "self_feedback")

    def __init__(self, cfg: TopologyConfig) -> None:
        super().__init__(cfg)
        if cfg.max_rounds is None:
            raise TopologyError(f"topology {self._name!r} needs topology.max_rounds (D-102)")
        self._max_rounds = cfg.max_rounds

    def run(self, example: Example, agents: AgentSet) -> TopologyResult:
        """Draft, then alternate feedback and rewrites until the stop word or the budget.

        Args:
            example: The item under work.
            agents: Must hold ``generator`` and ``self_feedback``; the latter must offer
                ``is_stop``.

        Returns:
            The last summary, the token cost and wall-clock time of every call, and the full
            trace: draft, feedback, rewrite, feedback, and so on, ending in feedback.

        Raises:
            TopologyError: If a role is missing or the feedback role cannot signal a stop.
            ServingError: If a model call fails.
        """
        self._require_roles(agents)
        generator, feedback_role = agents["generator"], agents["self_feedback"]
        if not isinstance(feedback_role, _SupportsStop):
            raise TopologyError(f"topology {self._name!r} needs a self_feedback role with is_stop")
        started = time.perf_counter()
        summary = generator.act(
            AgentState(example=example, history=(), round_index=0, include_role_preface=False)
        )
        trace: list[Message] = [summary]
        converged = False
        for round_index in range(self._max_rounds + 1):
            feedback = feedback_role.act(
                AgentState(
                    example=example,
                    history=tuple(trace),
                    round_index=round_index,
                    include_role_preface=False,
                )
            )
            trace.append(feedback)
            if feedback_role.is_stop(feedback):
                converged = True
                break
            if round_index == self._max_rounds:
                break
            summary = generator.act(
                AgentState(
                    example=example,
                    history=tuple(trace),
                    round_index=round_index + 1,
                    include_role_preface=False,
                )
            )
            trace.append(summary)
        return self._assemble(summary, tuple(trace), summary.round_index + 1, converged, started)
