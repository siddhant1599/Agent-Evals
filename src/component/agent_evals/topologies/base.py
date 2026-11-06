"""The topology interface: every topology, T0 through T5, returns the same ``TopologyResult``."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import ClassVar

from ..config import TopologyConfig
from ..types import AgentSet, Example, Message, TokenCost, TopologyResult

__all__ = ["Topology", "TopologyError"]


class TopologyError(ValueError):
    """Raised when a topology lacks a role or setting it needs, or builds an inconsistent result."""


class Topology(ABC):
    """Base for every topology.

    Args:
        cfg: What the topology binds once.
    """

    #: Roles this topology calls; the runner builds exactly these.
    ROLES: ClassVar[tuple[str, ...]]

    def __init__(self, cfg: TopologyConfig) -> None:
        self._name = cfg.name

    @abstractmethod
    def run(self, example: Example, agents: AgentSet) -> TopologyResult:
        """Produce the final output for one example.

        Args:
            example: The item under work.
            agents: Role name mapped to the agent filling it.

        Returns:
            The output with total token cost, wall-clock time, rounds, convergence and trace.

        Raises:
            TopologyError: If a needed role is missing.
            ServingError: If a model call fails.
        """

    def _require_roles(self, agents: AgentSet) -> None:
        missing = [role for role in self.ROLES if role not in agents]
        if missing:
            raise TopologyError(f"topology {self._name!r} needs roles {missing}")

    def _assemble(
        self,
        output_message: Message,
        trace: tuple[Message, ...],
        n_rounds: int,
        converged: bool,
        started: float,
    ) -> TopologyResult:
        """Build the result, so timing, cost and the output's finish reason are computed once.

        Args:
            output_message: The message whose text is the output.
            trace: Every message produced, oldest first.
            n_rounds: Rounds actually executed.
            converged: Whether the loop stopped on its own signal rather than its cap.
            started: ``time.perf_counter()`` read before the first model call.

        Returns:
            The result, with wall-clock time measured up to now and token cost summed over every
            message in ``trace``.

        Raises:
            TopologyError: If ``output_message`` is not in ``trace``.
        """
        if output_message not in trace:
            raise TopologyError(f"topology {self._name!r}: the output message is not in the trace")
        return TopologyResult(
            output=output_message.content,
            token_cost=sum((message.token_cost for message in trace), start=TokenCost.zero()),
            wall_clock_s=time.perf_counter() - started,
            n_rounds=n_rounds,
            converged=converged,
            trace=trace,
            finish_reason=output_message.finish_reason,
        )
