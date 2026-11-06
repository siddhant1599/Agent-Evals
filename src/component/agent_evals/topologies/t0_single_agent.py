"""T0: the single-agent baseline. One Generator call per example, no critique, no second round."""

from __future__ import annotations

import time

from ..registry import TOPOLOGIES
from ..types import AgentSet, AgentState, Example, TopologyResult
from .base import Topology

__all__ = ["SingleAgent"]


@TOPOLOGIES.register("t0_single_agent")
class SingleAgent(Topology):
    """The zero-shot baseline every multi-agent topology is compared against.

    ``n_rounds`` is 1 and ``converged`` is True: it runs one round and is done. Its Generator
    call carries no role preface (D-016, restored by D-048): T2 folds one into its prompt, T0
    does not.
    """

    ROLES = ("generator",)

    def run(self, example: Example, agents: AgentSet) -> TopologyResult:
        """Ask the Generator once and return its message as the output.

        Args:
            example: The item under work.
            agents: Must hold ``generator``.

        Returns:
            The Generator's text, its token cost, and wall-clock time measured around the call.

        Raises:
            TopologyError: If there is no Generator.
            ServingError: If the model call fails.
        """
        self._require_roles(agents)
        started = time.perf_counter()
        message = agents["generator"].act(
            AgentState(example=example, history=(), round_index=0, include_role_preface=False)
        )
        return self._assemble(message, (message,), n_rounds=1, converged=True, started=started)
