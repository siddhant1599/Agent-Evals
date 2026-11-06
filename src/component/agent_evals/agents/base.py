"""The agent-role interface: a role is its prompt template and its tool access, not a topology."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from ..config import AgentConfig
from ..serving.base import ChatClient, Completion
from ..types import AgentState, Message

__all__ = ["Agent", "AgentError", "VersionedTemplate"]


class AgentError(ValueError):
    """Raised when a role cannot be built for the requested configuration, or cannot act on a
    history it was not built to answer."""


@dataclass(frozen=True, slots=True)
class VersionedTemplate:
    """Name and version shared by every prompt template; the version lands in the manifest.

    Changing a template's wording or examples changes results, so it always comes with a new
    ``version``.

    Attributes:
        name: Stable template name.
        version: Version string recorded in the run manifest.
    """

    name: str
    version: str

    @property
    def label(self) -> str:
        """``name@version``, as recorded in the manifest."""
        return f"{self.name}@{self.version}"


class Agent(ABC):
    """Base for every role: Generator, Critic, Planner, Executor, Verifier, Judge.

    An agent holds one ``ChatClient`` and never branches on which backbone is behind it.

    Args:
        cfg: What the role binds once: the dataset its prompt is for, and the request seed.
        client: The serving client every model call goes through.
    """

    ROLE: ClassVar[str]

    #: Fixed persona/instruction for this role, folded into the user prompt of every call that
    #: opts in (D-048; supersedes D-044's separate system message -- no call ever sends one,
    #: matching T0's own serving calls exactly). What to do right now is still the per-call user
    #: message a prompt template renders.
    SYSTEM_PROMPT: ClassVar[str]

    def __init__(self, cfg: AgentConfig, client: ChatClient) -> None:
        self._client = client
        self._seed = cfg.seed

    @property
    @abstractmethod
    def prompt_templates(self) -> Mapping[str, str]:
        """Template name mapped to ``name@version`` for every prompt this role sends.

        Recorded in each run's manifest, so a changed prompt is visible in its results.
        """

    @abstractmethod
    def act(self, state: AgentState) -> Message:
        """Produce one message for the current state.

        Args:
            state: The example, the messages so far and the current round.

        Returns:
            The message, with the call's token cost, latency and finish reason.

        Raises:
            ServingError: If the model call fails.
        """

    def _message(self, state: AgentState, completion: Completion) -> Message:
        """Wrap one completion as this role's message for the current round, text unmodified."""
        return Message(
            role=self.ROLE,
            content=completion.text,
            round_index=state.round_index,
            token_cost=completion.token_cost,
            latency_s=completion.latency_s,
            finish_reason=completion.finish_reason,
        )
