"""Data contracts shared across every module.

These are the objects that cross module boundaries: examples in, messages between agents,
a single result shape out. They live here rather than inside any one module because all six
extension points depend on them and none of them owns them.

Everything here is a frozen dataclass. A result that has been written to ``results/`` is evidence,
and evidence that analysis code can quietly edit is not evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "TokenCost",
    "Example",
    "Message",
    "AgentState",
    "AgentSet",
    "TopologyResult",
    "SupportsAct",
]


@dataclass(frozen=True, slots=True)
class TokenCost:
    """Prompt and completion tokens for one or more model calls.

    Kept as two numbers rather than one total on purpose. Prefill (prompt tokens) is parallel
    and cheap per token; decode (completion tokens) is sequential and expensive per token, so
    two topologies with the same total can have very different wall-clock cost. H4 asks which
    topology is worth its cost, and the split cannot be recovered after the fact without
    re-running the sweep.

    Attributes:
        prompt_tokens: Tokens consumed reading the prompt. Non-negative.
        completion_tokens: Tokens produced by the model. Non-negative.
    """

    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError(
                f"token counts must be non-negative, got prompt={self.prompt_tokens} "
                f"completion={self.completion_tokens}"
            )

    @property
    def total(self) -> int:
        """Total tokens across prompt and completion."""
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenCost) -> TokenCost:
        """Sum two costs, so per-message costs aggregate into a topology total."""
        if not isinstance(other, TokenCost):
            return NotImplemented
        return TokenCost(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    @classmethod
    def zero(cls) -> TokenCost:
        """The additive identity, for use as a sum() start value."""
        return cls(prompt_tokens=0, completion_tokens=0)

    def as_dict(self) -> dict[str, int]:
        """Flat dict for the metrics writer. Keys are stable forever."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total,
        }


@dataclass(frozen=True, slots=True)
class Example:
    """One evidence-paired item, identical in shape across all three tiers.

    ``evidence`` is what every factuality metric, Verifier tool call and RL reward checks
    against. When a source is truncated to fit a context window, ``evidence`` must be the
    truncated text the model actually saw -- scoring faithfulness against text the model was
    never shown measures nothing.

    Attributes:
        id: Stable identifier from the source dataset.
        tier: 1 (summarization), 2 (grounded QA) or 3 (held-out stress test).
        task: Task name, e.g. "summarization" or "grounded_qa".
        source: The document, dialogue or passage set given to the model.
        evidence: The grounding text every factuality check runs against.
        reference: The gold summary or answer.
        question: The question, for grounded QA. None for summarization.
        gold: Gold annotations a single reference string cannot hold, as a JSON object, for the
            metrics that read them (D-126); e.g. ASQA's short answers per sub-question and both
            human answers. None when the dataset has none, which is every Tier 1 dataset.
        source_words: Whitespace word count of the original source, computed on construction.
        truncated: Whether serving shortened the evidence. Source retains the original text.
    """

    id: str
    tier: int
    task: str
    source: str
    evidence: str
    reference: str
    question: str | None = None
    gold: str | None = None
    source_words: int = field(init=False)
    truncated: bool = False

    def __post_init__(self) -> None:
        if self.tier not in (1, 2, 3):
            raise ValueError(f"tier must be 1, 2 or 3, got {self.tier!r}")
        if not self.id:
            raise ValueError("example id must be non-empty")
        if not isinstance(self.source, str):
            raise TypeError("example source must be a string")
        if self.gold is not None:
            if not isinstance(self.gold, str):
                raise TypeError("example gold must be a JSON string or None")
            try:
                parsed = json.loads(self.gold)
            except json.JSONDecodeError as exc:
                raise ValueError(f"example gold is not valid JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("example gold must be a JSON object")
        if not isinstance(self.truncated, bool):
            raise TypeError("example truncated flag must be a boolean")
        object.__setattr__(self, "source_words", len(self.source.split()))


@dataclass(frozen=True, slots=True)
class Message:
    """One utterance produced by one agent role.

    Carries its own cost so a topology total is the sum of its parts rather than a separately
    maintained counter that can drift from the trace.

    Attributes:
        role: The agent role that produced this, e.g. "generator", "critic", "judge".
        content: The text produced.
        round_index: Which round of the topology loop this belongs to. 0 for single-pass.
        token_cost: Tokens consumed and produced by this call.
        latency_s: Wall-clock seconds for this call. Non-negative.
        finish_reason: Why the model stopped, as the server reported it. ``"length"`` means the
            completion cap cut it off (D-012). None for a message no model call produced.
    """

    role: str
    content: str
    round_index: int
    token_cost: TokenCost
    latency_s: float
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError(f"round_index must be non-negative, got {self.round_index}")
        if self.latency_s < 0:
            raise ValueError(f"latency_s must be non-negative, got {self.latency_s}")


@dataclass(frozen=True, slots=True)
class AgentState:
    """What an agent sees when it is asked to act.

    Attributes:
        example: The item under work.
        history: Messages so far, oldest first. A tuple, so an agent cannot mutate the record.
        round_index: The round the topology is currently on.
        include_role_preface: Whether the calling topology wants this role's persona/instruction
            folded into the user prompt this round (D-048, delivered in-band -- no separate
            system message is ever sent, matching T0's own serving calls). Defaults to True; T0
            is the one exception (D-016's original zero-shot baseline, restored) and sets it
            False. Set by the topology, not the agent, since it is a policy choice about which
            protocol is running, not something a role decides about itself.
    """

    example: Example
    history: tuple[Message, ...]
    round_index: int
    include_role_preface: bool = True


@runtime_checkable
class SupportsAct(Protocol):
    """Anything a topology can call as an agent.

    Declared as a Protocol rather than importing the Agent base class so that ``types`` stays
    free of dependencies on ``agents/``. The concrete Agent ABC lives with the roles.
    """

    def act(self, state: AgentState) -> Message:
        """Produce one message given the current state."""
        ...


#: Role name -> the agent filling that role for this run. Homogeneous and heterogeneous
#: configurations differ only in which backbone each entry was built with; no topology ever
#: branches on that.
AgentSet = Mapping[str, SupportsAct]


@dataclass(frozen=True, slots=True)
class TopologyResult:
    """The single result shape returned by all six topologies, T0 through T5.

    T5 implements this identically to T0-T4; that identity is what makes H5 answerable, so it
    is a hard requirement rather than a convenience.

    Cost and latency are validated here rather than only at the writer, so an invalid result
    cannot be constructed in the first place. They are reported results, not telemetry.

    For a single-pass topology such as T0, ``n_rounds`` is 1 and ``converged`` is True: it ran
    one round and was done. Encoding that as None instead would push a null check into every
    consumer and put nulls in the master DataFrame.

    Attributes:
        output: The final text this topology produced.
        token_cost: Total tokens across every model call the topology made.
        wall_clock_s: Total wall-clock seconds. Non-negative.
        n_rounds: Rounds actually executed. At least 1.
        converged: Whether the loop stopped early on convergence rather than hitting its cap.
        trace: Every message produced, oldest first.
        finish_reason: Why the model stopped while writing ``output``; None if no model call
            produced it. Taken from the output's own message, not the last one in ``trace``,
            which can be a later check. Required, so a topology that does not set it fails.
    """

    output: str
    token_cost: TokenCost
    wall_clock_s: float
    n_rounds: int
    converged: bool
    trace: tuple[Message, ...]
    finish_reason: str | None

    def __post_init__(self) -> None:
        if self.wall_clock_s < 0:
            raise ValueError(f"wall_clock_s must be non-negative, got {self.wall_clock_s}")
        if self.n_rounds < 1:
            raise ValueError(f"n_rounds must be at least 1, got {self.n_rounds}")
        if not isinstance(self.trace, tuple):
            raise TypeError(
                f"trace must be a tuple so the result is immutable, got {type(self.trace).__name__}"
            )
