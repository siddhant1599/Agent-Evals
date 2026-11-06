"""What agent roles and topologies bind at construction."""

from __future__ import annotations

from dataclasses import dataclass

from .validation import ConfigError


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """What one agent role binds at construction.

    Built by the benchmark runner from a resolved ``ExperimentConfig`` rather than read from YAML:
    each value is already decided elsewhere, and repeating it in a file would let the two drift.

    Attributes:
        dataset: Registered dataset name. Selects the role's prompt template (D-016).
        seed: Per-request sampling seed sent with every model call; the run's seed (D-003).
    """

    dataset: str
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise ConfigError("agent dataset must be a non-empty string")
        if type(self.seed) is not int or self.seed < 0:
            raise ConfigError(f"agent seed must be a non-negative integer, got {self.seed!r}")


@dataclass(frozen=True, slots=True)
class TopologyConfig:
    """What a topology binds at construction.

    Attributes:
        name: Registered topology name, as selected by ``--topology``.
        max_rounds: Maximum feedback-to-refine iterations for a multi-round topology; None for a
            single-pass topology. A multi-round topology refuses to run without it. Safe range:
            None or >= 1.
        requires_adapter: Whether the topology must be served with a LoRA adapter for the
            backbone (``serving.lora_adapters``). True only for T5, so that T5 can never run as
            the base model and be mistaken for T0 (D-111).
    """

    name: str
    max_rounds: int | None = None
    requires_adapter: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("topology name must be a non-empty string")
        rounds = self.max_rounds
        if rounds is not None and (type(rounds) is not int or rounds < 1):
            raise ConfigError(
                f"topology.max_rounds must be None or an integer >= 1, got {rounds!r}"
            )
        if not isinstance(self.requires_adapter, bool):
            raise ConfigError("topology.requires_adapter must be true or false")
